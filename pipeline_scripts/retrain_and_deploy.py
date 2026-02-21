#!/usr/bin/env python3
"""
Retrain LightGBM on training Feature Store data, package, deploy to the
existing SageMaker endpoint, and register in Model Registry.
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "lightgbm", "sagemaker"])

import argparse, json, os, tarfile, shutil, time, boto3, numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",              required=True)
    p.add_argument("--endpoint-name",       required=True)
    p.add_argument("--fg-train",            required=True)
    p.add_argument("--fg-val",              required=True)
    p.add_argument("--target-recall",       type=float, default=0.70)
    p.add_argument("--role",                required=True)
    p.add_argument("--model-package-group", required=True)
    p.add_argument("--region",              default="us-east-1")
    args = p.parse_args()

    import sagemaker
    from sagemaker.feature_store.feature_group import FeatureGroup
    from sagemaker.sklearn import SKLearnModel
    from sagemaker.model_monitor import DataCaptureConfig

    sess = sagemaker.Session()
    s3   = boto3.client("s3", region_name=args.region)
    sm   = boto3.client("sagemaker", region_name=args.region)
    ts   = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    # ── 1. Load training and validation data from Feature Store ─────────────
    print("Loading training data from Feature Store...")
    feature_cols = ["pct_one_star", "pct_two_star", "smart_5_raw",
                    "smart_187_raw", "smart_188_raw", "smart_197_raw",
                    "smart_198_raw"]

    def load_fg(name):
        fg = FeatureGroup(name=name, sagemaker_session=sess)
        q = fg.athena_query()
        q.run(f'SELECT * FROM "{q.table_name}"',
              output_location=f"s3://{args.bucket}/athena-results/pipeline/")
        q.wait()
        return q.as_dataframe()

    df_train = load_fg(args.fg_train)
    df_val   = load_fg(args.fg_val)

    X_train = df_train[feature_cols].astype(float)
    y_train = df_train["failure"].astype(int)
    X_val   = df_val[feature_cols].astype(float)
    y_val   = df_val["failure"].astype(int)

    print(f"  Train: {X_train.shape}, positives: {y_train.sum()}")
    print(f"  Val:   {X_val.shape}, positives: {y_val.sum()}")

    # ── 2. Train LightGBM ───────────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = float(np.sqrt(n_neg / max(n_pos, 1)))

    params = dict(
        objective="binary", metric="auc", boosting_type="gbdt",
        scale_pos_weight=scale_pos, learning_rate=0.01, num_leaves=15,
        max_depth=4, min_child_samples=10, min_child_weight=1e-3,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
        max_bin=127, verbose=-1, seed=42, force_row_wise=True,
    )

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    model = lgb.train(params, lgb_train, num_boost_round=3000,
                      valid_sets=[lgb_val], valid_names=["val"],
                      callbacks=[lgb.early_stopping(300), lgb.log_evaluation(200)])

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best val AUC:   {model.best_score['val']['auc']:.4f}")

    # ── 3. Tune threshold for target recall ──────────────────────────────────
    y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
    precs, recs, threshs = precision_recall_curve(y_val, y_val_prob)

    candidates = []
    for pr, rc, th in zip(precs[:-1], recs[:-1], threshs):
        if rc >= args.target_recall:
            fp_cnt = int(((y_val_prob >= th).astype(int) & (y_val == 0).values).sum())
            candidates.append((th, rc, pr, fp_cnt))
    if candidates:
        candidates.sort(key=lambda x: (x[3], -x[1]))
        best_thresh = candidates[0][0]
    else:
        idx = np.argmin(np.abs(recs[:-1] - args.target_recall))
        best_thresh = float(threshs[idx])

    print(f"  Threshold: {best_thresh:.6f}")

    # ── 4. Package model artifacts ───────────────────────────────────────────
    work = "/tmp/retrain_artifacts"
    os.makedirs(work, exist_ok=True)

    model_file = os.path.join(work, "lgb_model.txt")
    model.save_model(model_file)

    meta = dict(threshold=float(best_thresh), target_recall=args.target_recall,
                feature_cols=feature_cols, best_iteration=model.best_iteration,
                scale_pos_weight=float(scale_pos),
                trained_date=datetime.utcnow().isoformat(),
                retrained_by="sagemaker-pipeline")
    meta_file = os.path.join(work, "model_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    # Copy inference.py from processing input
    inf_src = "/opt/ml/processing/input/inference/inference.py"
    inf_dst = os.path.join(work, "inference.py")
    if os.path.exists(inf_src):
        shutil.copy2(inf_src, inf_dst)
    else:
        raise FileNotFoundError("inference.py not provided as processing input")

    tar_path = os.path.join(work, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_file, arcname="lgb_model.txt")
        tar.add(meta_file,  arcname="model_metadata.json")

    model_s3_key = f"models/lgb-hdd-failure/{ts}/model.tar.gz"
    s3.upload_file(tar_path, args.bucket, model_s3_key)
    model_s3_uri = f"s3://{args.bucket}/{model_s3_key}"
    print(f"  Model uploaded: {model_s3_uri}")

    # ── 5. Create SageMaker model + deploy to existing endpoint ──────────────
    model_name = f"lgb-hdd-failure-{ts}"

    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=args.role,
        entry_point=inf_dst,
        framework_version="1.2-1",
        py_version="py3",
        name=model_name,
        sagemaker_session=sess,
        dependencies=["/opt/ml/processing/input/inference/requirements.txt"],
    )

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=f"s3://{args.bucket}/monitoring/data-capture/{model_name}",
        capture_options=["Input", "Output"],
        csv_content_types=["text/csv"],
        json_content_types=["application/json"],
    )

    # Create the model in SageMaker
    container_def = sklearn_model.prepare_container_def(instance_type="ml.m5.large")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer=container_def,
        ExecutionRoleArn=args.role,
    )
    print(f"  SageMaker model created: {model_name}")

    # Create new endpoint config with data capture
    ep_config_name = f"{model_name}-config"
    sm.create_endpoint_config(
        EndpointConfigName=ep_config_name,
        ProductionVariants=[dict(
            VariantName="AllTraffic",
            ModelName=model_name,
            InstanceType="ml.m5.large",
            InitialInstanceCount=1,
            InitialVariantWeight=1.0,
        )],
        DataCaptureConfig=dict(
            EnableCapture=True,
            InitialSamplingPercentage=100,
            DestinationS3Uri=f"s3://{args.bucket}/monitoring/data-capture/{model_name}",
            CaptureOptions=[
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
            CaptureContentTypeHeader={
                "CsvContentTypes": ["text/csv"],
                "JsonContentTypes": ["application/json"],
            },
        ),
    )
    print(f"  Endpoint config created: {ep_config_name}")

    # Update the existing endpoint to use the new config (blue/green)
    sm.update_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=ep_config_name,
    )
    print(f"  Updating endpoint {args.endpoint_name} → {model_name} ...")

    # Wait for endpoint to finish updating
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=args.endpoint_name,
                WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print(f"  ✓ Endpoint updated and InService!")

    # ── 6. Register in Model Registry ────────────────────────────────────────
    # Get container image from the endpoint config
    ep_desc = sm.describe_endpoint_config(EndpointConfigName=ep_config_name)
    model_desc = sm.describe_model(ModelName=model_name)
    image_uri = model_desc["PrimaryContainer"]["Image"]

    # Compute test metrics for registry
    y_val_pred = (y_val_prob >= best_thresh).astype(int)
    tp = int(((y_val_pred==1)&(y_val.values==1)).sum())
    fp = int(((y_val_pred==1)&(y_val.values==0)).sum())
    fn = int(((y_val_pred==0)&(y_val.values==1)).sum())
    recall = tp / max(tp+fn, 1)
    precision = tp / max(tp+fp, 1)
    both = len(np.unique(y_val.values)) > 1
    roc_auc = float(roc_auc_score(y_val, y_val_prob)) if both else 0.0

    try:
        sm.create_model_package_group(
            ModelPackageGroupName=args.model_package_group,
            ModelPackageGroupDescription="HDD failure prediction models",
        )
    except sm.exceptions.ClientError:
        pass  # already exists

    pkg_resp = sm.create_model_package(
        ModelPackageGroupName=args.model_package_group,
        ModelPackageDescription=f"Retrained {ts} — recall={recall:.4f}",
        InferenceSpecification=dict(
            Containers=[dict(Image=image_uri, ModelDataUrl=model_s3_uri)],
            SupportedTransformInstanceTypes=["ml.m5.large"],
            SupportedRealtimeInferenceInstanceTypes=["ml.m5.large"],
            SupportedContentTypes=["application/json"],
            SupportedResponseMIMETypes=["application/json"],
        ),
        ModelApprovalStatus="Approved",
        ModelMetrics=dict(
            ModelQuality=dict(
                Statistics=dict(
                    ContentType="application/json",
                    S3Uri=model_s3_uri.replace("model.tar.gz",
                                               "evaluation_metrics.json"),
                ),
            ),
        ),
    )
    pkg_arn = pkg_resp["ModelPackageArn"]
    print(f"  Registered in Model Registry: {pkg_arn}")

    # ── 7. Save retrain report to S3 ─────────────────────────────────────────
    report = dict(
        timestamp=ts, status="RETRAINED", endpoint=args.endpoint_name,
        new_model=model_name, new_config=ep_config_name,
        model_s3_uri=model_s3_uri, model_package_arn=pkg_arn,
        threshold=float(best_thresh), val_recall=recall,
        val_precision=precision, val_roc_auc=roc_auc,
        best_iteration=model.best_iteration,
    )
    for key in [f"monitoring/retrain-triggers/{ts}.json",
                f"monitoring/runs/{ts}.json"]:
        s3.put_object(Bucket=args.bucket, Key=key,
                      Body=json.dumps(report, indent=2),
                      ContentType="application/json")

    # Save evaluation metrics alongside model
    eval_metrics = dict(roc_auc=roc_auc, recall=recall,
                        precision=precision, threshold=float(best_thresh),
                        best_iteration=model.best_iteration,
                        scale_pos_weight=float(scale_pos),
                        n_features=len(feature_cols),
                        train_samples=len(X_train))
    s3.put_object(
        Bucket=args.bucket,
        Key=model_s3_key.replace("model.tar.gz", "evaluation_metrics.json"),
        Body=json.dumps(eval_metrics, indent=2),
        ContentType="application/json",
    )

    # ── 8. Update .env with new model details ────────────────────────────────
    env_path = "/opt/ml/processing/input/inference/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
        # Remove old deployment keys
        remove_keys = {"ENDPOINT_NAME", "MODEL_NAME", "ENDPOINT_CONFIG_NAME",
                       "MODEL_S3_URI", "MODEL_THRESHOLD", "MODEL_PACKAGE_ARN"}
        cleaned = []
        skip = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# LightGBM Model Endpoint"):
                skip = True; continue
            if skip and stripped == "":
                skip = False; continue
            skip = False
            key = stripped.split("=", 1)[0] if "=" in stripped else ""
            if key in remove_keys:
                continue
            cleaned.append(line)
        while cleaned and cleaned[-1].strip() == "":
            cleaned.pop()
        cleaned.append(f"\n# LightGBM Model Endpoint (retrained {ts})\n")
        cleaned.append(f"ENDPOINT_NAME=\'{args.endpoint_name}\'\n")
        cleaned.append(f"MODEL_NAME=\'{model_name}\'\n")
        cleaned.append(f"ENDPOINT_CONFIG_NAME=\'{ep_config_name}\'\n")
        cleaned.append(f"MODEL_S3_URI=\'{model_s3_uri}\'\n")
        cleaned.append(f"MODEL_THRESHOLD={best_thresh:.6f}\n")
        cleaned.append(f"MODEL_PACKAGE_ARN=\'{pkg_arn}\'\n")
        # Upload updated .env to S3 for persistence
        s3.put_object(Bucket=args.bucket, Key="config/.env",
                      Body="".join(cleaned), ContentType="text/plain")
        print(f"  Updated .env uploaded to s3://{args.bucket}/config/.env")

    print(f"\n{'='*80}")
    print(f"RETRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"  New model:    {model_name}")
    print(f"  Endpoint:     {args.endpoint_name} (updated)")
    print(f"  Threshold:    {best_thresh:.6f}")
    print(f"  Val Recall:   {recall:.4f}")
    print(f"  Val ROC-AUC:  {roc_auc:.4f}")
    print(f"  Registry:     {pkg_arn}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
