#!/usr/bin/env python3
"""Evaluate LightGBM model on production Feature Store data."""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])

import argparse, json, os, tarfile, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",        required=True)
    p.add_argument("--fg-prod",       required=True)
    p.add_argument("--target-recall", type=float, default=0.70)
    p.add_argument("--region",        default="us-east-1")
    p.add_argument("--force-retrain",      default="false")
    p.add_argument("--force-skip-retrain", default="false")
    args = p.parse_args()

    # Load constraints
    cdir = "/opt/ml/processing/input/constraints"
    with open(os.path.join(cdir, os.listdir(cdir)[0])) as f:
        constraints = json.load(f)

    # Load model
    mdir = "/opt/ml/processing/input/model"
    tar_path = os.path.join(mdir, [f for f in os.listdir(mdir) if f.endswith(".tar.gz")][0])
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("/tmp/model")
    model = lgb.Booster(model_file="/tmp/model/lgb_model.txt")
    with open("/tmp/model/model_metadata.json") as f:
        threshold = json.load(f)["threshold"]

    # Load production data from Feature Store
    import sagemaker
    from sagemaker.feature_store.feature_group import FeatureGroup
    fg = FeatureGroup(name=args.fg_prod, sagemaker_session=sagemaker.Session())
    q = fg.athena_query()
    q.run(f'SELECT * FROM "{q.table_name}"',
          output_location=f"s3://{args.bucket}/athena-results/pipeline/")
    q.wait()
    df = q.as_dataframe()

    feature_cols = constraints["feature_cols"]
    X = df[feature_cols].astype(float).values
    y = df["failure"].astype(int).values

    probs = model.predict(X)
    preds = (probs >= threshold).astype(int)
    tp = int(((preds==1)&(y==1)).sum())
    fp = int(((preds==1)&(y==0)).sum())
    fn = int(((preds==0)&(y==1)).sum())
    tn = int(((preds==0)&(y==0)).sum())
    recall    = tp / max(tp+fn, 1)
    precision = tp / max(tp+fp, 1)
    both = len(np.unique(y)) > 1
    roc_auc  = float(roc_auc_score(y, probs)) if both else 0.0
    avg_prec = float(average_precision_score(y, probs)) if both else 0.0

    retrain, reasons = False, []
    if recall < args.target_recall:
        retrain = True
        reasons.append(f"Recall {recall:.4f} < {args.target_recall}")

    bl = constraints.get("baselines", {})
    deg = sum([fp > bl.get("max_false_positives", float("inf")),
               roc_auc < bl.get("min_roc_auc", 0),
               avg_prec < bl.get("min_average_precision", 0)])
    if deg >= 2:
        retrain = True
        reasons.append(f"{deg} secondary metrics regressed")

    nulls = [c for c in feature_cols if float(df[c].isnull().mean()) > 0.05]
    if nulls:
        retrain = True
        reasons.append(f"High null rate: {nulls}")

    # ── Apply override parameters ────────────────────────────────────────────
    force_retrain = args.force_retrain.lower() == "true"
    force_skip    = args.force_skip_retrain.lower() == "true"
    if force_retrain and force_skip:
        retrain = False
        reasons = ["Suppressed by ForceSkipRetrain (conflict with ForceRetrain)"]
    elif force_retrain:
        retrain = True
        reasons.append("ForceRetrain override enabled")
    elif force_skip:
        retrain = False
        reasons = ["Suppressed by ForceSkipRetrain override"]

    result = dict(status="RETRAIN" if retrain else "HEALTHY",
                  recall=recall, precision=precision, roc_auc=roc_auc,
                  avg_precision=avg_prec, tp=tp, fp=fp, fn=fn, tn=tn,
                  degraded=int(deg), reasons=reasons, samples=len(y),
                  force_retrain=force_retrain, force_skip_retrain=force_skip)

    out = "/opt/ml/processing/output"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "evaluation.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
