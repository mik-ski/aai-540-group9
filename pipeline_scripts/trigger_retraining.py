#!/usr/bin/env python3
"""Save retraining trigger + monitoring result to S3."""
import argparse, json, os, boto3
from datetime import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",        required=True)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region",        default="us-east-1")
    args = p.parse_args()

    with open("/opt/ml/processing/input/evaluation/evaluation.json") as f:
        ev = json.load(f)

    s3 = boto3.client("s3", region_name=args.region)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    body = json.dumps(dict(timestamp=ts, endpoint=args.endpoint_name,
                           status="RETRAIN", evaluation=ev), indent=2)
    for k in [f"monitoring/retrain-triggers/{ts}.json",
              f"monitoring/runs/{ts}.json"]:
        s3.put_object(Bucket=args.bucket, Key=k, Body=body,
                      ContentType="application/json")
    print(f"Retrain trigger saved for {args.endpoint_name}")

if __name__ == "__main__":
    main()
