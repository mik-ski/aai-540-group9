#!/usr/bin/env python3
"""Save healthy monitoring result to S3."""
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
                           status="HEALTHY", evaluation=ev), indent=2)
    s3.put_object(Bucket=args.bucket, Key=f"monitoring/runs/{ts}.json",
                  Body=body, ContentType="application/json")
    print(f"Healthy report saved for {args.endpoint_name}")

if __name__ == "__main__":
    main()
