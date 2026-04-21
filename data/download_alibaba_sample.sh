#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/alibaba"
SAMPLE_LINES="${1:-1000000}"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -f machine_usage.tar.gz ]; then
  wget -c http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz -O machine_usage.tar.gz
fi

if [ ! -f machine_meta.tar.gz ]; then
  wget -c http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_meta.tar.gz -O machine_meta.tar.gz || true
fi

tar -xOzf machine_usage.tar.gz | head -n "$SAMPLE_LINES" > machine_usage_sample.csv
if [ -f machine_meta.tar.gz ]; then
  tar -xOzf machine_meta.tar.gz > machine_meta.csv || true
fi

echo "Created: $DATA_DIR/machine_usage_sample.csv"
[ -f machine_meta.csv ] && echo "Created: $DATA_DIR/machine_meta.csv"
wc -l machine_usage_sample.csv
