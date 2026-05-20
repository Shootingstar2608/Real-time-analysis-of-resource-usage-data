#!/usr/bin/env bash
# =============================================================
# train_models.sh — Train tất cả ML models (IF + BiLSTM + Forecaster)
# Chạy 1 lần trước khi khởi động streaming pipeline.
# Usage:
#   cd /home/peter/Real-time-analysis-of-resource-usage-data
#   bash scripts/train_models.sh
# =============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"

# Kiểm tra file data
TAR_GZ="$PROJECT_ROOT/data/alibaba/machine_usage.tar.gz"
if [ ! -f "$TAR_GZ" ]; then
    echo "❌  File data không tìm thấy: $TAR_GZ"
    echo "    Tải về bằng lệnh:"
    echo "    wget -c http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz -P data/alibaba/"
    exit 1
fi

echo "======================================================"
echo "  [1/2] Train Isolation Forest"
echo "======================================================"
$PYTHON ml_models/isolation_forest_model.py

echo ""
echo "======================================================"
echo "  [2/2] Train BiLSTM Autoencoder + LSTM Forecaster"
echo "======================================================"
$PYTHON ml_models/lstm_models.py

echo ""
echo "✅  Tất cả models đã được train và lưu vào models/saved/"
ls -lh models/saved/
