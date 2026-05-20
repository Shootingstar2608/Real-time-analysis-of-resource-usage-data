#!/usr/bin/env bash
# =============================================================
# run_pipeline.sh — Khởi động toàn bộ pipeline real-time
# Mở 3 terminal riêng biệt:
#   Terminal 1: Spark Streaming Consumer
#   Terminal 2: Kafka Producer
#   Terminal 3: Streamlit Dashboard
# Usage:
#   bash scripts/run_pipeline.sh
# =============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ── Kiểm tra Docker / Kafka ──────────────────────────────────
echo "🔍  Kiểm tra Docker containers..."
if ! docker compose ps 2>/dev/null | grep -q "kafka.*Up"; then
    echo "⚠️   Kafka chưa chạy. Khởi động docker-compose..."
    docker compose up -d
    echo "⏳  Đợi Kafka sẵn sàng (30s)..."
    sleep 30
fi

# ── Kiểm tra models đã train chưa ───────────────────────────
IF_MODEL="$PROJECT_ROOT/models/saved/isolation_forest.joblib"
if [ ! -f "$IF_MODEL" ]; then
    echo "⚠️   Models chưa được train. Chạy train trước..."
    bash "$PROJECT_ROOT/scripts/train_models.sh"
fi

echo ""
echo "======================================================"
echo "  Khởi động Pipeline"
echo "======================================================"

# ── Terminal 1: Spark Streaming Consumer ────────────────────
echo "🚀  [1] Spark Streaming Consumer → $LOG_DIR/spark_consumer.log"
nohup $PYTHON spark_processing/streaming_consumer.py \
    > "$LOG_DIR/spark_consumer.log" 2>&1 &
SPARK_PID=$!
echo "    PID: $SPARK_PID"

echo "⏳  Đợi Spark khởi động (15s)..."
sleep 15

# ── Terminal 2: Kafka Producer ──────────────────────────────
echo "📡  [2] Kafka Producer → $LOG_DIR/producer.log"
nohup $PYTHON kafka_stream/real_data_producer.py \
    --speed 1000 \
    > "$LOG_DIR/producer.log" 2>&1 &
PRODUCER_PID=$!
echo "    PID: $PRODUCER_PID"

# ── Terminal 3: Streamlit Dashboard ─────────────────────────
echo "📊  [3] Streamlit Dashboard → http://localhost:8501"
nohup streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    > "$LOG_DIR/dashboard.log" 2>&1 &
DASH_PID=$!
echo "    PID: $DASH_PID"

# ── Lưu PIDs để dễ dừng ─────────────────────────────────────
echo "$SPARK_PID $PRODUCER_PID $DASH_PID" > "$LOG_DIR/pipeline.pid"

echo ""
echo "✅  Pipeline đang chạy!"
echo "   Spark log:     tail -f $LOG_DIR/spark_consumer.log"
echo "   Producer log:  tail -f $LOG_DIR/producer.log"
echo "   Dashboard:     http://localhost:8501"
echo "   Kafka UI:      http://localhost:8080"
echo ""
echo "⛔  Để dừng pipeline: bash scripts/stop_pipeline.sh"
