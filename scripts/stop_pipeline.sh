#!/usr/bin/env bash
# stop_pipeline.sh — Dừng toàn bộ pipeline
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/logs/pipeline.pid"

if [ -f "$PID_FILE" ]; then
    read -r PIDS < "$PID_FILE"
    for pid in $PIDS; do
        kill "$pid" 2>/dev/null && echo "Stopped PID $pid" || echo "PID $pid not running"
    done
    rm -f "$PID_FILE"
else
    echo "No pipeline.pid found. Killing by process name..."
    pkill -f "streaming_consumer.py" 2>/dev/null || true
    pkill -f "real_data_producer.py" 2>/dev/null || true
    pkill -f "streamlit run dashboard" 2>/dev/null || true
fi

echo "✅  Pipeline stopped."
