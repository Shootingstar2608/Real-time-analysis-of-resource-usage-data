#!/usr/bin/env bash
# ============================================
# Download small Azure VM Traces subset
# - vmtable.csv.gz
# - first N vm_cpu_readings files
# Default: N = 4
# Usage:
#   bash data/download_data.sh
#   bash data/download_data.sh 2
#   bash data/download_data.sh 4
# ============================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/raw"
PROCESSED_DIR="${PROJECT_ROOT}/data/processed"

mkdir -p "${DATA_DIR}"
mkdir -p "${PROCESSED_DIR}"

BASE_URL="https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2"
CPU_FILE_COUNT="${1:-4}"

if ! [[ "${CPU_FILE_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] CPU_FILE_COUNT must be an integer."
    echo "Example: bash data/download_data.sh 4"
    exit 1
fi

if [ "${CPU_FILE_COUNT}" -lt 1 ]; then
    echo "[ERROR] CPU_FILE_COUNT must be >= 1"
    exit 1
fi

if [ "${CPU_FILE_COUNT}" -gt 5 ]; then
    echo "[WARN] You asked for ${CPU_FILE_COUNT} files."
    echo "[WARN] For this project/demo, 2-4 files is usually enough."
fi

download_file() {
    local url="$1"
    local output="$2"

    if [ -f "${output}" ]; then
        echo "  [SKIP] $(basename "${output}") already exists"
        return 0
    fi

    echo "  [GET ] $(basename "${output}")"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --retry-delay 2 -o "${output}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "${output}" "${url}"
    else
        echo "[ERROR] Neither curl nor wget is installed."
        exit 1
    fi
}

echo "============================================"
echo " Azure VM Traces 2019 - Small Downloader"
echo "============================================"
echo "Project root : ${PROJECT_ROOT}"
echo "Data dir     : ${DATA_DIR}"
echo "CPU files    : ${CPU_FILE_COUNT}"
echo ""

echo "[1/2] Downloading VM table..."
download_file \
    "${BASE_URL}/trace_data/vmtable/vmtable.csv.gz" \
    "${DATA_DIR}/vmtable.csv.gz"

echo ""
echo "[2/2] Downloading VM CPU reading files..."
for i in $(seq 1 "${CPU_FILE_COUNT}"); do
    FILE_NAME="vm_cpu_readings-file-${i}-of-195.csv.gz"
    FILE_URL="${BASE_URL}/trace_data/vm_cpu_readings/${FILE_NAME}"
    download_file "${FILE_URL}" "${DATA_DIR}/${FILE_NAME}"
done

echo ""
echo "Verifying downloaded files..."
echo "--------------------------------------------"
find "${DATA_DIR}" -maxdepth 1 -type f -name "*.csv.gz" -print | sort || true
echo "--------------------------------------------"
du -sh "${DATA_DIR}" || true

echo ""
echo "Done."
echo "You can now train the LSTM model with:"
echo "  python ml_models/lstm_models.py"