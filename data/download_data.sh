#!/bin/bash
# ============================================
# Download Azure Public Dataset (VM Traces 2019)
# Source: https://github.com/Azure/AzurePublicDataset
# ============================================

set -e

DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"

mkdir -p "$DATA_DIR"
mkdir -p "$PROCESSED_DIR"

echo "============================================"
echo "  Azure VM Traces 2019 - Data Downloader"
echo "============================================"
echo ""
echo "This dataset contains VM workload traces from"
echo "Microsoft Azure datacenters."
echo ""
echo "Source: https://github.com/Azure/AzurePublicDataset"
echo ""

# ============================================
# Download VM CPU readings (main dataset)
# ============================================
BASE_URL="https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2"

echo "[1/3] Downloading VM Table (metadata)..."
echo "  This file contains VM metadata (core count, memory, category)"

# vmtable file
wget -c "${BASE_URL}/vmtable.csv.gz" -O "${DATA_DIR}/vmtable.csv.gz" 2>/dev/null || \
curl -L -C - "${BASE_URL}/vmtable.csv.gz" -o "${DATA_DIR}/vmtable.csv.gz"

echo "[2/3] Downloading VM CPU readings..."
echo "  NOTE: Full dataset is ~35GB. We download a subset (~5-8GB)"
echo "  You can adjust the range below (files 0-50 for ~5GB)"

# Download subset of CPU reading files (each ~100MB)
for i in $(seq 0 50); do
    PADDED=$(printf "%04d" $i)
    FILE="vm_cpu_readings-file-${PADDED}-of-0195.csv.gz"
    
    if [ -f "${DATA_DIR}/${FILE}" ]; then
        echo "  [SKIP] ${FILE} already exists"
        continue
    fi
    
    echo "  [${i}/50] Downloading ${FILE}..."
    wget -q "${BASE_URL}/${FILE}" -O "${DATA_DIR}/${FILE}" 2>/dev/null || \
    curl -sL "${BASE_URL}/${FILE}" -o "${DATA_DIR}/${FILE}"
done

echo "[3/3] Downloading complete!"
echo ""

# ============================================
# Verify downloads
# ============================================
echo "Verifying downloads..."
TOTAL_SIZE=$(du -sh "${DATA_DIR}" | cut -f1)
FILE_COUNT=$(ls -1 "${DATA_DIR}"/*.csv.gz 2>/dev/null | wc -l)

echo "  Total files: ${FILE_COUNT}"
echo "  Total size: ${TOTAL_SIZE}"
echo ""
echo "============================================"
echo "  Download complete!"
echo "  Run 'python data/prepare_data.py' to process the data"
echo "============================================"
