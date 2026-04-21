# 📊 Real-time Analysis of Resource Usage Data

> Big Data Project: Kafka + Spark Streaming + ML/DL + Dashboard

This version supports **both Azure VM traces and Alibaba 2018 machine traces**.

## ✅ What changed

- Added **dataset auto-detection**: Azure or Alibaba
- Added **Alibaba `machine_usage` + `machine_meta` adapter**
- Normalized both datasets into one common schema
- Fixed dashboard ML-results path mismatch (`ml_results_lstm` vs `ml_results`)
- Extended Spark/Kafka payload schema with memory/network/disk fields
- Added **smoke tests** and **optional deep LSTM test**

---

## 📁 Project Structure

```text
Real-time-analysis-of-resource-usage-data-Phong_AI/
├── config/
├── data/
│   ├── dataset_utils.py              # Azure/Alibaba normalization
│   ├── download_data.sh              # Azure sample download
│   └── download_alibaba_sample.sh    # Alibaba sample download
├── kafka_stream/
│   └── real_data_producer.py         # Kafka producer for both datasets
├── ml_models/
│   └── lstm_models.py                # Dataset-aware training
├── spark_processing/
│   └── streaming_consumer.py         # Spark consumer (extended schema)
├── dashboard/
│   └── app.py                        # Dashboard with auto dataset loading
└── tests/
    ├── test_dataset_support.py
    ├── test_lstm_support.py
    └── run_smoke_tests.py
```

---

## 🚀 Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Azure flow

```bash
bash data/download_data.sh 1
python ml_models/lstm_models.py --dataset azure --data-dir data/raw
python kafka_stream/real_data_producer.py --dataset azure --data-dir data/raw --speed 100 --batch-size 500
streamlit run dashboard/app.py
```

### 3. Alibaba sample flow

Create a light sample first:

```bash
bash data/download_alibaba_sample.sh 1000000
```

Then train and stream from the sample:

```bash
python ml_models/lstm_models.py --dataset alibaba --data-dir data/alibaba --max-records 1000000
python kafka_stream/real_data_producer.py --dataset alibaba --data-dir data/alibaba --max-records 1000000 --speed 100 --batch-size 500
streamlit run dashboard/app.py
```

### 4. Run smoke tests

```bash
python tests/run_smoke_tests.py
```

Optional deeper training test:

```bash
RUN_TORCH_TRAINING_TESTS=1 python tests/run_smoke_tests.py
```

---

## 📦 Normalized schema

Both Azure and Alibaba records are converted into this common payload shape:

- `timestamp`
- `vm_id`
- `min_cpu`
- `max_cpu`
- `avg_cpu`
- `cpu_range`
- `vm_category`
- `vm_core_count`
- `vm_memory_gb`
- `avg_memory`
- `network_in_mbps`
- `network_out_mbps`
- `disk_io_percent`
- `mem_gps`
- `mkpi`
- `ingestion_timestamp`
- `source_file`
- `data_source`

For Alibaba `machine_usage`, `avg_cpu` is mapped from `cpu_util_percent`; `avg_memory`, network, disk, `mem_gps`, and `mkpi` are preserved from the original machine-usage schema. The official Alibaba 2018 schema documents `machine_usage` with `machine_id`, `time_stamp`, `cpu_util_percent`, `mem_util_percent`, `mem_gps`, `mkpi`, `net_in`, `net_out`, and `disk_io_percent`, and `machine_meta` with fields including `cpu_num`, `mem_size`, and `status`. citeturn387265view0turn815841view0

---

## 🧪 Test coverage added

- Dataset detection: Azure vs Alibaba
- Alibaba schema mapping and metadata enrichment
- Azure normalization compatibility
- Feature-selection logic for LSTM
- Sequence builder smoke test
- Forecast pair generation smoke test
- Optional deep anomaly-spike training test

---

## 📝 Notes

- Alibaba sample files created with `head -n 1000000` are supported directly.
- If `machine_meta` is missing, the pipeline still runs; VM metadata falls back to safe defaults.
- The dashboard now reads from both `output/ml_results_lstm` and legacy `output/ml_results`.
