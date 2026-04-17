# 📊 Real-time Analysis of Resource Usage Data

> Big Data Project: Kafka + Spark Streaming + ML/DL + Dashboard

## 🏗️ Architecture

```
┌──────────────┐    ┌─────────┐    ┌──────────────────┐    ┌──────────────┐
│  Data Source  │───▶│  Kafka  │───▶│ Spark Streaming  │───▶│  Dashboard   │
│ (Azure VM    │    │ Topics: │    │                  │    │  (Streamlit) │
│  Traces /    │    │ vm-cpu  │    │ ┌──────────────┐ │    │              │
│  Synthetic)  │    │ vm-mem  │    │ │ Aggregations │ │    │  📈 Charts   │
└──────────────┘    └─────────┘    │ │ (Window 5m)  │ │    │  🔍 Anomaly  │
                         │         │ └──────────────┘ │    │  📊 VM Stats │
                         │         │ ┌──────────────┐ │    └──────────────┘
                         │         │ │  ML Inference │ │           ▲
                         │         │ │ (Isolation    │ │           │
                         │         │ │  Forest)      │─┼───▶ Parquet Output
                         │         │ └──────────────┘ │
                         │         └──────────────────┘
                         │
                    ┌─────────┐
                    │  Kafka  │◀── Anomaly alerts
                    │ vm-anom │
                    └─────────┘
```

## 📁 Project Structure

```
Big_data/
├── config/
│   └── config.py                 # All configurations
├── data/
│   ├── download_data.sh          # Download Azure VM Traces
│   ├── prepare_data.py           # Process data / generate synthetic
│   ├── raw/                      # Raw downloaded data
│   └── processed/                # Cleaned data (Parquet + CSV)
├── kafka_stream/
│   └── producer.py               # Kafka Producer (data → Kafka)
├── spark_processing/
│   ├── streaming_consumer.py     # Spark Streaming (aggregation + rules)
│   └── streaming_ml_pipeline.py  # Spark Streaming + ML inference
├── ml_models/
│   ├── isolation_forest_model.py # Isolation Forest (anomaly detection)
│   └── lstm_models.py            # LSTM Autoencoder + Forecaster
├── dashboard/
│   └── app.py                    # Streamlit real-time dashboard
├── models/saved/                 # Trained model files
├── output/
│   ├── ml_results/               # ML inference Parquet output
│   └── plots/                    # Visualization images
├── docker-compose.yml            # Kafka + Spark infrastructure
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Infrastructure (Kafka + Spark)

```bash
docker-compose up -d
```

Verify services:
- Kafka UI: http://localhost:8080
- Spark Master UI: http://localhost:8081

### 3. Generate/Download Data

```bash
# Option A: Generate synthetic data (~5-8GB, recommended for testing)
python data/prepare_data.py --synthetic

# Option B: Download Azure VM Traces (real data)
bash data/download_data.sh
python data/prepare_data.py
```

### 4. Train ML Models

```bash
# Train Isolation Forest
python ml_models/isolation_forest_model.py

# Train LSTM models (Autoencoder + Forecaster)
python ml_models/lstm_models.py
```

### 5. Run the Pipeline

Open 3 terminals:

```bash
# Terminal 1: Start Kafka Producer (stream data)
python kafka_stream/producer.py --speed 100

# Terminal 2: Start Spark Streaming + ML
python spark_processing/streaming_ml_pipeline.py

# Terminal 3: Start Dashboard
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

## 🤖 ML/DL Models

### 1. Isolation Forest (Unsupervised Anomaly Detection)
- **Input**: Single data point (avg_cpu, max_cpu, memory, network, disk...)
- **Output**: is_anomaly (bool), anomaly_score, severity
- **Use case**: Detect point anomalies (CPU spike, unusual memory usage)
- **Advantage**: Fast inference, no labeled data needed

### 2. LSTM Autoencoder (Deep Anomaly Detection)
- **Input**: Sequence of 30 consecutive readings
- **Output**: Reconstruction error → anomaly if error > threshold
- **Use case**: Detect pattern anomalies (memory leak, missing daily cycle)
- **Advantage**: Captures temporal patterns

### 3. LSTM Forecaster (Resource Prediction)
- **Input**: Sequence of 30 readings
- **Output**: Predicted CPU/Memory for next 12 time steps
- **Use case**: Capacity planning, proactive auto-scaling
- **Insight**: "VM-X will reach 90% CPU in 1 hour"

## 📊 Dashboard Insights

| Page | Insight |
|------|---------|
| Overview | System-wide CPU/Memory trends, KPIs |
| Anomaly Detection | Real-time anomaly scatter plot, timeline, severity |
| Forecasting | Per-VM CPU/Memory prediction, correlation heatmap |
| VM Analytics | Top consumers, usage distribution by type, Network I/O |
| ML Results | Training plots, model comparison |

## 🔧 Configuration

Edit `config/config.py` to customize:
- Kafka broker address, topic names
- Spark batch interval, watermark delay
- ML model hyperparameters
- Dashboard refresh rate

## 📚 Dataset

**Azure VM Traces 2019** (Microsoft Research)
- Source: https://github.com/Azure/AzurePublicDataset
- 2.6M+ VMs from Azure datacenters
- Metrics: CPU, Memory (sampled every 5 minutes)
- Paper: *"Resource Central: Understanding and Predicting Workloads for Improved Resource Management in Large Cloud Platforms"*

## 👥 Team

- **Course**: Big Data
- **Topic**: Real-time Analysis of Resource Usage Data
- **Tech Stack**: Apache Kafka, Apache Spark (Structured Streaming), Scikit-learn, PyTorch, Streamlit
