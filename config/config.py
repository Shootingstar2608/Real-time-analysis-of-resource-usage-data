"""
Project Configuration
Real-time Analysis of Resource Usage Data
"""

# ============================================
# KAFKA CONFIGURATION
# ============================================
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_CPU = "vm-cpu-usage"
KAFKA_TOPIC_MEMORY = "vm-memory-usage"
KAFKA_TOPIC_ANOMALY = "vm-anomalies"
KAFKA_TOPIC_PREDICTIONS = "vm-predictions"
KAFKA_GROUP_ID = "resource-analysis-group"

# ============================================
# SPARK CONFIGURATION
# ============================================
SPARK_MASTER = "local[*]"
SPARK_APP_NAME = "ResourceUsageAnalysis"
SPARK_BATCH_INTERVAL = 5  # seconds
SPARK_WATERMARK_DELAY = "10 seconds"
SPARK_CHECKPOINT_DIR = "/tmp/spark-checkpoints"

# ============================================
# DATA CONFIGURATION
# ============================================
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models/saved"

# Azure VM Trace - Real Data Columns (no header in files)
# cpu_readings: 5 cols → timestamp_s, vm_id, min_cpu, max_cpu, avg_cpu
# vmtable: 11 cols → vm_id, sub_id, deploy_id, created, deleted,
#                     max_cpu, avg_cpu, p95_cpu, category, cores, memory
CPU_READINGS_COLUMNS = [
    "timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"
]
VMTABLE_COLUMNS = [
    "vm_id", "subscription_id", "deployment_id",
    "timestamp_vm_created", "timestamp_vm_deleted",
    "max_cpu_lifetime", "avg_cpu_lifetime", "p95_cpu_lifetime",
    "vm_category", "vm_core_count", "vm_memory_gb"
]

# Enriched record columns (after join cpu_readings + vmtable)
ENRICHED_COLUMNS = [
    "timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu",
    "cpu_range", "vm_category", "vm_core_count", "vm_memory_gb",
    "ingestion_timestamp"
]

# ============================================
# ML MODEL CONFIGURATION
# ============================================
# Isolation Forest
IF_CONTAMINATION = 0.05  # 5% anomalies expected
IF_N_ESTIMATORS = 100
IF_RANDOM_STATE = 42

# LSTM Autoencoder
LSTM_SEQUENCE_LENGTH = 30  # 30 time steps
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64
LSTM_THRESHOLD_PERCENTILE = 95  # reconstruction error threshold

# LSTM Forecasting
FORECAST_HORIZON = 12  # predict next 12 time steps
FORECAST_HIDDEN_SIZE = 128
FORECAST_EPOCHS = 100

# ============================================
# VISUALIZATION CONFIGURATION
# ============================================
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_INTERVAL = 3  # seconds
