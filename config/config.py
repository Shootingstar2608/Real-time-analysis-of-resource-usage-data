"""
Project Configuration
Real-time Analysis of Resource Usage Data
"""

# ============================================
# KAFKA CONFIGURATION
# ============================================
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_RESOURCE = "machine-resource-usage"
KAFKA_TOPIC_ANOMALY = "machine-anomalies"
KAFKA_TOPIC_PREDICTIONS = "machine-predictions"
KAFKA_GROUP_ID = "resource-analysis-group"

# ============================================
# SPARK CONFIGURATION
# ============================================
SPARK_MASTER = "local[4]"
SPARK_APP_NAME = "ResourceUsageAnalysis"
SPARK_BATCH_INTERVAL = 5  # seconds
SPARK_WATERMARK_DELAY = "10 seconds"
SPARK_CHECKPOINT_DIR = "/tmp/spark-checkpoints"

# ============================================
# DATA CONFIGURATION
# ============================================
DATA_DIR = "data/alibaba"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models/saved"

# Alibaba Cluster Trace V2018 - machine_usage columns (no header in files)
# schema: machine_id, time_stamp, cpu_util_percent, mem_util_percent, mem_gps, mkpi, net_in, net_out, disk_io_percent
ALIBABA_USAGE_COLUMNS = [
    "machine_id", "timestamp", "cpu_util_percent", "mem_util_percent",
    "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"
]

# Features to extract for the ML pipeline
FEATURES_COLUMNS = [
    "machine_id", "timestamp", "cpu_util_percent", "mem_util_percent", "disk_io_percent", "ingestion_timestamp"
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
LSTM_THRESHOLD_PERCENTILE = 99  # reconstruction error threshold

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
