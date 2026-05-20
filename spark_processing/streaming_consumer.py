"""
Spark Structured Streaming Consumer
Reads from Kafka topics, performs real-time aggregations for Alibaba cluster data,
runs ML inference, and writes results for visualization.
"""

import os
import sys
import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, avg, max as spark_max, min as spark_min,
    count, stddev, expr, when, lit, current_timestamp,
    to_timestamp, struct, to_json
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, TimestampType, FloatType, ArrayType
)
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_RESOURCE,
    KAFKA_TOPIC_ANOMALY,
    SPARK_MASTER,
    SPARK_APP_NAME,
    SPARK_CHECKPOINT_DIR,
    SPARK_WATERMARK_DELAY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# ML MODEL LOADING (Singleton Pattern)
# ============================================
_models = {}

def get_if_model():
    """Load Isolation Forest model once per executor"""
    if "if_model" not in _models:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "saved", "isolation_forest.joblib"
        )
        if os.path.exists(model_path):
            _models["if_model"] = joblib.load(model_path)
            logger.info(f"Loaded Isolation Forest model from {model_path}")
        else:
            logger.warning(f"IF model not found at {model_path}")
            return None
    return _models["if_model"]


def get_lstm_model():
    """Load BiLSTM Autoencoder model once (lazy singleton)."""
    if "lstm_model" not in _models:
        try:
            import torch
            import pickle
            from sklearn.preprocessing import MinMaxScaler
            # Local import to avoid circular deps
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from ml_models.lstm_models import BiLSTMAttentionAutoencoder, LSTM_SEQUENCE_LENGTH
            from config.config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_THRESHOLD_PERCENTILE

            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "saved", "lstm_autoencoder"
            )
            if not os.path.exists(model_dir):
                logger.warning(f"LSTM model dir not found: {model_dir}")
                _models["lstm_model"] = None
                return None

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            n_features = 3  # cpu_util_percent, mem_util_percent, disk_io_percent
            net = BiLSTMAttentionAutoencoder(
                n_features=n_features,
                hidden_size=LSTM_HIDDEN_SIZE,
                num_layers=LSTM_NUM_LAYERS
            ).to(device)
            net.load_state_dict(
                torch.load(os.path.join(model_dir, "model.pth"), map_location=device)
            )
            net.eval()

            threshold = np.load(os.path.join(model_dir, "threshold.npy"))
            scaler    = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), 'rb'))

            _models["lstm_model"] = {
                "net": net, "threshold": threshold,
                "scaler": scaler, "device": device,
                "seq_len": LSTM_SEQUENCE_LENGTH
            }
            logger.info(f"Loaded BiLSTM model from {model_dir}")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
            _models["lstm_model"] = None
    return _models["lstm_model"]


# ============================================
# SCHEMAS
# ============================================
ALIBABA_SCHEMA = StructType([
    StructField("machine_id", StringType(), True),
    StructField("timestamp", DoubleType(), True),
    StructField("cpu_util_percent", DoubleType(), True),
    StructField("mem_util_percent", DoubleType(), True),
    StructField("disk_io_percent", DoubleType(), True),
    StructField("ingestion_timestamp", StringType(), True),
    StructField("source_file", StringType(), True),
])


def create_spark_session():
    """Create Spark session with Kafka integration"""
    java_opts = (
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED"
    )
    spark = SparkSession.builder \
        .master(SPARK_MASTER) \
        .appName(SPARK_APP_NAME) \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.0") \
        .config("spark.sql.streaming.checkpointLocation", SPARK_CHECKPOINT_DIR) \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.streaming.kafka.maxRatePerPartition", "100") \
        .config("spark.kafka.consumer.cache.timeout", "60000") \
        .config("spark.sql.streaming.stateStore.stateSchemaCheck", "false") \
        .config("spark.driver.extraJavaOptions", java_opts) \
        .config("spark.executor.extraJavaOptions", java_opts) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully")
    return spark


def read_kafka_stream(spark, topic, schema):
    """Read streaming data from Kafka topic"""
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", topic) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 5000) \
        .option("failOnDataLoss", "false") \
        .load()
    
    parsed_stream = raw_stream \
        .select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp")
        ) \
        .select("data.*", "kafka_timestamp") \
        .withColumn("event_time", to_timestamp(col("ingestion_timestamp")))
    
    return parsed_stream


def compute_realtime_aggregations(metric_stream):
    """
    Real-time windowed aggregations:
    - Per-machine metrics (5-min windows)
    - Global system metrics (30-sec windows)
    """
    # 1. Per-machine 5-minute window aggregation
    per_machine_stats = metric_stream \
        .withWatermark("event_time", SPARK_WATERMARK_DELAY) \
        .groupBy(
            window(col("event_time"), "5 minutes"),
            col("machine_id")
        ) \
        .agg(
            avg("cpu_util_percent").alias("avg_cpu_5min"),
            avg("mem_util_percent").alias("avg_mem_5min"),
            avg("disk_io_percent").alias("avg_disk_5min"),
            count("*").alias("sample_count"),
        )
    
    # 2. Global system metrics (30-sec windows)
    global_stats = metric_stream \
        .withWatermark("event_time", SPARK_WATERMARK_DELAY) \
        .groupBy(
            window(col("event_time"), "30 seconds")
        ) \
        .agg(
            avg("cpu_util_percent").alias("global_avg_cpu"),
            avg("mem_util_percent").alias("global_avg_mem"),
            avg("disk_io_percent").alias("global_avg_disk"),
            count("*").alias("total_samples"),
        )
    
    return per_machine_stats, global_stats


def detect_anomalies_rule_based(metric_stream):
    """
    Rule-based anomaly detection (complementing ML models):
    - Resource saturation across CPU, RAM, Disk
    """
    anomalies = metric_stream \
        .withWatermark("event_time", SPARK_WATERMARK_DELAY) \
        .groupBy(
            window(col("event_time"), "2 minutes"),
            col("machine_id")
        ) \
        .agg(
            avg("cpu_util_percent").alias("window_avg_cpu"),
            avg("mem_util_percent").alias("window_avg_mem"),
            avg("disk_io_percent").alias("window_avg_disk"),
            count("*").alias("sample_count"),
        ) \
        .withColumn(
            "anomaly_type",
            when(col("window_avg_cpu") > 90, "HIGH_CPU_SUSTAINED")
            .when(col("window_avg_mem") > 95, "MEMORY_LIMIT_REACHED")
            .when(col("window_avg_disk") > 90, "HIGH_DISK_IO")
            .otherwise(None)
        ) \
        .filter(col("anomaly_type").isNotNull()) \
        .withColumn("severity", 
            when(col("anomaly_type") == "MEMORY_LIMIT_REACHED", "CRITICAL")
            .when(col("anomaly_type") == "HIGH_CPU_SUSTAINED", "HIGH")
            .when(col("anomaly_type") == "HIGH_DISK_IO", "MEDIUM")
            .otherwise("LOW")
        )
    
    return anomalies


def _run_lstm_inference(pandas_df):
    """
    Run BiLSTM Autoencoder inference on a pandas DataFrame.
    Returns a boolean array (is_anomaly_lstm) and float array (lstm_recon_error).
    """
    import torch
    lstm = get_lstm_model()
    if lstm is None or len(pandas_df) < lstm["seq_len"]:
        n = len(pandas_df)
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=float)

    try:
        net       = lstm["net"]
        threshold = lstm["threshold"]
        scaler    = lstm["scaler"]
        device    = lstm["device"]
        seq_len   = lstm["seq_len"]

        feat_cols = ['cpu_util_percent', 'mem_util_percent', 'disk_io_percent']
        features  = pandas_df[feat_cols].fillna(0).values
        scaled    = scaler.transform(features)

        # Build sliding-window sequences
        sequences = np.array([
            scaled[i: i + seq_len]
            for i in range(len(scaled) - seq_len + 1)
        ])  # (N, seq_len, 3)

        X_t = torch.FloatTensor(sequences).to(device)
        with torch.no_grad():
            recon, _ = net(X_t)
            errors = torch.mean((X_t - recon) ** 2, dim=[1, 2]).cpu().numpy()

        # Pad front with NaN so array length == len(pandas_df)
        pad   = seq_len - 1
        full_errors = np.concatenate([np.full(pad, np.nan), errors])
        # NaN rows → treat as normal
        valid_errors = np.where(np.isnan(full_errors), 0.0, full_errors)
        is_anomaly   = valid_errors > threshold
        return is_anomaly, valid_errors
    except Exception as e:
        logger.warning(f"LSTM inference failed: {e}")
        n = len(pandas_df)
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=float)


def process_batch_with_ml(batch_df, epoch_id):
    """
    foreachBatch handler: called per micro-batch with a regular Spark DataFrame.
    Runs Isolation Forest + BiLSTM Autoencoder and saves results as Parquet.
    """
    if batch_df.isEmpty():
        return

    try:
        pandas_df = batch_df.select(
            "machine_id", "cpu_util_percent", "mem_util_percent", "disk_io_percent",
            "timestamp", "ingestion_timestamp"
        ).toPandas()

        if pandas_df.empty:
            return

        # ── Isolation Forest ──
        if_model = get_if_model()
        if if_model is not None:
            feats = pandas_df[['cpu_util_percent', 'mem_util_percent', 'disk_io_percent']].fillna(0)
            scores = if_model.decision_function(feats)
            pandas_df['if_anomaly_score'] = scores
            pandas_df['is_anomaly_if']    = scores < -0.05
        else:
            pandas_df['if_anomaly_score'] = 0.0
            pandas_df['is_anomaly_if']    = False

        # backward-compat alias kept for dashboard
        pandas_df['is_anomaly_ai'] = pandas_df['is_anomaly_if']

        # ── BiLSTM Autoencoder ──
        lstm_flags, lstm_errors = _run_lstm_inference(pandas_df)
        pandas_df['lstm_recon_error']  = lstm_errors
        pandas_df['is_anomaly_lstm']   = lstm_flags

        # ── Ensemble: anomaly if flagged by either model ──
        pandas_df['is_anomaly_ensemble'] = (
            pandas_df['is_anomaly_if'] | pandas_df['is_anomaly_lstm']
        )

        # ── Save to Parquet ──
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "ml_results"
        )
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, f"batch_{epoch_id}.parquet")
        pandas_df.to_parquet(out_file, index=False)

        n_if   = int(pandas_df['is_anomaly_if'].sum())
        n_lstm = int(pandas_df['is_anomaly_lstm'].sum())
        n_ens  = int(pandas_df['is_anomaly_ensemble'].sum())
        logger.info(
            f"Batch {epoch_id}: {len(pandas_df):,} rows | "
            f"IF={n_if} | LSTM={n_lstm} | Ensemble={n_ens}"
        )

        # ── Print top-5 ensemble anomalies ──
        top5_df = pandas_df[pandas_df['is_anomaly_ensemble']].nsmallest(
            5, 'if_anomaly_score'
        )[['machine_id', 'cpu_util_percent', 'mem_util_percent',
           'disk_io_percent', 'if_anomaly_score', 'lstm_recon_error']]

        if not top5_df.empty:
            logger.info(f"\n{'='*90}")
            logger.info(f"   ENSEMBLE TOP-5 ANOMALIES (Batch {epoch_id})")
            logger.info(f"{'='*90}")
            for _, row in top5_df.iterrows():
                mid = str(row['machine_id'])
                mid = mid[:16] + '...' if len(mid) > 16 else mid
                logger.info(
                    f"  {mid:<19} CPU:{row['cpu_util_percent']:5.1f}% "
                    f"RAM:{row['mem_util_percent']:5.1f}% "
                    f"DISK:{row['disk_io_percent']:5.1f}% "
                    f"IF:{row['if_anomaly_score']:6.3f} "
                    f"LSTM_err:{row['lstm_recon_error']:.4f}"
                )
            logger.info(f"{'='*90}\n")

    except Exception as e:
        logger.error(f"Error processing batch {epoch_id}: {e}", exc_info=True)


def write_to_kafka(df, topic, query_name, output_mode="update"):
    """Write stream results back to Kafka topic"""
    return df \
        .select(
            to_json(struct("*")).alias("value")
        ) \
        .writeStream \
        .queryName(query_name) \
        .outputMode(output_mode) \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("topic", topic) \
        .option("checkpointLocation", 
                os.path.join(SPARK_CHECKPOINT_DIR, query_name)) \
        .start()


def _start_queries(spark, metric_stream):
    """Build and start all streaming queries. Returns the list of active queries."""
    queries = []

    # Rule-based anomalies → Kafka
    anomalies = detect_anomalies_rule_based(metric_stream)
    queries.append(
        write_to_kafka(anomalies, KAFKA_TOPIC_ANOMALY, "anomalies_to_kafka")
    )

    # ML inference (IF + BiLSTM) → Parquet
    logger.info("Setting up AI (Isolation Forest + BiLSTM) via foreachBatch...")
    ml_query = (
        metric_stream.writeStream
        .queryName("ml_inference_foreach")
        .foreachBatch(process_batch_with_ml)
        .option(
            "checkpointLocation",
            os.path.join(SPARK_CHECKPOINT_DIR, "ml_inference_foreach")
        )
        .trigger(processingTime="5 seconds")
        .start()
    )
    queries.append(ml_query)
    return queries


def main():
    import time

    logger.info("=" * 60)
    logger.info("  Starting Spark Streaming Pipeline (Alibaba Multi-Metrics)")
    logger.info("=" * 60)

    spark = create_spark_session()
    metric_stream = read_kafka_stream(spark, KAFKA_TOPIC_RESOURCE, ALIBABA_SCHEMA)

    # Compute aggregations (not written anywhere yet, available for extension)
    compute_realtime_aggregations(metric_stream)

    queries = _start_queries(spark, metric_stream)
    logger.info(f"Started {len(queries)} streaming queries — waiting for data...")
    logger.info("Press Ctrl+C to stop")

    # ── Auto-restart loop: recover from transient Kafka failures ──
    MAX_RESTARTS = 10
    restarts = 0
    while restarts < MAX_RESTARTS:
        try:
            spark.streams.awaitAnyTermination()
            # awaitAnyTermination returned normally → all streams finished
            logger.info("All streaming queries finished.")
            break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping all queries...")
            break
        except Exception as e:
            restarts += 1
            logger.error(
                f"Streaming query failed (restart {restarts}/{MAX_RESTARTS}): {e}"
            )
            if restarts >= MAX_RESTARTS:
                logger.error("Max restarts reached — exiting.")
                break

            # Stop failed queries
            for q in spark.streams.active:
                try:
                    q.stop()
                except Exception:
                    pass

            wait = min(5 * restarts, 30)   # exponential back-off up to 30s
            logger.info(f"Waiting {wait}s before restarting...")
            time.sleep(wait)

            # Re-read stream and restart queries
            metric_stream = read_kafka_stream(spark, KAFKA_TOPIC_RESOURCE, ALIBABA_SCHEMA)
            queries = _start_queries(spark, metric_stream)
            logger.info(f"Queries restarted (attempt {restarts}).")

    # Clean shutdown
    for q in spark.streams.active:
        try:
            q.stop()
        except Exception:
            pass
    spark.stop()
    logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()
