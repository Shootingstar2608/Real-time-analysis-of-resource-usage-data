"""
Spark Structured Streaming Consumer
Reads from Kafka topics, performs real-time aggregations,
runs rule-based anomaly detection + LSTM inference,
and writes results for visualization.
"""

import os
import sys
import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json,
    col,
    window,
    avg,
    max as spark_max,
    min as spark_min,
    count,
    stddev,
    when,
    to_timestamp,
    struct,
    to_json,
    lit,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    BooleanType,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_CPU,
    KAFKA_TOPIC_ANOMALY,
    KAFKA_TOPIC_PREDICTIONS,
    SPARK_MASTER,
    SPARK_APP_NAME,
    SPARK_CHECKPOINT_DIR,
    SPARK_WATERMARK_DELAY,
    MODEL_DIR,
    LSTM_SEQUENCE_LENGTH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LSTM_RESULTS_DIR = os.path.join(OUTPUT_DIR, "ml_results_lstm")
PER_VM_DIR = os.path.join(OUTPUT_DIR, "per_vm_stats")
PER_CATEGORY_DIR = os.path.join(OUTPUT_DIR, "per_category_stats")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LSTM_RESULTS_DIR, exist_ok=True)
os.makedirs(PER_VM_DIR, exist_ok=True)
os.makedirs(PER_CATEGORY_DIR, exist_ok=True)
os.makedirs(SPARK_CHECKPOINT_DIR, exist_ok=True)

# Singleton model cache + per-VM rolling history
_models = {}
_vm_history = {}

# ============================================
# SCHEMAS
# ============================================

CPU_SCHEMA = StructType([
    StructField("timestamp", DoubleType(), True),
    StructField("vm_id", StringType(), True),
    StructField("min_cpu", DoubleType(), True),
    StructField("max_cpu", DoubleType(), True),
    StructField("avg_cpu", DoubleType(), True),
    StructField("cpu_range", DoubleType(), True),
    StructField("vm_category", StringType(), True),
    StructField("vm_core_count", IntegerType(), True),
    StructField("vm_memory_gb", IntegerType(), True),
    StructField("avg_memory", DoubleType(), True),
    StructField("network_in_mbps", DoubleType(), True),
    StructField("network_out_mbps", DoubleType(), True),
    StructField("disk_io_percent", DoubleType(), True),
    StructField("mem_gps", DoubleType(), True),
    StructField("mkpi", DoubleType(), True),
    StructField("ingestion_timestamp", StringType(), True),
    StructField("source_file", StringType(), True),
    StructField("data_source", StringType(), True),
])


def create_spark_session():
    """Create Spark session with Kafka integration."""
    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(SPARK_APP_NAME)
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .config("spark.sql.streaming.checkpointLocation", SPARK_CHECKPOINT_DIR)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.kafka.maxRatePerPartition", "1000")
        .config("spark.sql.streaming.stateStore.stateSchemaCheck", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully.")
    return spark


def read_kafka_stream(spark, topic, schema):
    """Read streaming data from Kafka topic."""
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed_stream = (
        raw_stream.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
        )
        .select("data.*", "kafka_timestamp")
        .withColumn("event_time", to_timestamp(col("ingestion_timestamp")))
    )

    return parsed_stream


def compute_realtime_aggregations(cpu_stream):
    """
    Real-time windowed aggregations:
    - Per-VM metrics (5-min windows)
    - Per-Category metrics (1-min windows)
    - Global system metrics (30-sec windows)
    """
    per_vm_stats = (
        cpu_stream.withWatermark("event_time", SPARK_WATERMARK_DELAY)
        .groupBy(
            window(col("event_time"), "5 minutes"),
            col("vm_id"),
            col("vm_category"),
        )
        .agg(
            avg("avg_cpu").alias("avg_cpu_5min"),
            spark_max("max_cpu").alias("peak_cpu_5min"),
            spark_min("min_cpu").alias("min_cpu_5min"),
            stddev("avg_cpu").alias("cpu_stddev_5min"),
            avg("cpu_range").alias("avg_cpu_range_5min"),
            count("*").alias("sample_count"),
        )
    )

    per_category_stats = (
        cpu_stream.withWatermark("event_time", SPARK_WATERMARK_DELAY)
        .groupBy(
            window(col("event_time"), "1 minute"),
            col("vm_category"),
        )
        .agg(
            avg("avg_cpu").alias("category_avg_cpu"),
            spark_max("max_cpu").alias("category_peak_cpu"),
            stddev("avg_cpu").alias("category_cpu_stddev"),
            count("*").alias("vm_count"),
        )
    )

    global_stats = (
        cpu_stream.withWatermark("event_time", SPARK_WATERMARK_DELAY)
        .groupBy(window(col("event_time"), "30 seconds"))
        .agg(
            avg("avg_cpu").alias("global_avg_cpu"),
            spark_max("max_cpu").alias("global_peak_cpu"),
            stddev("avg_cpu").alias("global_cpu_stddev"),
            avg("cpu_range").alias("global_avg_cpu_range"),
            count("*").alias("total_samples"),
            count(when(col("avg_cpu") > 80, True)).alias("high_cpu_count"),
            count(when(col("avg_cpu") < 5, True)).alias("idle_count"),
        )
    )

    return per_vm_stats, per_category_stats, global_stats


def detect_anomalies_rule_based(cpu_stream):
    """
    Rule-based anomaly detection.
    """
    anomalies = (
        cpu_stream.withWatermark("event_time", SPARK_WATERMARK_DELAY)
        .groupBy(
            window(col("event_time"), "2 minutes"),
            col("vm_id"),
            col("vm_category"),
        )
        .agg(
            avg("avg_cpu").alias("window_avg_cpu"),
            spark_max("max_cpu").alias("window_max_cpu"),
            spark_min("min_cpu").alias("window_min_cpu"),
            stddev("avg_cpu").alias("window_cpu_stddev"),
            avg("cpu_range").alias("window_avg_cpu_range"),
            count("*").alias("sample_count"),
        )
        .withColumn(
            "anomaly_type",
            when(col("window_avg_cpu") > 90, "HIGH_CPU_SUSTAINED")
            .when(col("window_cpu_stddev") > 25, "HIGH_VARIANCE")
            .when(col("window_avg_cpu_range") > 60, "EXTREME_FLUCTUATION")
            .when(col("window_max_cpu") > 95, "CPU_SATURATION")
            .otherwise(None)
        )
        .filter(col("anomaly_type").isNotNull())
        .withColumn(
            "severity",
            when(col("anomaly_type") == "CPU_SATURATION", "CRITICAL")
            .when(col("anomaly_type") == "HIGH_CPU_SUSTAINED", "HIGH")
            .when(col("anomaly_type") == "HIGH_VARIANCE", "MEDIUM")
            .when(col("anomaly_type") == "EXTREME_FLUCTUATION", "MEDIUM")
            .otherwise("LOW")
        )
    )

    return anomalies


def write_to_console(df, query_name, output_mode="update"):
    """Write stream to console (for debugging)."""
    return (
        df.writeStream
        .queryName(query_name)
        .outputMode(output_mode)
        .format("console")
        .option("truncate", "false")
        .option("numRows", 20)
        .start()
    )


def write_to_kafka(df, topic, query_name, output_mode="update"):
    """Write stream results back to Kafka topic."""
    return (
        df.select(to_json(struct("*")).alias("value"))
        .writeStream
        .queryName(query_name)
        .outputMode(output_mode)
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", topic)
        .option("checkpointLocation", os.path.join(SPARK_CHECKPOINT_DIR, query_name))
        .start()
    )


def write_to_parquet(df, path, query_name):
    """Write stream to Parquet files."""
    return (
        df.writeStream
        .queryName(query_name)
        .outputMode("append")
        .format("parquet")
        .option("path", path)
        .option("checkpointLocation", os.path.join(SPARK_CHECKPOINT_DIR, query_name))
        .trigger(processingTime="30 seconds")
        .start()
    )


def get_lstm_model():
    """
    Load LSTM model only once in the current Python process.
    """
    if "lstm_autoencoder" in _models:
        return _models["lstm_autoencoder"]

    model_path = os.path.join(MODEL_DIR, "lstm_autoencoder", "model.pth")
    if not os.path.exists(model_path):
        logger.warning(
            "LSTM model not found. Expected at models/saved/lstm_autoencoder/model.pth. "
            "Run: python ml_models/lstm_models.py"
        )
        return None

    from ml_models.lstm_models import AnomalyDetectorLSTM

    model = AnomalyDetectorLSTM()
    model.load()
    _models["lstm_autoencoder"] = model
    logger.info("LSTM Autoencoder loaded into singleton cache.")
    return model


def update_vm_history(vm_id, new_vm_rows, max_history=None):
    """
    Keep rolling history per VM so LSTM can score across batches.
    """
    max_history = max_history or max(LSTM_SEQUENCE_LENGTH * 4, 120)

    history_df = _vm_history.get(vm_id)
    if history_df is None:
        history_df = new_vm_rows.copy()
    else:
        history_df = pd.concat([history_df, new_vm_rows], ignore_index=True)

    history_df = history_df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    ).reset_index(drop=True)

    if len(history_df) > max_history:
        history_df = history_df.tail(max_history).reset_index(drop=True)

    _vm_history[vm_id] = history_df
    return history_df


def process_batch_with_lstm(batch_df, batch_id):
    """
    foreachBatch handler for LSTM inference.
    Converts the micro-batch to pandas, maintains per-VM rolling history,
    scores sequences with the loaded autoencoder, and appends results.
    """
    if batch_df.limit(1).count() == 0:
        logger.info(f"[LSTM] Batch {batch_id}: empty batch, skipping.")
        return

    model = get_lstm_model()
    if model is None:
        logger.warning(f"[LSTM] Batch {batch_id}: model unavailable, skipping.")
        return

    columns_needed = [
        "timestamp",
        "vm_id",
        "min_cpu",
        "max_cpu",
        "avg_cpu",
        "cpu_range",
        "vm_category",
        "vm_core_count",
        "vm_memory_gb",
        "avg_memory",
        "network_in_mbps",
        "network_out_mbps",
        "disk_io_percent",
        "mem_gps",
        "mkpi",
        "ingestion_timestamp",
        "source_file",
        "data_source",
        "event_time",
    ]

    pdf = batch_df.select(*columns_needed).toPandas()
    if pdf.empty:
        logger.info(f"[LSTM] Batch {batch_id}: empty pandas batch, skipping.")
        return

    pdf = pdf.sort_values(["vm_id", "timestamp"]).reset_index(drop=True)

    result_frames = []

    for vm_id, vm_batch in pdf.groupby("vm_id", sort=False):
        vm_batch = vm_batch.sort_values("timestamp").reset_index(drop=True)
        history_df = update_vm_history(vm_id, vm_batch)

        if len(history_df) < LSTM_SEQUENCE_LENGTH:
            continue

        anomalies, errors = model.predict(history_df)
        if len(errors) == 0:
            continue

        aligned_df = history_df.iloc[LSTM_SEQUENCE_LENGTH - 1:].copy().reset_index(drop=True)
        aligned_df["lstm_recon_error"] = errors
        aligned_df["is_anomaly_lstm"] = anomalies.astype(bool)
        aligned_df["model_name"] = "lstm_autoencoder"
        aligned_df["prediction_type"] = "anomaly_score"
        aligned_df["batch_id"] = int(batch_id)

        # only emit rows corresponding to current micro-batch
        n_new_rows = len(vm_batch)
        aligned_df = aligned_df.tail(min(n_new_rows, len(aligned_df))).copy()

        if not aligned_df.empty:
            result_frames.append(aligned_df)

    if not result_frames:
        logger.info(f"[LSTM] Batch {batch_id}: no enough per-VM history yet.")
        return

    result_pdf = pd.concat(result_frames, ignore_index=True)
    spark = batch_df.sparkSession
    result_sdf = spark.createDataFrame(result_pdf)

    # Save all LSTM scores to parquet
    result_sdf.write.mode("append").parquet(LSTM_RESULTS_DIR)

    # Send only LSTM anomalies to Kafka for realtime visualization
    anomaly_sdf = (
        result_sdf
        .filter(col("is_anomaly_lstm") == True)
        .withColumn("anomaly_type", lit("LSTM_RECONSTRUCTION"))
        .withColumn("severity", when(col("lstm_recon_error") > 0.10, "HIGH").otherwise("MEDIUM"))
    )

    if anomaly_sdf.limit(1).count() > 0:
        (
            anomaly_sdf.select(to_json(struct("*")).alias("value"))
            .write
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
            .option("topic", KAFKA_TOPIC_PREDICTIONS)
            .save()
        )
        logger.info(
            f"[LSTM] Batch {batch_id}: wrote {anomaly_sdf.count()} anomaly row(s) to Kafka."
        )

    logger.info(
        f"[LSTM] Batch {batch_id}: wrote {result_sdf.count()} scored row(s) to {LSTM_RESULTS_DIR}."
    )


def main():
    """Main entry point for Spark Streaming application."""
    logger.info("=" * 60)
    logger.info("  Starting Spark Structured Streaming Pipeline")
    logger.info("=" * 60)

    spark = create_spark_session()

    logger.info("Connecting to Kafka topic...")
    cpu_stream = read_kafka_stream(spark, KAFKA_TOPIC_CPU, CPU_SCHEMA)

    logger.info("Setting up real-time aggregations...")
    per_vm_stats, per_category_stats, global_stats = compute_realtime_aggregations(cpu_stream)

    logger.info("Setting up rule-based anomaly detection...")
    anomalies = detect_anomalies_rule_based(cpu_stream)

    queries = []

    # Console sinks
    queries.append(write_to_console(global_stats, "global_stats_console"))
    queries.append(write_to_console(anomalies, "anomalies_console"))

    # Rule-based anomalies to Kafka
    queries.append(write_to_kafka(anomalies, KAFKA_TOPIC_ANOMALY, "anomalies_to_kafka"))

    # Aggregations to parquet
    queries.append(write_to_parquet(per_vm_stats, PER_VM_DIR, "per_vm_to_parquet"))
    queries.append(write_to_parquet(per_category_stats, PER_CATEGORY_DIR, "per_category_to_parquet"))

    # LSTM inference via foreachBatch
    queries.append(
        cpu_stream.writeStream
        .queryName("lstm_inference")
        .outputMode("append")
        .foreachBatch(process_batch_with_lstm)
        .option("checkpointLocation", os.path.join(SPARK_CHECKPOINT_DIR, "lstm_inference"))
        .trigger(processingTime="30 seconds")
        .start()
    )

    logger.info(f"Started {len(queries)} streaming queries.")
    logger.info("Waiting for data...")
    logger.info("Press Ctrl+C to stop.")

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Stopping all streaming queries...")
        for q in queries:
            q.stop()
        spark.stop()
        logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()