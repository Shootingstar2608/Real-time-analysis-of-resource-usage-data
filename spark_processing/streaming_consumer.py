"""
Spark Structured Streaming Consumer
Reads from Kafka topics, performs real-time aggregations,
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
    IntegerType, BooleanType, TimestampType
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_CPU,
    KAFKA_TOPIC_MEMORY,
    KAFKA_TOPIC_ANOMALY,
    KAFKA_TOPIC_PREDICTIONS,
    SPARK_APP_NAME,
    SPARK_CHECKPOINT_DIR,
    SPARK_WATERMARK_DELAY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# SCHEMAS
# ============================================
# Schema khớp với real_data_producer.py (Azure VM Traces)
CPU_SCHEMA = StructType([
    StructField("timestamp", DoubleType(), True),        # seconds (relative)
    StructField("vm_id", StringType(), True),             # encrypted hash
    StructField("min_cpu", DoubleType(), True),
    StructField("max_cpu", DoubleType(), True),
    StructField("avg_cpu", DoubleType(), True),
    StructField("cpu_range", DoubleType(), True),         # max - min
    StructField("vm_category", StringType(), True),       # Interactive/Delay-insensitive/Unknown
    StructField("vm_core_count", IntegerType(), True),
    StructField("vm_memory_gb", IntegerType(), True),
    StructField("ingestion_timestamp", StringType(), True),
    StructField("source_file", StringType(), True),
])


def create_spark_session():
    """Create Spark session with Kafka integration"""
    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.sql.streaming.checkpointLocation", SPARK_CHECKPOINT_DIR) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.streaming.kafka.maxRatePerPartition", "1000") \
        .config("spark.sql.streaming.stateStore.stateSchemaCheck", "false") \
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
        .option("failOnDataLoss", "false") \
        .load()
    
    # Parse JSON value
    parsed_stream = raw_stream \
        .select(
            from_json(
                col("value").cast("string"), 
                schema
            ).alias("data"),
            col("timestamp").alias("kafka_timestamp")
        ) \
        .select("data.*", "kafka_timestamp") \
        .withColumn(
            "event_time",
            to_timestamp(col("ingestion_timestamp"))
        )
    
    return parsed_stream


def compute_realtime_aggregations(cpu_stream):
    """
    Real-time windowed aggregations (CPU-only, real Azure data):
    - Per-VM metrics (5-min windows)
    - Per-Category metrics (1-min windows)
    - Global system metrics (30-sec windows)
    """
    
    # ============================================
    # 1. Per-VM 5-minute window aggregation
    # ============================================
    per_vm_stats = cpu_stream \
        .withWatermark("event_time", SPARK_WATERMARK_DELAY) \
        .groupBy(
            window(col("event_time"), "5 minutes"),
            col("vm_id"),
            col("vm_category")
        ) \
        .agg(
            avg("avg_cpu").alias("avg_cpu_5min"),
            spark_max("max_cpu").alias("peak_cpu_5min"),
            spark_min("min_cpu").alias("min_cpu_5min"),
            stddev("avg_cpu").alias("cpu_stddev_5min"),
            avg("cpu_range").alias("avg_cpu_range_5min"),
            count("*").alias("sample_count"),
        )
    
    # ============================================
    # 2. Per-Category 1-minute aggregation
    # ============================================
    per_category_stats = cpu_stream \
        .withWatermark("event_time", SPARK_WATERMARK_DELAY) \
        .groupBy(
            window(col("event_time"), "1 minute"),
            col("vm_category")
        ) \
        .agg(
            avg("avg_cpu").alias("category_avg_cpu"),
            spark_max("max_cpu").alias("category_peak_cpu"),
            stddev("avg_cpu").alias("category_cpu_stddev"),
            count("*").alias("vm_count"),
        )
    
    # ============================================
    # 3. Global system metrics (30-sec windows)
    # ============================================
    global_stats = cpu_stream \
        .withWatermark("event_time", SPARK_WATERMARK_DELAY) \
        .groupBy(
            window(col("event_time"), "30 seconds")
        ) \
        .agg(
            avg("avg_cpu").alias("global_avg_cpu"),
            spark_max("max_cpu").alias("global_peak_cpu"),
            stddev("avg_cpu").alias("global_cpu_stddev"),
            avg("cpu_range").alias("global_avg_cpu_range"),
            count("*").alias("total_samples"),
            count(when(col("avg_cpu") > 80, True)).alias("high_cpu_count"),
            count(when(col("avg_cpu") < 5, True)).alias("idle_count"),
        )
    
    return per_vm_stats, per_category_stats, global_stats


def detect_anomalies_rule_based(cpu_stream):
    """
    Rule-based anomaly detection (complementing ML models):
    - CPU > 90% sustained
    - Sudden CPU spike (stddev > threshold)
    - Unusual pattern for VM type
    """
    
    anomalies = cpu_stream \
        .withWatermark("event_time", SPARK_WATERMARK_DELAY) \
        .groupBy(
            window(col("event_time"), "2 minutes"),
            col("vm_id"),
            col("vm_category")
        ) \
        .agg(
            avg("avg_cpu").alias("window_avg_cpu"),
            spark_max("max_cpu").alias("window_max_cpu"),
            spark_min("min_cpu").alias("window_min_cpu"),
            stddev("avg_cpu").alias("window_cpu_stddev"),
            avg("cpu_range").alias("window_avg_cpu_range"),
            count("*").alias("sample_count"),
        ) \
        .withColumn(
            "anomaly_type",
            when(col("window_avg_cpu") > 90, "HIGH_CPU_SUSTAINED")
            .when(col("window_cpu_stddev") > 25, "HIGH_VARIANCE")
            .when(col("window_avg_cpu_range") > 60, "EXTREME_FLUCTUATION")
            .when(col("window_max_cpu") > 95, "CPU_SATURATION")
            .otherwise(None)
        ) \
        .filter(col("anomaly_type").isNotNull()) \
        .withColumn("severity", 
            when(col("anomaly_type") == "CPU_SATURATION", "CRITICAL")
            .when(col("anomaly_type") == "HIGH_CPU_SUSTAINED", "HIGH")
            .when(col("anomaly_type") == "HIGH_VARIANCE", "MEDIUM")
            .when(col("anomaly_type") == "EXTREME_FLUCTUATION", "MEDIUM")
            .otherwise("LOW")
        )
    
    return anomalies


def write_to_console(df, query_name, output_mode="update"):
    """Write stream to console (for debugging)"""
    return df.writeStream \
        .queryName(query_name) \
        .outputMode(output_mode) \
        .format("console") \
        .option("truncate", "false") \
        .option("numRows", 20) \
        .start()


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


def write_to_parquet(df, path, query_name):
    """Write stream to Parquet files (data lake)"""
    return df.writeStream \
        .queryName(query_name) \
        .outputMode("append") \
        .format("parquet") \
        .option("path", path) \
        .option("checkpointLocation",
                os.path.join(SPARK_CHECKPOINT_DIR, query_name)) \
        .trigger(processingTime="30 seconds") \
        .start()


def main():
    """Main entry point for Spark Streaming application"""
    logger.info("=" * 60)
    logger.info("  Starting Spark Structured Streaming Pipeline")
    logger.info("=" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    
    # Read from Kafka (CPU only - real Azure data)
    logger.info("Connecting to Kafka topic...")
    cpu_stream = read_kafka_stream(spark, KAFKA_TOPIC_CPU, CPU_SCHEMA)
    
    # Compute aggregations
    logger.info("Setting up real-time aggregations...")
    per_vm_stats, per_category_stats, global_stats = \
        compute_realtime_aggregations(cpu_stream)
    
    # Detect anomalies
    logger.info("Setting up anomaly detection...")
    anomalies = detect_anomalies_rule_based(cpu_stream)
    
    # ============================================
    # OUTPUT SINKS
    # ============================================
    queries = []
    
    # Console output (for development/debugging)
    queries.append(
        write_to_console(global_stats, "global_stats_console")
    )
    queries.append(
        write_to_console(anomalies, "anomalies_console")
    )
    
    # Write anomalies to Kafka (for dashboard consumption)
    queries.append(
        write_to_kafka(anomalies, KAFKA_TOPIC_ANOMALY, "anomalies_to_kafka")
    )
    
    # Write aggregated stats to Parquet (data lake)
    queries.append(
        write_to_parquet(
            per_vm_stats,
            "output/per_vm_stats",
            "per_vm_to_parquet"
        )
    )
    queries.append(
        write_to_parquet(
            per_category_stats,
            "output/per_category_stats", 
            "per_category_to_parquet"
        )
    )
    
    logger.info(f"Started {len(queries)} streaming queries")
    logger.info("Waiting for data...")
    logger.info("Press Ctrl+C to stop")
    
    # Wait for termination
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
