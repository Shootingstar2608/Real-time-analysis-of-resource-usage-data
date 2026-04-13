"""
Real Data Kafka Producer
Đọc trực tiếp machine_usage data (.csv hoặc .csv.gz) và đẩy vào Kafka.
Không cần qua bước prepare_data trung gian.

Schema thực tế:
    machine_usage: machine_id, time_stamp, cpu_util_percent, mem_util_percent,
                                 mem_gps, mkpi, net_in, net_out, disk_io_percent

Usage:
  python kafka_stream/real_data_producer.py
  python kafka_stream/real_data_producer.py --speed 500 --batch-size 200
  python kafka_stream/real_data_producer.py --max-records 1000000
"""

import os
import sys
import csv
import gzip
import json
import time
import glob
import signal
import logging
import argparse
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_CPU,
    DATA_DIR,
)

# ============================================
# Logging
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Graceful shutdown
running = True


def signal_handler(sig, frame):
    global running
    logger.info("\n⏹ Đang dừng producer...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _to_float_or_none(value):
    value = value.strip()
    if value == "":
        return None
    return float(value)


def _to_int_or_none(value):
    value = value.strip()
    if value == "":
        return None
    return int(float(value))


# ============================================
# KAFKA PRODUCER CLASS
# ============================================
class RealDataProducer:
    """
    Producer đọc trực tiếp từ machine_usage files và đẩy vào Kafka.
    """

    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                 speed_factor=100, max_retries=5):
        self.speed_factor = speed_factor
        self.total_sent = 0
        self.total_errors = 0

        # Retry connecting to Kafka
        for attempt in range(1, max_retries + 1):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3,
                    batch_size=32768,        # 32KB batch
                    linger_ms=20,            # đợi 20ms để batch
                    buffer_memory=67108864,  # 64MB buffer
                    compression_type='gzip',
                    max_request_size=5242880,  # 5MB max request
                )
                logger.info(f"✅ Kafka Producer connected: {bootstrap_servers}")
                return
            except NoBrokersAvailable:
                logger.warning(
                    f"⏳ Kafka chưa sẵn sàng (attempt {attempt}/{max_retries}). "
                    f"Thử lại sau 5s..."
                )
                time.sleep(5)

        raise ConnectionError(
            f"Không thể kết nối Kafka sau {max_retries} lần thử. "
            f"Kiểm tra docker compose up -d"
        )

    def send_record(self, topic, key, value):
        """Gửi 1 record vào Kafka topic"""
        try:
            self.producer.send(topic=topic, key=key, value=value)
            self.total_sent += 1
        except KafkaError as e:
            self.total_errors += 1
            if self.total_errors % 1000 == 1:
                logger.error(f"Kafka send error: {e}")

    def stream_from_raw_files(self, raw_dir, batch_size=500, max_records=None):
        """
        Đọc machine_usage files (.csv/.csv.gz) và gửi Kafka.

        machine_usage schema (9 cột, không header):
          0: machine_id
          1: time_stamp
          2: cpu_util_percent
          3: mem_util_percent
          4: mem_gps
          5: mkpi
          6: net_in
          7: net_out
          8: disk_io_percent
        """
        global running

        patterns = [
            os.path.join(raw_dir, "machine_usage-*.csv.gz"),
            os.path.join(raw_dir, "machine_usage*.csv"),
            os.path.join(raw_dir, "machine_usage*.csv.gz"),
        ]
        usage_files = []
        for pattern in patterns:
            usage_files.extend(glob.glob(pattern))
        usage_files = sorted(set(usage_files))

        if not usage_files:
            logger.error(f"❌ Không tìm thấy file machine_usage tại {raw_dir}")
            logger.info("Download trước: bash data/download_data.sh")
            return

        logger.info(f"📂 Tìm thấy {len(usage_files)} file machine_usage")
        if max_records:
            logger.info(f"🔢 Giới hạn: {max_records:,} records")

        start_time = time.time()
        batch_count = 0
        records_in_batch = 0

        for file_idx, data_file in enumerate(usage_files):
            if not running:
                break

            filename = os.path.basename(data_file)
            logger.info(
                f"\n📄 [{file_idx+1}/{len(usage_files)}] Streaming {filename}..."
            )

            try:
                opener = gzip.open if data_file.endswith(".gz") else open
                with opener(data_file, 'rt', encoding='utf-8') as f:
                    reader = csv.reader(f)

                    for row in reader:
                        if not running:
                            break

                        if len(row) < 9:
                            continue

                        # Parse machine_usage row
                        try:
                            machine_id = row[0].strip()
                            time_stamp = float(row[1])
                            cpu_util_percent = _to_int_or_none(row[2])
                            mem_util_percent = _to_int_or_none(row[3])
                            mem_gps = _to_float_or_none(row[4])
                            mkpi = _to_int_or_none(row[5])
                            net_in = _to_float_or_none(row[6])
                            net_out = _to_float_or_none(row[7])
                            disk_io_percent = _to_float_or_none(row[8])
                        except (ValueError, IndexError):
                            continue

                        # Record đúng schema machine_usage
                        record = {
                            'machine_id': machine_id,
                            'time_stamp': time_stamp,
                            'cpu_util_percent': cpu_util_percent,
                            'mem_util_percent': mem_util_percent,
                            'mem_gps': mem_gps,
                            'mkpi': mkpi,
                            'net_in': net_in,
                            'net_out': net_out,
                            'disk_io_percent': disk_io_percent,
                        }

                        # Thêm metadata ingestion
                        record['ingestion_timestamp'] = datetime.now().isoformat()
                        record['source_file'] = filename

                        # Gửi vào Kafka
                        self.send_record(KAFKA_TOPIC_CPU, machine_id, record)
                        records_in_batch += 1

                        # Flush theo batch
                        if records_in_batch >= batch_size:
                            self.producer.flush()
                            batch_count += 1
                            records_in_batch = 0

                            # Delay để simulate real-time
                            # Data gốc 5-phút interval,
                            # speed_factor quy đổi: 300s / speed / batch
                            delay = 300.0 / self.speed_factor / batch_size
                            time.sleep(max(delay, 0.001))

                        # Progress log mỗi 100K records
                        if self.total_sent % 100_000 == 0 and self.total_sent > 0:
                            elapsed = time.time() - start_time
                            rate = self.total_sent / elapsed
                            logger.info(
                                f"  📊 Sent: {self.total_sent:,} | "
                                f"Errors: {self.total_errors:,} | "
                                f"Rate: {rate:,.0f} msg/s"
                            )

                        # Kiểm tra giới hạn
                        if max_records and self.total_sent >= max_records:
                            logger.info(
                                f"🏁 Đạt giới hạn {max_records:,} records"
                            )
                            running = False
                            break

            except Exception as e:
                logger.error(f"❌ Lỗi đọc file {filename}: {e}")
                continue

        # Flush cuối
        self.producer.flush()

        # Báo cáo
        elapsed = time.time() - start_time
        rate = self.total_sent / elapsed if elapsed > 0 else 0

        logger.info("\n" + "=" * 60)
        logger.info("📊 KẾT QUẢ STREAMING")
        logger.info("=" * 60)
        logger.info(f"  Total sent:        {self.total_sent:,}")
        logger.info(f"  Total errors:      {self.total_errors:,}")
        logger.info(f"  Duration:          {elapsed:.1f}s")
        logger.info(f"  Avg rate:          {rate:,.0f} msg/s")
        logger.info(f"  Batches flushed:   {batch_count:,}")
        logger.info("=" * 60)

    def close(self):
        """Đóng producer"""
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed.")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description='Real Data Kafka Producer - machine_usage → Kafka'
    )
    parser.add_argument(
        '--speed', type=int, default=200,
        help='Tốc độ replay (default: 200x, tức 5-min→1.5ms/record)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=500,
        help='Batch size cho flush (default: 500)'
    )
    parser.add_argument(
        '--max-records', type=int, default=None,
        help='Giới hạn số records gửi (default: tất cả). VD: --max-records 1000000'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Thư mục chứa file .csv.gz (default: data/raw)'
    )
    args = parser.parse_args()

    # Xác định thư mục data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = args.data_dir or os.path.join(project_root, DATA_DIR)

    logger.info("=" * 60)
    logger.info("  🚀 REAL DATA KAFKA PRODUCER")
    logger.info("  machine_usage → Kafka")
    logger.info("=" * 60)
    logger.info(f"  Data dir:    {raw_dir}")
    logger.info(f"  Speed:       {args.speed}x")
    logger.info(f"  Batch size:  {args.batch_size}")
    logger.info(f"  Max records: {args.max_records or 'unlimited'}")
    logger.info(f"  Kafka:       {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"  Topic:       {KAFKA_TOPIC_CPU}")
    logger.info("")

    # 1. Tạo producer
    producer = RealDataProducer(speed_factor=args.speed)

    # 2. Stream data
    try:
        producer.stream_from_raw_files(
            raw_dir=raw_dir,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
