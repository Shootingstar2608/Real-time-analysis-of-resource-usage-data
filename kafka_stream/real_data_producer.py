"""
Real Data Kafka Producer
Đọc trực tiếp file Alibaba cluster trace data (machine_usage.tar.gz) và đẩy vào Kafka.
KHÔNG CẦN giải nén ra CSV — đọc on-the-fly từ file nén luôn.

Schema thực tế (9 cột, không có header trong file nguyên bản):
  0: machine_id
  1: time_stamp
  2: cpu_util_percent
  3: mem_util_percent
  4: mem_gps
  5: mkpi
  6: net_in
  7: net_out
  8: disk_io_percent

Usage:
  python kafka_stream/real_data_producer.py
  python kafka_stream/real_data_producer.py --speed 500 --batch-size 200
  python kafka_stream/real_data_producer.py --max-records 1000000
"""

import os
import sys
import csv
import json
import time
import tarfile
import signal
import logging
import argparse
from datetime import datetime
from io import TextIOWrapper

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_RESOURCE,
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
    logger.info("\n Đang dừng producer...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================
# KAFKA PRODUCER CLASS
# ============================================
class AlibabaDataProducer:
    """
    Producer đọc trực tiếp từ file .tar.gz của Alibaba (on-the-fly)
    và đẩy vào Kafka. Không cần giải nén ra ổ cứng.
    """

    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, speed_factor=100, max_retries=5):
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
                logger.info(f" Kafka Producer connected: {bootstrap_servers}")
                return
            except NoBrokersAvailable:
                logger.warning(
                    f" Kafka chưa sẵn sàng (attempt {attempt}/{max_retries}). "
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

    def stream_from_tar_gz(self, tar_gz_path, batch_size=500, max_records=None):
        """
        Đọc trực tiếp từ file .tar.gz mà KHÔNG giải nén ra ổ cứng.
        Sử dụng tarfile + TextIOWrapper để stream on-the-fly.
        """
        global running

        if not os.path.exists(tar_gz_path):
            logger.error(f" Không tìm thấy file {tar_gz_path}")
            logger.info("Tải trước: wget -c http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz -P data/alibaba/")
            return

        file_size_gb = os.path.getsize(tar_gz_path) / (1024**3)
        logger.info(f" Đọc trực tiếp từ {tar_gz_path} ({file_size_gb:.2f} GB nén)")
        if max_records:
            logger.info(f" Giới hạn: {max_records:,} records")

        start_time = time.time()
        batch_count = 0
        records_in_batch = 0

        try:
            with tarfile.open(tar_gz_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if not running:
                        break
                    if not member.isfile():
                        continue

                    logger.info(f"\n Streaming member: {member.name} ({member.size / (1024**3):.2f} GB uncompressed)")

                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    text_stream = TextIOWrapper(f, encoding='utf-8')
                    reader = csv.reader(text_stream)

                    for row in reader:
                        if not running:
                            break

                        if len(row) < 9:
                            continue

                        # Parse metrics
                        try:
                            machine_id = row[0].strip()
                            timestamp_s = float(row[1]) if row[1] else 0.0
                            cpu_util = float(row[2]) if row[2] else 0.0
                            mem_util = float(row[3]) if row[3] else 0.0
                            disk_io = float(row[8]) if row[8] else 0.0
                        except (ValueError, IndexError):
                            continue

                        # Tạo record
                        record = {
                            'machine_id': machine_id,
                            'timestamp': timestamp_s,
                            'cpu_util_percent': round(cpu_util, 2),
                            'mem_util_percent': round(mem_util, 2),
                            'disk_io_percent': round(disk_io, 2),
                            'ingestion_timestamp': datetime.now().isoformat(),
                            'source_file': member.name,
                        }

                        # Gửi vào Kafka
                        self.send_record(KAFKA_TOPIC_RESOURCE, machine_id, record)
                        records_in_batch += 1

                        # Flush theo batch
                        if records_in_batch >= batch_size:
                            self.producer.flush()
                            batch_count += 1
                            records_in_batch = 0

                            # Delay để simulate real-time
                            delay = 300.0 / self.speed_factor / batch_size
                            time.sleep(max(delay, 0.001))

                        # Progress log mỗi 100K records
                        if self.total_sent % 100_000 == 0 and self.total_sent > 0:
                            elapsed = time.time() - start_time
                            rate = self.total_sent / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"   Sent: {self.total_sent:,} | "
                                f"Errors: {self.total_errors:,} | "
                                f"Rate: {rate:,.0f} msg/s"
                            )

                        # Kiểm tra giới hạn
                        if max_records and self.total_sent >= max_records:
                            logger.info(f" Đạt giới hạn {max_records:,} records")
                            running = False
                            break

                    text_stream.close()

        except Exception as e:
            logger.error(f" Lỗi đọc file: {e}")

        # Flush cuối
        self.producer.flush()

        # Báo cáo
        elapsed = time.time() - start_time
        rate = self.total_sent / elapsed if elapsed > 0 else 0

        logger.info("\n" + "=" * 60)
        logger.info(" KẾT QUẢ STREAMING")
        logger.info("=" * 60)
        logger.info(f"  Total sent:        {self.total_sent:,}")
        logger.info(f"  Total errors:      {self.total_errors:,}")
        logger.info(f"  Duration:          {elapsed:.1f}s")
        logger.info(f"  Avg rate:          {rate:,.0f} msg/s")
        logger.info(f"  Batches flushed:   {batch_count:,}")
        logger.info("=" * 60)

    def close(self):
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed.")


def main():
    parser = argparse.ArgumentParser(description='Real Data Kafka Producer - Alibaba Data → Kafka')
    parser.add_argument('--speed', type=int, default=1000, help='Tốc độ replay')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size cho flush')
    parser.add_argument('--max-records', type=int, default=None, help='Giới hạn số records gửi')
    args = parser.parse_args()

    # Xác định đường dẫn file tar.gz
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tar_gz_path = os.path.join(project_root, "data", "alibaba", "machine_usage.tar.gz")

    logger.info("=" * 60)
    logger.info("   REAL DATA KAFKA PRODUCER (ALIBABA DATASET)")
    logger.info("   Đọc trực tiếp từ .tar.gz — không cần giải nén")
    logger.info("=" * 60)
    logger.info(f"  Data file:   {tar_gz_path}")
    logger.info(f"  Speed:       {args.speed}x")
    logger.info(f"  Batch size:  {args.batch_size}")
    logger.info(f"  Max records: {args.max_records or 'unlimited'}")
    logger.info(f"  Kafka:       {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"  Topic:       {KAFKA_TOPIC_RESOURCE}")
    logger.info("")

    producer = AlibabaDataProducer(speed_factor=args.speed)
    try:
        producer.stream_from_tar_gz(
            tar_gz_path=tar_gz_path,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
