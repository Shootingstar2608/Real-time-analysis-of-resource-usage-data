"""
Real Data Kafka Producer
Đọc trực tiếp file .csv.gz (Azure VM Traces) → enrich với vmtable → đẩy vào Kafka.
Không cần qua bước prepare_data trung gian.

Schema thực tế:
  cpu_readings: timestamp_s, vm_id, min_cpu, max_cpu, avg_cpu  (5 cột, không header)
  vmtable:      vm_id, sub_id, deploy_id, created, deleted,
                max_cpu, avg_cpu, p95_cpu, category, cores, memory  (11 cột, không header)

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
from collections import defaultdict

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


# ============================================
# VM TABLE LOADER
# ============================================
def load_vm_table(raw_dir):
    """
    Load vmtable.csv.gz để enrich CPU readings với metadata.
    Trả về dict: vm_id → {category, core_count, memory_gb}

    vmtable schema (11 cột, không header):
      0: encrypted_vm_id
      1: encrypted_subscription_id
      2: encrypted_deployment_id
      3: timestamp_vm_created
      4: timestamp_vm_deleted
      5: max_cpu (lifetime)
      6: avg_cpu (lifetime)
      7: p95_cpu (lifetime)
      8: vm_category (Interactive / Delay-insensitive / Unknown)
      9: vm_virtual_core_count_bucket
     10: vm_memory_bucket (GB)
    """
    vmtable_path = os.path.join(raw_dir, "vmtable.csv.gz")

    if not os.path.exists(vmtable_path):
        logger.warning(f"vmtable.csv.gz không tìm thấy tại {vmtable_path}")
        logger.warning("Producer sẽ chạy không có metadata VM")
        return {}

    logger.info(f"📋 Đang load vmtable từ {vmtable_path}...")
    vm_lookup = {}
    count = 0

    with gzip.open(vmtable_path, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 11:
                continue
            vm_id = row[0].strip()
            try:
                vm_lookup[vm_id] = {
                    'vm_category': row[8].strip() if row[8].strip() else 'Unknown',
                    'vm_core_count': int(row[9]) if row[9].strip() else 0,
                    'vm_memory_gb': int(row[10]) if row[10].strip() else 0,
                }
            except (ValueError, IndexError):
                vm_lookup[vm_id] = {
                    'vm_category': 'Unknown',
                    'vm_core_count': 0,
                    'vm_memory_gb': 0,
                }
            count += 1

    logger.info(f"  ✅ Loaded {count:,} VMs vào lookup table")

    # Thống kê
    categories = defaultdict(int)
    for info in vm_lookup.values():
        categories[info['vm_category']] += 1
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"     {cat}: {cnt:,}")

    return vm_lookup


# ============================================
# KAFKA PRODUCER CLASS
# ============================================
class RealDataProducer:
    """
    Producer đọc trực tiếp từ file .csv.gz và đẩy vào Kafka.
    Mỗi record được enrich với vmtable metadata trước khi gửi.
    """

    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                 speed_factor=100, max_retries=5):
        self.speed_factor = speed_factor
        self.total_sent = 0
        self.total_errors = 0
        self.total_enriched = 0
        self.total_not_enriched = 0

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

    def stream_from_raw_files(self, raw_dir, vm_lookup,
                               batch_size=500, max_records=None):
        """
        Đọc file cpu_readings .csv.gz → enrich → gửi Kafka.

        cpu_readings schema (5 cột, không header):
          0: timestamp (giây, relative, mỗi 300s = 5 phút)
          1: encrypted_vm_id
          2: min_cpu (%)
          3: max_cpu (%)
          4: avg_cpu (%)
        """
        global running

        # Tìm tất cả cpu_readings files
        pattern = os.path.join(raw_dir, "vm_cpu_readings-*.csv.gz")
        cpu_files = sorted(glob.glob(pattern))

        if not cpu_files:
            logger.error(f"❌ Không tìm thấy file cpu_readings tại {raw_dir}")
            logger.info("Download trước: bash data/download_data.sh")
            return

        logger.info(f"📂 Tìm thấy {len(cpu_files)} file CPU readings")
        if max_records:
            logger.info(f"🔢 Giới hạn: {max_records:,} records")

        start_time = time.time()
        batch_count = 0
        records_in_batch = 0

        for file_idx, gz_file in enumerate(cpu_files):
            if not running:
                break

            filename = os.path.basename(gz_file)
            logger.info(
                f"\n📄 [{file_idx+1}/{len(cpu_files)}] Streaming {filename}..."
            )

            try:
                with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                    reader = csv.reader(f)

                    for row in reader:
                        if not running:
                            break

                        if len(row) < 5:
                            continue

                        # Parse CPU readings
                        try:
                            timestamp_s = float(row[0])
                            vm_id = row[1].strip()
                            min_cpu = float(row[2])
                            max_cpu = float(row[3])
                            avg_cpu = float(row[4])
                        except (ValueError, IndexError):
                            continue

                        # Tạo record cơ bản
                        record = {
                            'timestamp': timestamp_s,
                            'vm_id': vm_id,
                            'min_cpu': round(min_cpu, 4),
                            'max_cpu': round(max_cpu, 4),
                            'avg_cpu': round(avg_cpu, 4),
                            'cpu_range': round(max_cpu - min_cpu, 4),
                        }

                        # Enrich với vmtable metadata
                        vm_info = vm_lookup.get(vm_id)
                        if vm_info:
                            record['vm_category'] = vm_info['vm_category']
                            record['vm_core_count'] = vm_info['vm_core_count']
                            record['vm_memory_gb'] = vm_info['vm_memory_gb']
                            self.total_enriched += 1
                        else:
                            record['vm_category'] = 'Unknown'
                            record['vm_core_count'] = 0
                            record['vm_memory_gb'] = 0
                            self.total_not_enriched += 1

                        # Thêm metadata ingestion
                        record['ingestion_timestamp'] = datetime.now().isoformat()
                        record['source_file'] = filename

                        # Gửi vào Kafka
                        self.send_record(KAFKA_TOPIC_CPU, vm_id, record)
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
                                f"Rate: {rate:,.0f} msg/s | "
                                f"Enriched: {self.total_enriched:,}"
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
        logger.info(f"  Enriched (vmtable):{self.total_enriched:,}")
        logger.info(f"  Not enriched:      {self.total_not_enriched:,}")
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
        description='Real Data Kafka Producer - Azure VM Traces → Kafka'
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
    logger.info("  Azure VM Traces → Kafka")
    logger.info("=" * 60)
    logger.info(f"  Data dir:    {raw_dir}")
    logger.info(f"  Speed:       {args.speed}x")
    logger.info(f"  Batch size:  {args.batch_size}")
    logger.info(f"  Max records: {args.max_records or 'unlimited'}")
    logger.info(f"  Kafka:       {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"  Topic:       {KAFKA_TOPIC_CPU}")
    logger.info("")

    # 1. Load VM metadata
    vm_lookup = load_vm_table(raw_dir)

    # 2. Tạo producer
    producer = RealDataProducer(speed_factor=args.speed)

    # 3. Stream data
    try:
        producer.stream_from_raw_files(
            raw_dir=raw_dir,
            vm_lookup=vm_lookup,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
