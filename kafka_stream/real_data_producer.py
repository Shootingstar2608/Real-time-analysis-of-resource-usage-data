"""
Generic real-data Kafka producer.
Supports both Azure VM traces and Alibaba 2018 machine_usage samples.
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_CPU, DATA_DIR, ALIBABA_DATA_DIR
from data.dataset_utils import detect_dataset_type, iter_normalized_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

running = True


def signal_handler(sig, frame):
    global running
    logger.info("\n⏹ Stopping producer...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class RealDataProducer:
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, speed_factor: int = 100, max_retries: int = 5):
        self.speed_factor = speed_factor
        self.total_sent = 0
        self.total_errors = 0

        for attempt in range(1, max_retries + 1):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                    retries=3,
                    batch_size=32768,
                    linger_ms=20,
                    buffer_memory=67108864,
                    compression_type="gzip",
                    max_request_size=5242880,
                )
                logger.info(f"✅ Kafka Producer connected: {bootstrap_servers}")
                return
            except NoBrokersAvailable:
                logger.warning(
                    f"Kafka not ready (attempt {attempt}/{max_retries}). Retrying in 5s..."
                )
                time.sleep(5)

        raise ConnectionError(f"Could not connect to Kafka after {max_retries} attempts.")

    def send_record(self, topic: str, key: str, value: dict) -> None:
        try:
            self.producer.send(topic=topic, key=key, value=value)
            self.total_sent += 1
        except KafkaError as exc:
            self.total_errors += 1
            if self.total_errors <= 5 or self.total_errors % 1000 == 0:
                logger.error(f"Kafka send error: {exc}")

    def stream_records(
        self,
        data_dir: str,
        dataset: str = "auto",
        batch_size: int = 500,
        max_records: Optional[int] = None,
    ) -> None:
        global running

        resolved_dataset = detect_dataset_type(data_dir) if dataset == "auto" else dataset.lower()
        logger.info(f"📦 Dataset detected/resolved: {resolved_dataset}")
        logger.info(f"📂 Data directory: {data_dir}")
        if max_records is not None:
            logger.info(f"🔢 Max records: {max_records:,}")

        start_time = time.time()
        sent_since_flush = 0
        prev_ts = None

        for idx, record in enumerate(
            iter_normalized_records(data_dir=data_dir, dataset=resolved_dataset, max_records=max_records),
            start=1,
        ):
            if not running:
                break

            vm_id = str(record.get("vm_id", "unknown"))
            self.send_record(KAFKA_TOPIC_CPU, vm_id, record)
            sent_since_flush += 1

            current_ts = float(record.get("timestamp", 0.0) or 0.0)
            if sent_since_flush >= batch_size:
                self.producer.flush()
                sent_since_flush = 0

                if prev_ts is not None:
                    delta_seconds = max(current_ts - prev_ts, 0.0)
                    delay = delta_seconds / max(float(self.speed_factor), 1.0)
                    time.sleep(min(max(delay, 0.01), 2.0))
                else:
                    time.sleep(0.05)
                prev_ts = current_ts

            if idx % 10000 == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                logger.info(
                    f"  Sent: {self.total_sent:,} | Errors: {self.total_errors:,} | "
                    f"Rate: {self.total_sent / elapsed:.0f} msg/s"
                )

        self.producer.flush()
        elapsed = max(time.time() - start_time, 1e-9)
        logger.info("=" * 60)
        logger.info("Streaming complete")
        logger.info(f"  Total sent: {self.total_sent:,}")
        logger.info(f"  Total errors: {self.total_errors:,}")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Avg rate: {self.total_sent / elapsed:.0f} msg/s")
        logger.info("=" * 60)

    def close(self):
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed.")


def main():
    parser = argparse.ArgumentParser(description="Kafka producer for Azure or Alibaba resource traces")
    parser.add_argument("--dataset", choices=["auto", "azure", "alibaba"], default="auto")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing raw dataset files")
    parser.add_argument("--speed", type=int, default=100, help="Replay speed factor")
    parser.add_argument("--batch-size", type=int, default=500, help="Kafka flush batch size")
    parser.add_argument("--max-records", type=int, default=None, help="Optional record limit")
    args = parser.parse_args()

    if args.data_dir is None:
        if args.dataset == "alibaba":
            args.data_dir = ALIBABA_DATA_DIR
        elif args.dataset == "azure":
            args.data_dir = DATA_DIR
        else:
            args.data_dir = ALIBABA_DATA_DIR if os.path.exists(ALIBABA_DATA_DIR) else DATA_DIR

    producer = RealDataProducer(speed_factor=args.speed)
    try:
        producer.stream_records(
            data_dir=args.data_dir,
            dataset=args.dataset,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
