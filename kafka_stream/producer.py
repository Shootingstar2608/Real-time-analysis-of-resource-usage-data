"""
Kafka Producer - Streams resource usage data in real-time
Reads processed CSV files and publishes to Kafka topics,
simulating real-time data ingestion from VM monitoring agents.
"""

import os
import sys
import json
import time
import glob
import signal
import logging
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_CPU,
    KAFKA_TOPIC_MEMORY,
    PROCESSED_DATA_DIR,
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
    logger.info("Shutting down producer...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ResourceUsageProducer:
    """
    Kafka Producer that streams VM resource usage data.
    Simulates real-time by replaying historical data with configurable speed.
    """
    
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, speed_factor=100):
        """
        Args:
            bootstrap_servers: Kafka broker address
            speed_factor: How fast to replay data (100x = 5min interval → 3sec)
        """
        self.speed_factor = speed_factor
        self.total_sent = 0
        self.total_errors = 0
        
        # Initialize Kafka Producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            batch_size=16384,        # 16KB batch
            linger_ms=10,            # Wait 10ms to batch messages
            buffer_memory=33554432,  # 32MB buffer
            compression_type='gzip',
        )
        logger.info(f"Producer initialized. Server: {bootstrap_servers}")
    
    def delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err:
            self.total_errors += 1
            logger.error(f"Delivery failed: {err}")
        else:
            self.total_sent += 1
    
    def send_record(self, topic, key, value):
        """Send a single record to Kafka topic"""
        try:
            future = self.producer.send(
                topic=topic,
                key=key,
                value=value
            )
            future.add_callback(
                lambda metadata: None  # Success - counted in batch
            ).add_errback(
                lambda exc: logger.error(f"Send failed: {exc}")
            )
            self.total_sent += 1
        except KafkaError as e:
            self.total_errors += 1
            logger.error(f"Failed to send message: {e}")
    
    def stream_from_files(self, data_dir=None, batch_size=100):
        """
        Read processed CSV files and stream to Kafka.
        Simulates real-time by introducing delays between batches.
        """
        global running
        
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                PROCESSED_DATA_DIR
            )
        
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        
        if not csv_files:
            logger.error(f"No CSV files found in {data_dir}")
            logger.info("Run 'python data/prepare_data.py --synthetic' first")
            return
        
        logger.info(f"Found {len(csv_files)} data files to stream")
        
        start_time = time.time()
        
        for file_idx, csv_file in enumerate(csv_files):
            if not running:
                break
                
            logger.info(f"[{file_idx+1}/{len(csv_files)}] "
                       f"Streaming {os.path.basename(csv_file)}...")
            
            # Read in chunks to handle large files
            for chunk in pd.read_csv(csv_file, chunksize=batch_size):
                if not running:
                    break
                
                for _, row in chunk.iterrows():
                    if not running:
                        break
                    
                    record = row.to_dict()
                    
                    # Add ingestion metadata
                    record['ingestion_timestamp'] = datetime.now().isoformat()
                    record['source_file'] = os.path.basename(csv_file)
                    
                    vm_id = str(record.get('vm_id', 'unknown'))
                    
                    # Send CPU metrics to CPU topic
                    cpu_record = {
                        'timestamp': record.get('timestamp'),
                        'vm_id': vm_id,
                        'vm_type': record.get('vm_type', 'unknown'),
                        'vm_core_count': record.get('vm_core_count', 0),
                        'min_cpu': record.get('min_cpu', 0),
                        'max_cpu': record.get('max_cpu', 0),
                        'avg_cpu': record.get('avg_cpu', 0),
                        'p95_cpu': record.get('p95_cpu', 0),
                        'ingestion_timestamp': record['ingestion_timestamp'],
                    }
                    self.send_record(KAFKA_TOPIC_CPU, vm_id, cpu_record)
                    
                    # Send Memory metrics to Memory topic
                    memory_record = {
                        'timestamp': record.get('timestamp'),
                        'vm_id': vm_id,
                        'vm_type': record.get('vm_type', 'unknown'),
                        'vm_memory_gb': record.get('vm_memory_gb', 0),
                        'avg_memory': record.get('avg_memory', 0),
                        'max_memory': record.get('max_memory', 0),
                        'network_in_mbps': record.get('network_in_mbps', 0),
                        'network_out_mbps': record.get('network_out_mbps', 0),
                        'disk_read_mbps': record.get('disk_read_mbps', 0),
                        'disk_write_mbps': record.get('disk_write_mbps', 0),
                        'ingestion_timestamp': record['ingestion_timestamp'],
                    }
                    self.send_record(KAFKA_TOPIC_MEMORY, vm_id, memory_record)
                
                # Flush batch
                self.producer.flush()
                
                # Simulate real-time delay
                # Original interval is 5 min, speed_factor controls replay speed
                delay = (5 * 60) / self.speed_factor / batch_size
                time.sleep(max(delay, 0.01))
                
                # Progress report
                if self.total_sent % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = self.total_sent / elapsed
                    logger.info(
                        f"  Sent: {self.total_sent:,} | "
                        f"Errors: {self.total_errors:,} | "
                        f"Rate: {rate:.0f} msg/s"
                    )
        
        # Final flush
        self.producer.flush()
        elapsed = time.time() - start_time
        
        logger.info("=" * 50)
        logger.info(f"Streaming complete!")
        logger.info(f"  Total sent: {self.total_sent:,}")
        logger.info(f"  Total errors: {self.total_errors:,}")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Avg rate: {self.total_sent/elapsed:.0f} msg/s")
        logger.info("=" * 50)
    
    def close(self):
        """Close producer connection"""
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Resource Usage Producer')
    parser.add_argument('--speed', type=int, default=100,
                       help='Replay speed factor (default: 100)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for streaming (default: 100)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing processed CSV files')
    args = parser.parse_args()
    
    producer = ResourceUsageProducer(speed_factor=args.speed)
    
    try:
        producer.stream_from_files(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.close()
