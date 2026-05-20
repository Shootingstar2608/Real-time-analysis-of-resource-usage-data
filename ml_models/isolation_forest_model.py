import os
import sys
import pandas as pd
import tarfile
import csv
import logging
import joblib
from io import TextIOWrapper
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE, MODEL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_isolation_forest():
    """Train Isolation Forest for point-anomaly detection.
    Đọc trực tiếp từ file .tar.gz — không cần giải nén ra CSV."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tar_gz_path = os.path.join(project_root, "data", "alibaba", "machine_usage.tar.gz")
    
    if not os.path.exists(tar_gz_path):
        logger.error(f"File not found: {tar_gz_path}")
        logger.error("Download trước: wget -c http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz -P data/alibaba/")
        return

    logger.info(f"Loading data for Isolation Forest training from {tar_gz_path}...")
    records = []
    max_records = 500_000  # Lấy 500K records đầu tiên để train

    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            
            logger.info(f"Reading member: {member.name}")
            f = tar.extractfile(member)
            if f is None:
                continue
            
            text_stream = TextIOWrapper(f, encoding='utf-8')
            reader = csv.reader(text_stream)
            
            for i, row in enumerate(reader):
                if len(records) >= max_records:
                    break
                if len(row) < 9:
                    continue
                try:
                    # 0: machine_id, 1: timestamp, 2: cpu, 3: mem, ..., 8: disk
                    records.append({
                        'cpu_util_percent': float(row[2]) if row[2] else 0.0,
                        'mem_util_percent': float(row[3]) if row[3] else 0.0,
                        'disk_io_percent': float(row[8]) if row[8] else 0.0,
                    })
                except (ValueError, IndexError):
                    continue
            
            text_stream.close()
            
            if len(records) >= max_records:
                break

    df = pd.DataFrame(records)
    logger.info(f"Training on {len(df):,} records...")

    # Initialize and train
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1
    )
    
    clf.fit(df)
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    joblib.dump(clf, model_path)
    
    logger.info(f"Isolation Forest model saved to {model_path}")

if __name__ == "__main__":
    train_isolation_forest()
