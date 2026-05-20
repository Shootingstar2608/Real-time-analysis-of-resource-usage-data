"""
Deep Learning Model: LSTM Autoencoder for Anomaly Detection
+ LSTM for Resource Usage Forecasting

Two models in one file:
1. LSTM Autoencoder: Learns normal patterns, flags anomalies by high 
   reconstruction error
2. LSTM Forecaster: Predicts future CPU/Memory usage (next N time steps)

Insights generated:
- Anomaly detection with temporal context (sequence-aware)
- CPU/Memory usage forecasting (capacity planning)
- Pattern recognition (daily/weekly cycles)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LSTM_SEQUENCE_LENGTH,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_LEARNING_RATE,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_THRESHOLD_PERCENTILE,
    FORECAST_HORIZON,
    FORECAST_HIDDEN_SIZE,
    FORECAST_EPOCHS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ml_models.explainability import plot_attention_weights, extract_shap_values, plot_feature_importance
except ImportError as e:
    logger.warning(f"Could not import explainability module: {e}")
    plot_attention_weights = None
    plot_feature_importance = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# MODEL 1: BiLSTM + ATTENTION AUTOENCODER FOR ANOMALY DETECTION
# ================================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attn_weights = self.attention(lstm_output) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Context vector: weighted sum of lstm outputs
        context = torch.sum(attn_weights * lstm_output, dim=1) # (batch, hidden_size)
        return context, attn_weights


class BiLSTMAttentionAutoencoder(nn.Module):
    """
    BiLSTM + Temporal Attention Autoencoder.
    Replaces the standard LSTMAutoencoder.
    """
    def __init__(self, n_features, hidden_size=LSTM_HIDDEN_SIZE, 
                 num_layers=LSTM_NUM_LAYERS):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_length = LSTM_SEQUENCE_LENGTH
        
        # BiLSTM Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Temporal Attention
        self.attention = TemporalAttention(hidden_size * 2)
        
        # BiLSTM Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, n_features)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        enc_out, _ = self.encoder(x)
        
        context, attn_weights = self.attention(enc_out)
        
        # Repeat context vector for decoder input
        dec_input = context.unsqueeze(1).repeat(1, x.size(1), 1)
        
        dec_out, _ = self.decoder_lstm(dec_input)
        reconstruction = self.fc(dec_out)
        
        return reconstruction, attn_weights


class AnomalyDetectorLSTM:
    """
    Complete BiLSTM Autoencoder pipeline for anomaly detection.
    """
    
    # Features phải khớp với data từ Kafka streaming
    FEATURE_COLUMNS = [
        'cpu_util_percent', 'mem_util_percent', 'disk_io_percent',
    ]
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None
        self.training_losses = []
    
    def _create_sequences(self, data, seq_length=LSTM_SEQUENCE_LENGTH):
        """Create sliding window sequences"""
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    def _prepare_data(self, df, machine_id=None, fit_scaler=True):
        """Prepare data for LSTM: sort by time, create sequences"""
        if machine_id:
            id_col = 'machine_id' if 'machine_id' in df.columns else 'vm_id'
            df = df[df[id_col] == machine_id].copy()
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        features = df[self.FEATURE_COLUMNS].fillna(0).values
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        sequences = self._create_sequences(features_scaled)
        return sequences
    
    def fit(self, df, machine_id=None, epochs=LSTM_EPOCHS):
        """Train the LSTM Autoencoder"""
        logger.info("Preparing sequences for BiLSTM-Attention...")
        sequences = self._prepare_data(df, machine_id, fit_scaler=True)
        
        n_features = sequences.shape[2]
        logger.info(f"  Sequences: {sequences.shape[0]:,}")
        logger.info(f"  Sequence length: {sequences.shape[1]}")
        logger.info(f"  Features: {n_features}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
        
        # Initialize model
        self.model = BiLSTMAttentionAutoencoder(
            n_features=n_features,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS
        ).to(device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LSTM_LEARNING_RATE
        )
        criterion = nn.MSELoss()
        
        # Training loop
        logger.info(f"Training BiLSTM-Attention Autoencoder for {epochs} epochs...")
        self.training_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                reconstruction, _ = self.model(batch_x)
                loss = criterion(reconstruction, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        # Calculate threshold from training data
        self.model.eval()
        with torch.no_grad():
            reconstructions, _ = self.model(X_tensor)
            errors = torch.mean(
                (X_tensor - reconstructions) ** 2, dim=[1, 2]
            ).cpu().numpy()
        
        self.threshold = np.mean(errors) + 3 * np.std(errors)
        logger.info(f"  Threshold (Mean + 3*Std): {self.threshold:.6f}")
        
        return errors
    
    def predict(self, df, machine_id=None):
        """Detect anomalies in new data"""
        sequences = self._prepare_data(df, machine_id, fit_scaler=False)
        X_tensor = torch.FloatTensor(sequences).to(device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructions, attn_weights = self.model(X_tensor)
            errors = torch.mean(
                (X_tensor - reconstructions) ** 2, dim=[1, 2]
            ).cpu().numpy()
        
        anomalies = errors > self.threshold
        return anomalies, errors, attn_weights.cpu().numpy()
    
    def save(self, path=None):
        """Save model"""
        if path is None:
            path = os.path.join(MODEL_DIR, "lstm_autoencoder")
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        np.save(os.path.join(path, "threshold.npy"), self.threshold)
        import pickle
        pickle.dump(self.scaler, open(os.path.join(path, "scaler.pkl"), 'wb'))
        logger.info(f"LSTM Autoencoder saved to {path}/")
    
    def load(self, path=None, n_features=8):
        """Load model"""
        if path is None:
            path = os.path.join(MODEL_DIR, "lstm_autoencoder")
        
        self.model = BiLSTMAttentionAutoencoder(n_features=n_features).to(device)
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        self.threshold = np.load(os.path.join(path, "threshold.npy"))
        import pickle
        self.scaler = pickle.load(open(os.path.join(path, "scaler.pkl"), 'rb'))
        logger.info(f"BiLSTM Autoencoder loaded from {path}/")


# ================================================================
# MODEL 2: LSTM FORECASTER (CPU/Memory Prediction)
# ================================================================

class LSTMForecaster(nn.Module):
    """
    LSTM model for forecasting future resource usage.
    
    Insight: Predict CPU/Memory usage for the next N time steps
    → Enables proactive capacity planning & auto-scaling
    """
    
    def __init__(self, n_features, hidden_size=FORECAST_HIDDEN_SIZE,
                 num_layers=2, forecast_horizon=FORECAST_HORIZON):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use only the last time step's output
        last_output = lstm_out[:, -1, :]
        forecast = self.fc(last_output)
        return forecast


class ResourceForecaster:
    """
    Complete forecasting pipeline.
    Predicts future CPU/Memory for capacity planning.
    """
    
    def __init__(self, target='cpu_util_percent'):
        self.target = target
        self.feature_cols = [
            'cpu_util_percent', 'mem_util_percent', 'disk_io_percent',
        ]
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.training_losses = []
    
    def _prepare_forecast_data(self, df, seq_length=LSTM_SEQUENCE_LENGTH,
                                horizon=FORECAST_HORIZON):
        """Create (input_sequence, future_target) pairs"""
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        features = self.scaler_x.fit_transform(df[self.feature_cols].values)
        target = self.scaler_y.fit_transform(
            df[[self.target]].values
        ).flatten()
        
        X, y = [], []
        for i in range(len(features) - seq_length - horizon + 1):
            X.append(features[i:i + seq_length])
            y.append(target[i + seq_length:i + seq_length + horizon])
        
        return np.array(X), np.array(y)
    
    def fit(self, df, machine_id=None, epochs=FORECAST_EPOCHS):
        """Train forecasting model"""
        if machine_id:
            id_col = 'machine_id' if 'machine_id' in df.columns else 'vm_id'
            df = df[df[id_col] == machine_id].copy()
        
        logger.info(f"Training LSTM Forecaster (target: {self.target})...")
        X, y = self._prepare_forecast_data(df)
        
        # Train/val split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
        logger.info(f"  Forecast horizon: {FORECAST_HORIZON} steps")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
        
        # Model
        n_features = X.shape[2]
        self.model = LSTMForecaster(
            n_features=n_features,
            hidden_size=FORECAST_HIDDEN_SIZE,
            forecast_horizon=FORECAST_HORIZON
        ).to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        
        best_val_loss = float('inf')
        self.training_losses = []
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            scheduler.step(val_loss)
            self.training_losses.append((train_loss, val_loss))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} | "
                           f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(best_state)
        logger.info(f"  Best val loss: {best_val_loss:.6f}")
        
        return X_val, y_val
    
    def predict(self, input_sequence):
        """Predict future values given an input sequence"""
        self.model.eval()
        
        if isinstance(input_sequence, pd.DataFrame):
            features = self.scaler_x.transform(input_sequence[self.feature_cols].values)
            input_sequence = features[-LSTM_SEQUENCE_LENGTH:]
        
        X = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = self.model(X).cpu().numpy()[0]
        
        # Inverse transform
        prediction = self.scaler_y.inverse_transform(
            prediction.reshape(-1, 1)
        ).flatten()
        
        return prediction
    
    def save(self, path=None):
        """Save model"""
        if path is None:
            path = os.path.join(MODEL_DIR, f"lstm_forecaster_{self.target}")
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        import pickle
        pickle.dump(self.scaler_x, open(os.path.join(path, "scaler_x.pkl"), 'wb'))
        pickle.dump(self.scaler_y, open(os.path.join(path, "scaler_y.pkl"), 'wb'))
        logger.info(f"Forecaster saved to {path}/")


# ================================================================
# VISUALIZATION
# ================================================================

def visualize_lstm_results(df, anomalies, errors, threshold,
                           forecast_actual, forecast_pred,
                           training_losses, save_dir="output/plots"):
    """Generate comprehensive visualization for LSTM models"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Deep Learning Analysis - LSTM Models', 
                 fontsize=16, fontweight='bold')
    
    # 1. Reconstruction Error Timeline
    ax = axes[0, 0]
    ax.plot(errors, color='steelblue', alpha=0.7, linewidth=0.5)
    ax.axhline(y=threshold, color='red', linestyle='--', 
               label=f'Threshold ({threshold:.4f})')
    anomaly_idx = np.where(anomalies)[0]
    ax.scatter(anomaly_idx, errors[anomaly_idx], c='red', s=10, 
               label='Anomaly', zorder=5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('LSTM Autoencoder: Reconstruction Error')
    ax.legend()
    
    # 2. Training Loss Curve (Autoencoder)
    ax = axes[0, 1]
    ax.plot(training_losses, color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Autoencoder Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Error Distribution
    ax = axes[0, 2]
    ax.hist(errors[~anomalies], bins=50, alpha=0.6, color='blue', label='Normal')
    ax.hist(errors[anomalies], bins=30, alpha=0.6, color='red', label='Anomaly')
    ax.axvline(x=threshold, color='red', linestyle='--')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # 4. Forecasting: Actual vs Predicted
    ax = axes[1, 0]
    n_show = min(200, len(forecast_actual))
    ax.plot(forecast_actual[:n_show], label='Actual', color='blue', linewidth=1)
    ax.plot(forecast_pred[:n_show], label='Predicted', color='orange', 
            linewidth=1, linestyle='--')
    ax.fill_between(range(n_show), 
                     forecast_pred[:n_show] * 0.9, 
                     forecast_pred[:n_show] * 1.1,
                     alpha=0.2, color='orange', label='±10% band')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('CPU Usage %')
    ax.set_title('CPU Forecast: Actual vs Predicted')
    ax.legend()
    
    # 5. Forecast Error Analysis
    ax = axes[1, 1]
    forecast_errors = forecast_actual - forecast_pred
    ax.hist(forecast_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--')
    mae = np.mean(np.abs(forecast_errors))
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Forecast Error Distribution (MAE: {mae:.2f}%)')
    
    # 6. Anomaly Pattern Breakdown
    ax = axes[1, 2]
    if len(anomaly_idx) > 0:
        # Show anomaly density over time
        window_size = max(len(errors) // 50, 1)
        anomaly_density = pd.Series(anomalies.astype(float)).rolling(
            window=window_size, center=True
        ).mean()
        ax.fill_between(range(len(anomaly_density)), anomaly_density, 
                        alpha=0.6, color='coral')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Anomaly Density')
        ax.set_title('Anomaly Density Over Time')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "lstm_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"LSTM plots saved to {plot_path}")
    
    return plot_path


# ================================================================
# MAIN TRAINING PIPELINE
# ================================================================

def synthesize_anomalies(df, anomaly_ratio=0.05):
    """Inject synthetic anomalies AFTER scaling to guarantee detectability"""
    df = df.copy().reset_index(drop=True)
    labels = np.zeros(len(df))
    n_anomalies = int(len(df) * anomaly_ratio)
    n_events = max(1, n_anomalies // 5)
    cpu_col = 'cpu_util_percent'
    mem_col = 'mem_util_percent'
    for _ in range(n_events):
        start = np.random.randint(0, max(1, len(df) - 10))
        length = np.random.randint(3, 8)
        end = min(len(df), start + length)
        labels[start:end] = 1
        # Spike: set to max value + extra to be clearly out-of-distribution
        # TEACHER FEEDBACK: Focus on RAM (Memory leaks are the real killers, CPU is volatile)
        mem_max = df[mem_col].max()
        mem_idx = df.columns.get_loc(mem_col)
        # Increase memory usage by 30-50%
        df.iloc[start:end, mem_idx] += np.random.uniform(30, 50)
        df.iloc[start:end, mem_idx] = np.clip(df.iloc[start:end, mem_idx], 0, 100.0)
    return df, labels

def calculate_event_wise_f1(true_labels, pred_labels):
    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[-min_len:]
    pred_labels = pred_labels[-min_len:]
    
    adjusted_preds = np.copy(pred_labels)
    in_anomaly = False
    start = 0
    true_anomalies = []
    for i, label in enumerate(true_labels):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            true_anomalies.append((start, i - 1))
    if in_anomaly:
        true_anomalies.append((start, len(true_labels) - 1))
        
    for start, end in true_anomalies:
        if np.any(pred_labels[start:end+1] == 1):
            adjusted_preds[start:end+1] = 1
            
    tp = np.sum((adjusted_preds == 1) & (true_labels == 1))
    fp = np.sum((adjusted_preds == 1) & (true_labels == 0))
    fn = np.sum((adjusted_preds == 0) & (true_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall


def find_optimal_threshold(errors, true_labels, n_candidates=200):
    """Scan candidate thresholds to find the one maximising Event-wise F1."""
    # Align lengths (errors come from sequences, true_labels from raw rows)
    min_len = min(len(errors), len(true_labels))
    errors = errors[-min_len:]
    true_labels = true_labels[-min_len:]

    candidates = np.linspace(errors.min(), errors.max(), n_candidates)
    best_f1, best_thresh = 0.0, candidates[0]
    for thresh in candidates:
        preds = (errors > thresh).astype(int)
        f1, _, _ = calculate_event_wise_f1(true_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def train_all_models():
    """Train both LSTM models and generate visualizations"""
    import glob
    import tarfile
    import csv
    from collections import defaultdict
    from io import TextIOWrapper

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tar_gz_path = os.path.join(project_root, "data", "alibaba", "machine_usage.tar.gz")
    
    if not os.path.exists(tar_gz_path):
        logger.error(f"No Alibaba data found at {tar_gz_path}. Download data first.")
        return
    
    logger.info(f"Loading data from {tar_gz_path} to find a suitable machine...")
    machine_data_map = defaultdict(list)
    TARGET_RECORDS = 10000     # Số lượng bản ghi cần thiết cho 1 machine
    MAX_SCAN_LINES = 10_000_000  # Quét tối đa 10 triệu dòng
    target_machine_id = None

    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if target_machine_id:
                break
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text_stream = TextIOWrapper(f, encoding='utf-8')
            reader = csv.reader(text_stream)
            for i, row in enumerate(reader):
                if i >= MAX_SCAN_LINES:
                    break
                if len(row) < 9:
                    continue
                try:
                    machine_id = row[0].strip()
                    # Schema: machine_id, timestamp, cpu, mem, mem_gps, mkpi, net_in, net_out, disk
                    cpu_util  = float(row[2]) if row[2] else 0.0
                    mem_util  = float(row[3]) if row[3] else 0.0
                    disk_io   = float(row[8]) if row[8] else 0.0
                    machine_data_map[machine_id].append({
                        'timestamp':        float(row[1]) if row[1] else 0.0,
                        'machine_id':       machine_id,
                        'cpu_util_percent': cpu_util,
                        'mem_util_percent': mem_util,
                        'disk_io_percent':  disk_io,
                    })
                    if len(machine_data_map[machine_id]) >= TARGET_RECORDS:
                        target_machine_id = machine_id
                        break
                except (ValueError, IndexError):
                    continue
            text_stream.close()

    if not target_machine_id:
        target_machine_id = max(machine_data_map, key=lambda k: len(machine_data_map[k]))
        if len(machine_data_map[target_machine_id]) < 30:
            logger.error(f"Không tìm thấy machine có đủ 30 bản ghi sau khi quét {MAX_SCAN_LINES} dòng")
            return

    vm_data = pd.DataFrame(machine_data_map[target_machine_id])
    vm_data = vm_data.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Selected machine: {target_machine_id[:16]}... with {len(vm_data)} records.")
    
    # ---- Model 1: BiLSTM Autoencoder ----
    logger.info("\n" + "=" * 50)
    logger.info("  Training BiLSTM-Attention Autoencoder (70/30 & 80/20 Splits)")
    logger.info("=" * 50)
    
    # Split 70/30
    split_70 = int(len(vm_data) * 0.7)
    train_70 = vm_data.iloc[:split_70].copy()
    test_70 = vm_data.iloc[split_70:].copy()
    # Keep training data clean; inject synthetic anomalies only into test for evaluation.
    test_70_eval, labels_test_70 = synthesize_anomalies(test_70.copy(), anomaly_ratio=0.05)
    
    ae_detector_70 = AnomalyDetectorLSTM()
    ae_detector_70.fit(train_70, epochs=LSTM_EPOCHS)
    preds_70, errors_70, attn_70 = ae_detector_70.predict(test_70_eval)
    f1_70, prec_70, rec_70 = calculate_event_wise_f1(labels_test_70, preds_70.astype(int))
    logger.info(f"[70/30 Split] Event-wise F1: {f1_70:.4f} | Precision: {prec_70:.4f} | Recall: {rec_70:.4f}")
    
    # Split 80/20
    split_80 = int(len(vm_data) * 0.8)
    train_80 = vm_data.iloc[:split_80].copy()
    test_80 = vm_data.iloc[split_80:].copy()
    # Keep training data clean; inject synthetic anomalies only into test for evaluation.
    test_80_eval, labels_test_80 = synthesize_anomalies(test_80.copy(), anomaly_ratio=0.05)
    
    ae_detector = AnomalyDetectorLSTM()
    ae_errors = ae_detector.fit(train_80, epochs=LSTM_EPOCHS)
    ae_anomalies, ae_test_errors, ae_attn_weights = ae_detector.predict(test_80_eval)
    f1_80, prec_80, rec_80 = calculate_event_wise_f1(labels_test_80, ae_anomalies.astype(int))
    logger.info(f"[80/20 Split] Event-wise F1: {f1_80:.4f} | Precision: {prec_80:.4f} | Recall: {rec_80:.4f}")
    
    ae_detector.save()
    
    # Plot attention
    try:
        os.makedirs("output/plots", exist_ok=True)
        # Average attention across the batch for the last prediction
        mean_attn = ae_attn_weights[-1].mean(axis=1)  # shape: (seq_length,)
        if plot_attention_weights is not None:
            plot_attention_weights(
                mean_attn, LSTM_SEQUENCE_LENGTH,
                AnomalyDetectorLSTM.FEATURE_COLUMNS,
                "output/plots/attention_weights.png"
            )
            logger.info("Saved attention weights plot to output/plots/attention_weights.png")
    except Exception as e:
        logger.warning(f"Failed to plot attention weights: {e}")
    
    # ---- Model 2: LSTM Forecaster ----
    logger.info("\n" + "=" * 50)
    logger.info("  Training LSTM Forecaster (RAM)")
    logger.info("=" * 50)
    
    forecaster = ResourceForecaster(target='mem_util_percent')
    X_val, y_val = forecaster.fit(vm_data, epochs=FORECAST_EPOCHS)
    forecaster.save()
    
    # Get forecast predictions for visualization
    forecast_preds = []
    for seq in X_val[:200]:
        pred = forecaster.predict(seq)
        forecast_preds.append(pred[0])  # first forecast step
    forecast_preds = np.array(forecast_preds)
    forecast_actual = forecaster.scaler_y.inverse_transform(
        y_val[:200, 0].reshape(-1, 1)
    ).flatten()
    
    # ---- Visualization ----
    visualize_lstm_results(
        df=vm_data,
        anomalies=ae_anomalies,
        errors=ae_test_errors,
        threshold=ae_detector.threshold,
        forecast_actual=forecast_actual,
        forecast_pred=forecast_preds,
        training_losses=ae_detector.training_losses,
    )
    
    logger.info("\n[DONE] All LSTM models trained and saved!")


if __name__ == "__main__":
    train_all_models()
