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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# MODEL 1: LSTM AUTOENCODER FOR ANOMALY DETECTION
# ================================================================

class LSTMEncoder(nn.Module):
    """Encoder: compresses sequence into latent representation"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
    
    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    """Decoder: reconstructs sequence from latent representation"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        output, _ = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series anomaly detection.
    
    How it works:
    1. Encoder compresses a sequence of resource metrics into a 
       fixed-size latent vector
    2. Decoder tries to reconstruct the original sequence
    3. High reconstruction error = anomalous pattern
    
    This captures temporal anomalies that Isolation Forest misses:
    - Gradual memory leaks (slow upward trend)
    - Periodic pattern violations (e.g., no daily cycle)
    - Correlated multi-metric anomalies
    """
    
    def __init__(self, n_features, hidden_size=LSTM_HIDDEN_SIZE, 
                 num_layers=LSTM_NUM_LAYERS):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_length = LSTM_SEQUENCE_LENGTH
        
        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers)
        self.decoder = LSTMDecoder(n_features, hidden_size, num_layers, n_features)
    
    def forward(self, x):
        # Encode
        hidden, cell = self.encoder(x)
        
        # Decode (use input as decoder input for teacher forcing)
        # Reverse sequence for better gradient flow
        decoder_input = torch.flip(x, dims=[1])
        reconstruction = self.decoder(decoder_input, hidden, cell)
        reconstruction = torch.flip(reconstruction, dims=[1])
        
        return reconstruction


class AnomalyDetectorLSTM:
    """
    Complete LSTM Autoencoder pipeline for anomaly detection.
    """
    
    FEATURE_COLUMNS = [
        'min_cpu', 'max_cpu', 'avg_cpu', 'cpu_range',
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
    
    def _prepare_data(self, df, vm_id=None):
        """Prepare data for LSTM: sort by time, create sequences"""
        if vm_id:
            df = df[df['vm_id'] == vm_id].copy()
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        features = df[self.FEATURE_COLUMNS].values
        features_scaled = self.scaler.fit_transform(features)
        
        sequences = self._create_sequences(features_scaled)
        return sequences
    
    def fit(self, df, vm_id=None, epochs=LSTM_EPOCHS):
        """Train the LSTM Autoencoder"""
        logger.info("Preparing sequences for LSTM...")
        sequences = self._prepare_data(df, vm_id)
        
        n_features = sequences.shape[2]
        logger.info(f"  Sequences: {sequences.shape[0]:,}")
        logger.info(f"  Sequence length: {sequences.shape[1]}")
        logger.info(f"  Features: {n_features}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
        
        # Initialize model
        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS
        ).to(device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LSTM_LEARNING_RATE
        )
        criterion = nn.MSELoss()
        
        # Training loop
        logger.info(f"Training LSTM Autoencoder for {epochs} epochs...")
        self.training_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                reconstruction = self.model(batch_x)
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
            reconstructions = self.model(X_tensor)
            errors = torch.mean(
                (X_tensor - reconstructions) ** 2, dim=[1, 2]
            ).cpu().numpy()
        
        self.threshold = np.percentile(errors, LSTM_THRESHOLD_PERCENTILE)
        logger.info(f"  Threshold (p{LSTM_THRESHOLD_PERCENTILE}): {self.threshold:.6f}")
        
        return errors
    
    def predict(self, df, vm_id=None):
        """Detect anomalies in new data"""
        sequences = self._prepare_data(df, vm_id)
        X_tensor = torch.FloatTensor(sequences).to(device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            errors = torch.mean(
                (X_tensor - reconstructions) ** 2, dim=[1, 2]
            ).cpu().numpy()
        
        anomalies = errors > self.threshold
        return anomalies, errors
    
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
        
        self.model = LSTMAutoencoder(n_features=n_features).to(device)
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        self.threshold = np.load(os.path.join(path, "threshold.npy"))
        import pickle
        self.scaler = pickle.load(open(os.path.join(path, "scaler.pkl"), 'rb'))
        logger.info(f"LSTM Autoencoder loaded from {path}/")


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
    
    def __init__(self, target='avg_cpu'):
        self.target = target
        self.feature_cols = [
            'min_cpu', 'max_cpu', 'avg_cpu', 'cpu_range',
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
    
    def fit(self, df, vm_id=None, epochs=FORECAST_EPOCHS):
        """Train forecasting model"""
        if vm_id:
            df = df[df['vm_id'] == vm_id].copy()
        
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

def train_all_models():
    """Train both LSTM models and generate visualizations"""
    import glob
    import gzip
    import csv
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    
    cpu_files = sorted(glob.glob(os.path.join(raw_dir, "vm_cpu_readings-*.csv.gz")))
    if not cpu_files:
        logger.error("No CPU reading files found. Download data first.")
        return
    
    # Load data from first file (sample for LSTM training)
    logger.info(f"Loading data from {os.path.basename(cpu_files[0])}...")
    records = []
    max_records = 200_000  # LSTM trains per-VM, so smaller set is fine
    
    with gzip.open(cpu_files[0], 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                records.append({
                    'timestamp': float(row[0]),
                    'vm_id': row[1].strip(),
                    'min_cpu': float(row[2]),
                    'max_cpu': float(row[3]),
                    'avg_cpu': float(row[4]),
                    'cpu_range': float(row[3]) - float(row[2]),
                })
                if len(records) >= max_records:
                    break
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(records)
    logger.info(f"Total records: {len(df):,}")
    
    # Pick VM with most records for LSTM training (per-VM model)
    vm_counts = df['vm_id'].value_counts()
    sample_vm = vm_counts.index[0]
    vm_data = df[df['vm_id'] == sample_vm].sort_values('timestamp')
    logger.info(f"Training on VM: {sample_vm[:16]}... ({len(vm_data):,} records)")
    
    # ---- Model 1: LSTM Autoencoder ----
    logger.info("\n" + "=" * 50)
    logger.info("  Training LSTM Autoencoder")
    logger.info("=" * 50)
    
    ae_detector = AnomalyDetectorLSTM()
    ae_errors = ae_detector.fit(vm_data, epochs=LSTM_EPOCHS)
    ae_anomalies, ae_test_errors = ae_detector.predict(vm_data)
    ae_detector.save()
    
    # ---- Model 2: LSTM Forecaster ----
    logger.info("\n" + "=" * 50)
    logger.info("  Training LSTM Forecaster (CPU)")
    logger.info("=" * 50)
    
    forecaster = ResourceForecaster(target='avg_cpu')
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
