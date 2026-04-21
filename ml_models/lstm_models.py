"""
Deep Learning Model: LSTM Autoencoder for Anomaly Detection
+ LSTM for Resource Usage Forecasting
"""

import os
import sys
import csv
import glob
import gzip
import pickle
import logging
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    DATA_DIR,
    ALIBABA_DATA_DIR,
)
from data.dataset_utils import (
    detect_dataset_type,
    load_aggregated_series_by_timestamp,
    select_feature_columns,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# MODEL 1: LSTM AUTOENCODER FOR ANOMALY DETECTION
# ================================================================

class LSTMEncoder(nn.Module):
    """Encoder: compresses sequence into latent representation."""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    """Decoder: reconstructs sequence from latent representation."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, _ = self.lstm(x, (hidden, cell))
        return self.fc(output)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series anomaly detection.
    """

    def __init__(self, n_features, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_length = LSTM_SEQUENCE_LENGTH

        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers)
        self.decoder = LSTMDecoder(n_features, hidden_size, num_layers, n_features)

    def forward(self, x):
        hidden, cell = self.encoder(x)

        decoder_input = torch.flip(x, dims=[1])
        reconstruction = self.decoder(decoder_input, hidden, cell)
        reconstruction = torch.flip(reconstruction, dims=[1])

        return reconstruction


class AnomalyDetectorLSTM:
    """
    Complete LSTM Autoencoder pipeline for anomaly detection.
    """

    FEATURE_COLUMNS = ["min_cpu", "max_cpu", "avg_cpu", "cpu_range"]

    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns or list(self.FEATURE_COLUMNS)
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None
        self.training_losses = []
        self.n_features = len(self.feature_columns)

    def _create_sequences(self, data, seq_length=LSTM_SEQUENCE_LENGTH):
        """Create sliding window sequences."""
        if len(data) < seq_length:
            return np.empty((0, seq_length, data.shape[1]), dtype=np.float32)

        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])

        return np.asarray(sequences, dtype=np.float32)

    def _prepare_data(self, df, vm_id=None, fit_scaler=False):
        """
        Prepare data for LSTM:
        - optional VM filter
        - sort by time
        - scale features
        - create sliding windows
        """
        if df is None or len(df) == 0:
            return np.empty((0, LSTM_SEQUENCE_LENGTH, len(self.feature_columns)), dtype=np.float32)

        work_df = df.copy()

        if vm_id is not None:
            work_df = work_df[work_df["vm_id"] == vm_id].copy()

        if work_df.empty:
            return np.empty((0, LSTM_SEQUENCE_LENGTH, len(self.feature_columns)), dtype=np.float32)

        missing_cols = [c for c in self.feature_columns if c not in work_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        work_df = work_df.sort_values("timestamp").reset_index(drop=True)
        features = work_df[self.feature_columns].astype(float).values

        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            if not hasattr(self.scaler, "scale_"):
                raise RuntimeError("Scaler is not fitted. Call fit() or load() first.")
            features_scaled = self.scaler.transform(features)

        return self._create_sequences(features_scaled)

    def _reconstruction_errors(self, sequences):
        """Compute reconstruction errors for sequences."""
        if sequences.size == 0:
            return np.array([], dtype=np.float32)

        X_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()

        return errors

    def fit(self, df, vm_id=None, epochs=LSTM_EPOCHS):
        """Train the LSTM Autoencoder."""
        logger.info("Preparing sequences for LSTM Autoencoder...")
        sequences = self._prepare_data(df, vm_id=vm_id, fit_scaler=True)

        if sequences.size == 0:
            raise ValueError(
                f"Not enough records to create sequences of length {LSTM_SEQUENCE_LENGTH}."
            )

        self.n_features = sequences.shape[2]
        logger.info(f"  Sequences: {sequences.shape[0]:,}")
        logger.info(f"  Sequence length: {sequences.shape[1]}")
        logger.info(f"  Features: {self.n_features}")

        X_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

        self.model = LSTMAutoencoder(
            n_features=self.n_features,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)
        criterion = nn.MSELoss()

        logger.info(f"Training LSTM Autoencoder for {epochs} epochs...")
        self.training_losses = []
        best_state = None
        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
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

            if n_batches == 0:
                raise RuntimeError("No training batches were created for LSTM Autoencoder.")

            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = deepcopy(self.model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        errors = self._reconstruction_errors(sequences)
        self.threshold = float(np.percentile(errors, LSTM_THRESHOLD_PERCENTILE))
        logger.info(f"  Threshold (p{LSTM_THRESHOLD_PERCENTILE}): {self.threshold:.6f}")

        return errors

    def predict(self, df, vm_id=None):
        """Detect anomalies in new data."""
        if self.model is None:
            raise RuntimeError("Model is not loaded/trained.")
        if self.threshold is None:
            raise RuntimeError("Threshold is not set. Call fit() or load() first.")

        sequences = self._prepare_data(df, vm_id=vm_id, fit_scaler=False)
        if sequences.size == 0:
            return np.array([], dtype=bool), np.array([], dtype=np.float32)

        errors = self._reconstruction_errors(sequences)
        anomalies = errors > self.threshold
        return anomalies, errors

    def save(self, path=None):
        """Save model, threshold, scaler, and metadata."""
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        path = path or os.path.join(MODEL_DIR, "lstm_autoencoder")
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        np.save(os.path.join(path, "threshold.npy"), np.array([self.threshold], dtype=np.float32))
        np.save(os.path.join(path, "training_losses.npy"), np.array(self.training_losses, dtype=np.float32))

        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        meta = {
            "n_features": self.n_features,
            "feature_columns": self.feature_columns,
            "seq_length": LSTM_SEQUENCE_LENGTH,
            "hidden_size": LSTM_HIDDEN_SIZE,
            "num_layers": LSTM_NUM_LAYERS,
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"LSTM Autoencoder saved to {path}/")

    def load(self, path=None):
        """Load model, threshold, scaler, and metadata."""
        path = path or os.path.join(MODEL_DIR, "lstm_autoencoder")

        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        self.n_features = int(meta["n_features"])
        self.feature_columns = list(meta["feature_columns"])

        self.model = LSTMAutoencoder(
            n_features=self.n_features,
            hidden_size=meta.get("hidden_size", LSTM_HIDDEN_SIZE),
            num_layers=meta.get("num_layers", LSTM_NUM_LAYERS),
        ).to(device)

        self.model.load_state_dict(
            torch.load(os.path.join(path, "model.pth"), map_location=device)
        )
        self.model.eval()

        threshold_arr = np.load(os.path.join(path, "threshold.npy"))
        self.threshold = float(threshold_arr.reshape(-1)[0])

        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        training_loss_path = os.path.join(path, "training_losses.npy")
        if os.path.exists(training_loss_path):
            self.training_losses = np.load(training_loss_path).tolist()
        else:
            self.training_losses = []

        logger.info(f"LSTM Autoencoder loaded from {path}/")


# ================================================================
# MODEL 2: LSTM FORECASTER (CPU Prediction)
# ================================================================

class LSTMForecaster(nn.Module):
    """
    LSTM model for forecasting future resource usage.
    """

    def __init__(self, n_features, hidden_size=FORECAST_HIDDEN_SIZE, num_layers=2, forecast_horizon=FORECAST_HORIZON):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class ResourceForecaster:
    """
    Complete forecasting pipeline.
    Predicts future CPU usage for capacity planning.
    """

    def __init__(self, target="avg_cpu", feature_cols=None):
        self.target = target
        self.feature_cols = list(feature_cols) if feature_cols is not None else ["min_cpu", "max_cpu", "avg_cpu", "cpu_range"]
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.training_losses = []
        self.n_features = len(self.feature_cols)

    def _prepare_forecast_data(self, df, seq_length=LSTM_SEQUENCE_LENGTH, horizon=FORECAST_HORIZON, fit_scalers=False):
        """Create (input_sequence, future_target) pairs."""
        if df is None or len(df) == 0:
            return (
                np.empty((0, seq_length, len(self.feature_cols)), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32),
            )

        work_df = df.sort_values("timestamp").reset_index(drop=True).copy()

        missing_cols = [c for c in self.feature_cols + [self.target] if c not in work_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for forecasting: {missing_cols}")

        feature_values = work_df[self.feature_cols].astype(float).values
        target_values = work_df[[self.target]].astype(float).values

        if fit_scalers:
            features = self.scaler_x.fit_transform(feature_values)
            target = self.scaler_y.fit_transform(target_values).flatten()
        else:
            if not hasattr(self.scaler_x, "scale_") or not hasattr(self.scaler_y, "scale_"):
                raise RuntimeError("Forecaster scalers are not fitted. Call fit() or load() first.")
            features = self.scaler_x.transform(feature_values)
            target = self.scaler_y.transform(target_values).flatten()

        X, y = [], []
        for i in range(len(features) - seq_length - horizon + 1):
            X.append(features[i:i + seq_length])
            y.append(target[i + seq_length:i + seq_length + horizon])

        if len(X) == 0:
            return (
                np.empty((0, seq_length, len(self.feature_cols)), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32),
            )

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def fit(self, df, vm_id=None, epochs=FORECAST_EPOCHS):
        """Train forecasting model."""
        work_df = df.copy()
        if vm_id is not None:
            work_df = work_df[work_df["vm_id"] == vm_id].copy()

        logger.info(f"Training LSTM Forecaster (target: {self.target})...")
        X, y = self._prepare_forecast_data(work_df, fit_scalers=True)

        if len(X) < 2:
            raise ValueError(
                f"Not enough data to create forecasting samples "
                f"(need at least sequence_length + horizon rows)."
            )

        split = max(int(0.8 * len(X)), 1)
        if split >= len(X):
            split = len(X) - 1

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
        logger.info(f"  Forecast horizon: {FORECAST_HORIZON} steps")

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

        self.n_features = X.shape[2]
        self.model = LSTMForecaster(
            n_features=self.n_features,
            hidden_size=FORECAST_HIDDEN_SIZE,
            forecast_horizon=FORECAST_HORIZON,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        best_state = None
        self.training_losses = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
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

            if n_batches == 0:
                raise RuntimeError("No training batches were created for LSTM Forecaster.")

            train_loss = epoch_loss / n_batches

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            scheduler.step(val_loss)
            self.training_losses.append((train_loss, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info(f"  Best val loss: {best_val_loss:.6f}")
        return X_val, y_val

    def predict(self, input_sequence):
        """Predict future values given an input sequence."""
        if self.model is None:
            raise RuntimeError("Forecast model is not loaded/trained.")

        self.model.eval()

        if isinstance(input_sequence, pd.DataFrame):
            features = self.scaler_x.transform(input_sequence[self.feature_cols].astype(float).values)
            input_sequence = features[-LSTM_SEQUENCE_LENGTH:]
        else:
            input_sequence = np.asarray(input_sequence, dtype=np.float32)
            if input_sequence.ndim == 3:
                input_sequence = input_sequence[0]

        if input_sequence.ndim != 2:
            raise ValueError("input_sequence must have shape (seq_len, n_features).")

        if input_sequence.shape[0] < LSTM_SEQUENCE_LENGTH:
            raise ValueError(
                f"input_sequence has only {input_sequence.shape[0]} rows, "
                f"but {LSTM_SEQUENCE_LENGTH} are required."
            )

        input_sequence = input_sequence[-LSTM_SEQUENCE_LENGTH:]
        X = torch.tensor(input_sequence, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(X).cpu().numpy()[0]

        prediction = self.scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
        return prediction

    def save(self, path=None):
        """Save model, scalers, and metadata."""
        if self.model is None:
            raise RuntimeError("No trained forecast model to save.")

        path = path or os.path.join(MODEL_DIR, f"lstm_forecaster_{self.target}")
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        np.save(os.path.join(path, "training_losses.npy"), np.array(self.training_losses, dtype=object))

        with open(os.path.join(path, "scaler_x.pkl"), "wb") as f:
            pickle.dump(self.scaler_x, f)
        with open(os.path.join(path, "scaler_y.pkl"), "wb") as f:
            pickle.dump(self.scaler_y, f)

        meta = {
            "target": self.target,
            "feature_cols": self.feature_cols,
            "n_features": self.n_features,
            "forecast_horizon": FORECAST_HORIZON,
            "hidden_size": FORECAST_HIDDEN_SIZE,
            "num_layers": 2,
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"Forecaster saved to {path}/")

    def load(self, path=None):
        """Load forecaster, scalers, and metadata."""
        path = path or os.path.join(MODEL_DIR, f"lstm_forecaster_{self.target}")

        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        self.target = meta["target"]
        self.feature_cols = list(meta["feature_cols"])
        self.n_features = int(meta["n_features"])

        self.model = LSTMForecaster(
            n_features=self.n_features,
            hidden_size=meta.get("hidden_size", FORECAST_HIDDEN_SIZE),
            num_layers=meta.get("num_layers", 2),
            forecast_horizon=meta.get("forecast_horizon", FORECAST_HORIZON),
        ).to(device)

        self.model.load_state_dict(
            torch.load(os.path.join(path, "model.pth"), map_location=device)
        )
        self.model.eval()

        with open(os.path.join(path, "scaler_x.pkl"), "rb") as f:
            self.scaler_x = pickle.load(f)
        with open(os.path.join(path, "scaler_y.pkl"), "rb") as f:
            self.scaler_y = pickle.load(f)

        training_loss_path = os.path.join(path, "training_losses.npy")
        if os.path.exists(training_loss_path):
            self.training_losses = np.load(training_loss_path, allow_pickle=True).tolist()
        else:
            self.training_losses = []

        logger.info(f"Forecaster loaded from {path}/")


# ================================================================
# VISUALIZATION
# ================================================================

def visualize_lstm_results(
    df,
    anomalies,
    errors,
    threshold,
    forecast_actual,
    forecast_pred,
    training_losses,
    save_dir="output/plots",
):
    """Generate visualization for LSTM models."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Deep Learning Analysis - LSTM Models", fontsize=16, fontweight="bold")

    # 1. Reconstruction Error Timeline
    ax = axes[0, 0]
    if len(errors) > 0:
        ax.plot(errors, alpha=0.7, linewidth=0.7)
        ax.axhline(y=threshold, linestyle="--", label=f"Threshold ({threshold:.4f})")
        anomaly_idx = np.where(anomalies)[0]
        if len(anomaly_idx) > 0:
            ax.scatter(anomaly_idx, errors[anomaly_idx], s=10, label="Anomaly", zorder=5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No autoencoder results", ha="center", va="center")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("LSTM Autoencoder: Reconstruction Error")

    # 2. Training Loss Curve
    ax = axes[0, 1]
    if len(training_losses) > 0:
        ax.plot(training_losses, linewidth=1.5)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No training losses", ha="center", va="center")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")

    # 3. Error Distribution
    ax = axes[0, 2]
    if len(errors) > 0:
        normal_mask = ~anomalies
        if normal_mask.any():
            ax.hist(errors[normal_mask], bins=50, alpha=0.6, label="Normal")
        if anomalies.any():
            ax.hist(errors[anomalies], bins=30, alpha=0.6, label="Anomaly")
        ax.axvline(x=threshold, linestyle="--")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No error distribution", ha="center", va="center")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")

    # 4. Forecasting: Actual vs Predicted
    ax = axes[1, 0]
    if len(forecast_actual) > 0 and len(forecast_pred) > 0:
        n_show = min(200, len(forecast_actual), len(forecast_pred))
        ax.plot(forecast_actual[:n_show], label="Actual", linewidth=1)
        ax.plot(forecast_pred[:n_show], label="Predicted", linewidth=1, linestyle="--")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No forecast results", ha="center", va="center")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("CPU Usage %")
    ax.set_title("CPU Forecast: Actual vs Predicted")

    # 5. Forecast Error Analysis
    ax = axes[1, 1]
    if len(forecast_actual) > 0 and len(forecast_pred) > 0:
        n_show = min(len(forecast_actual), len(forecast_pred))
        forecast_errors = forecast_actual[:n_show] - forecast_pred[:n_show]
        mae = np.mean(np.abs(forecast_errors))
        ax.hist(forecast_errors, bins=50, alpha=0.7)
        ax.axvline(x=0, linestyle="--")
        ax.set_title(f"Forecast Error Distribution (MAE: {mae:.2f}%)")
    else:
        ax.text(0.5, 0.5, "No forecast error results", ha="center", va="center")
        ax.set_title("Forecast Error Distribution")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")

    # 6. Anomaly Density
    ax = axes[1, 2]
    if len(errors) > 0 and anomalies.any():
        window_size = max(len(errors) // 50, 1)
        anomaly_density = (
            pd.Series(anomalies.astype(float)).rolling(window=window_size, center=True).mean()
        )
        ax.fill_between(range(len(anomaly_density)), anomaly_density, alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No anomaly density to show", ha="center", va="center")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Anomaly Density")
    ax.set_title("Anomaly Density Over Time")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "lstm_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"LSTM plots saved to {plot_path}")
    return plot_path


# ================================================================
# DATA LOADING / TRAINING ENTRYPOINT
# ================================================================


def load_global_aggregated_series(raw_dir, dataset="auto", target_unique_timestamps=None, max_records=None):
    """Dataset-agnostic aggregated time series loader."""
    if target_unique_timestamps is None:
        target_unique_timestamps = max(LSTM_SEQUENCE_LENGTH + FORECAST_HORIZON + 20, 50)

    df = load_aggregated_series_by_timestamp(
        data_dir=raw_dir,
        dataset=dataset,
        target_unique_timestamps=target_unique_timestamps,
        max_records=max_records,
    )
    if df.empty:
        raise ValueError(f"No normalized records could be loaded from {raw_dir}")
    return df


def select_training_vm(df):
    """Pick a VM with enough records for both autoencoder and forecasting."""
    min_required = LSTM_SEQUENCE_LENGTH + FORECAST_HORIZON + 10
    vm_counts = df["vm_id"].value_counts()

    eligible = vm_counts[vm_counts >= min_required]
    if not eligible.empty:
        return eligible.index[0]

    if not vm_counts.empty:
        return vm_counts.index[0]

    raise ValueError("No VM records available for training.")


def train_all_models(dataset="auto", data_dir=None, target_unique_timestamps=None, max_records=None):
    """Train both LSTM models and generate visualizations."""
    if data_dir is None:
        if dataset == "alibaba":
            data_dir = ALIBABA_DATA_DIR
        elif dataset == "azure":
            data_dir = DATA_DIR
        else:
            data_dir = ALIBABA_DATA_DIR if os.path.exists(ALIBABA_DATA_DIR) else DATA_DIR

    resolved_dataset = detect_dataset_type(data_dir) if dataset == "auto" else dataset

    agg_df = load_global_aggregated_series(
        data_dir,
        dataset=resolved_dataset,
        target_unique_timestamps=target_unique_timestamps,
        max_records=max_records,
    )

    if agg_df.empty:
        raise ValueError("Aggregated dataframe is empty.")

    feature_cols = select_feature_columns(agg_df)
    logger.info(f"Resolved dataset: {resolved_dataset}")
    logger.info(f"Training data directory: {data_dir}")
    logger.info(f"Global aggregated time-series length: {len(agg_df):,}")
    logger.info(f"Selected feature columns: {feature_cols}")

    min_required_ae = LSTM_SEQUENCE_LENGTH
    min_required_fc = LSTM_SEQUENCE_LENGTH + FORECAST_HORIZON

    if len(agg_df) < min_required_ae:
        raise ValueError(
            f"Not enough aggregated timestamps for autoencoder. "
            f"Need at least {min_required_ae}, got {len(agg_df)}."
        )

    logger.info("\n" + "=" * 50)
    logger.info("  Training LSTM Autoencoder")
    logger.info("=" * 50)

    ae_detector = AnomalyDetectorLSTM(feature_columns=feature_cols)
    ae_detector.fit(agg_df, epochs=LSTM_EPOCHS)
    ae_anomalies, ae_test_errors = ae_detector.predict(agg_df)
    ae_detector.save()

    logger.info("\n" + "=" * 50)
    logger.info("  Training LSTM Forecaster (CPU)")
    logger.info("=" * 50)

    forecast_actual = np.array([])
    forecast_preds = np.array([])

    if len(agg_df) >= min_required_fc + 5:
        forecaster = ResourceForecaster(target="avg_cpu", feature_cols=feature_cols)
        X_val, y_val = forecaster.fit(agg_df, epochs=FORECAST_EPOCHS)
        forecaster.save()

        preds = []
        n_forecast_samples = min(200, len(X_val))
        for seq in X_val[:n_forecast_samples]:
            pred = forecaster.predict(seq)
            preds.append(pred[0])

        forecast_preds = np.array(preds)
        forecast_actual = forecaster.scaler_y.inverse_transform(
            y_val[:n_forecast_samples, 0].reshape(-1, 1)
        ).flatten()
    else:
        logger.warning(
            "Not enough aggregated timestamps for forecasting model. "
            "Skipping forecaster training."
        )

    visualize_lstm_results(
        df=agg_df,
        anomalies=ae_anomalies,
        errors=ae_test_errors,
        threshold=ae_detector.threshold,
        forecast_actual=forecast_actual,
        forecast_pred=forecast_preds,
        training_losses=ae_detector.training_losses,
    )

    logger.info("\n[DONE] All LSTM models trained and saved!")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM models on Azure or Alibaba traces")
    parser.add_argument("--dataset", choices=["auto", "azure", "alibaba"], default="auto")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--target-unique-timestamps", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    train_all_models(
        dataset=args.dataset,
        data_dir=args.data_dir,
        target_unique_timestamps=args.target_unique_timestamps,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
