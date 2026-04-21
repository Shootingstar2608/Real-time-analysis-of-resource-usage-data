import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import unittest

import numpy as np
import pandas as pd

from data.dataset_utils import select_feature_columns
from ml_models.lstm_models import AnomalyDetectorLSTM, ResourceForecaster, LSTM_SEQUENCE_LENGTH


class LSTMSupportTests(unittest.TestCase):
    def _make_df(self, n=80, spike=False):
        t = np.arange(n, dtype=float)
        base_cpu = 30 + 5 * np.sin(t / 5.0)
        base_mem = 40 + 3 * np.cos(t / 6.0)
        if spike:
            base_cpu = base_cpu.copy()
            base_mem = base_mem.copy()
            base_cpu[55:60] += 35
            base_mem[55:60] += 25
        df = pd.DataFrame({
            "timestamp": t,
            "vm_id": ["vmA"] * n,
            "min_cpu": base_cpu - 3,
            "max_cpu": base_cpu + 4,
            "avg_cpu": base_cpu,
            "cpu_range": np.full(n, 7.0),
            "avg_memory": base_mem,
            "network_in_mbps": 10 + np.sin(t / 7.0),
            "network_out_mbps": 8 + np.cos(t / 8.0),
            "disk_io_percent": 4 + np.sin(t / 4.0),
            "mem_gps": 1.5 + 0.1 * np.cos(t / 5.0),
            "mkpi": 12 + np.sin(t / 9.0),
        })
        return df

    def test_feature_selection_prefers_non_constant_metrics(self):
        df = self._make_df()
        features = select_feature_columns(df)
        self.assertIn("avg_cpu", features)
        self.assertIn("avg_memory", features)
        self.assertNotIn("cpu_range", features)

    def test_sequence_preparation_shape(self):
        df = self._make_df(n=64)
        features = select_feature_columns(df)
        model = AnomalyDetectorLSTM(feature_columns=features)
        seqs = model._prepare_data(df, fit_scaler=True)
        self.assertEqual(seqs.shape[0], len(df) - LSTM_SEQUENCE_LENGTH + 1)
        self.assertEqual(seqs.shape[1], LSTM_SEQUENCE_LENGTH)
        self.assertEqual(seqs.shape[2], len(features))

    def test_forecaster_pair_generation_shape(self):
        df = self._make_df(n=96)
        features = select_feature_columns(df)
        forecaster = ResourceForecaster(target="avg_cpu", feature_cols=features)
        X, y = forecaster._prepare_forecast_data(df, fit_scalers=True)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[2], len(features))
        self.assertGreater(X.shape[0], 0)

    @unittest.skipUnless(os.environ.get("RUN_TORCH_TRAINING_TESTS") == "1", "Optional deep training test")
    def test_autoencoder_spike_has_higher_error(self):
        train_df = self._make_df()
        clean_df = self._make_df()
        spike_df = self._make_df(spike=True)
        features = select_feature_columns(train_df)

        model = AnomalyDetectorLSTM(feature_columns=features)
        model.fit(train_df, epochs=1)
        _, clean_errors = model.predict(clean_df)
        _, spike_errors = model.predict(spike_df)

        self.assertGreater(spike_errors.mean(), clean_errors.mean())


if __name__ == "__main__":
    unittest.main()
