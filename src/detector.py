"""
Anomaly detection module: load trained model and score daily features.

Uses pre-extracted autoencoder weights (numpy) and StandardScaler to compute
reconstruction error (MSE) for each animal-day. Days with error above
the threshold are flagged as anomalies.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib


def _relu(x):
    return np.maximum(0, x)


def _forward(x, weights):
    """Run autoencoder forward pass using extracted weights."""
    # Encoder
    h = _relu(x @ np.array(weights['encoder_layer0']['weights']) + np.array(weights['encoder_layer0']['bias']))
    h = _relu(h @ np.array(weights['encoder_layer1']['weights']) + np.array(weights['encoder_layer1']['bias']))
    # Decoder
    h = _relu(h @ np.array(weights['decoder_layer0']['weights']) + np.array(weights['decoder_layer0']['bias']))
    h = h @ np.array(weights['decoder_layer1']['weights']) + np.array(weights['decoder_layer1']['bias'])
    return h


class AnomalyDetector:
    """Loads a trained autoencoder model and scores new data for anomalies."""

    def __init__(self):
        self.weights = None
        self.scaler = None
        self.config = None

    def load(self, model_dir):
        """
        Load model artifacts from disk.

        Parameters
        ----------
        model_dir : str
            Path to directory containing weights.json, scaler.pkl, config.json.
        """
        config_path = os.path.join(model_dir, 'config.json')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        weights_path = os.path.join(model_dir, 'weights.json')

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.scaler = joblib.load(scaler_path)

        with open(weights_path, 'r') as f:
            self.weights = json.load(f)

    def score(self, daily_features):
        """
        Score daily features for anomalies.

        Parameters
        ----------
        daily_features : pd.DataFrame
            Must contain the feature columns specified in config.

        Returns
        -------
        pd.DataFrame
            Copy of input with added columns: ae_score, ae_anomaly.
        """
        feature_cols = self.config['feature_cols']
        threshold = self.config['threshold']

        df = daily_features.copy()

        # Fill NaN in max_absence_hours with median (first day per animal may lack it)
        if df['max_absence_hours'].isna().any():
            median_val = df['max_absence_hours'].median()
            df['max_absence_hours'] = df['max_absence_hours'].fillna(median_val)

        X = self.scaler.transform(df[feature_cols])
        X_reconstructed = _forward(X, self.weights)
        mse = np.mean((X - X_reconstructed) ** 2, axis=1)

        df['ae_score'] = mse
        df['ae_anomaly'] = (mse > threshold).astype(int)

        return df
