"""
Anomaly detection module: load trained model and score daily features.

Uses a pre-trained TensorFlow autoencoder and StandardScaler to compute
reconstruction error (MSE) for each animal-day. Days with error above
the threshold are flagged as anomalies.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib


class AnomalyDetector:
    """Loads a trained autoencoder model and scores new data for anomalies."""

    def __init__(self):
        self.autoencoder = None
        self.scaler = None
        self.config = None

    def load(self, model_dir):
        """
        Load model artifacts from disk.

        Parameters
        ----------
        model_dir : str
            Path to directory containing autoencoder.keras, scaler.pkl, config.json.
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        config_path = os.path.join(model_dir, 'config.json')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        model_path = os.path.join(model_dir, 'autoencoder.keras')

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.scaler = joblib.load(scaler_path)
        self.autoencoder = tf.keras.models.load_model(model_path, compile=False)

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
        X_reconstructed = self.autoencoder(X, training=False).numpy()
        mse = np.mean((X - X_reconstructed) ** 2, axis=1)

        df['ae_score'] = mse
        df['ae_anomaly'] = (mse > threshold).astype(int)

        return df
