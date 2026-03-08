"""
End-to-end pipeline: raw scanner data -> anomaly-scored daily features.

Usage:
    from src.pipeline import run_pipeline
    results = run_pipeline('data/raw/scanner_export.csv')
"""

import os
from .preprocessing import load_scanner_csv, clean_sessions
from .features import merge_bouts, compute_daily_features
from .detector import AnomalyDetector


def run_pipeline(scanner_data_path, model_dir='models', gap_threshold=60,
                 max_duration_sec=1800):
    """
    Run the full anomaly detection pipeline.

    Parameters
    ----------
    scanner_data_path : str
        Path to CSV file with scanner records (raw or pre-cleaned).
    model_dir : str
        Path to directory containing trained model artifacts.
    gap_threshold : int
        Max gap (seconds) between sessions to merge into one bout.
    max_duration_sec : int
        Maximum valid session duration (seconds).

    Returns
    -------
    pd.DataFrame
        Daily features with anomaly scores and flags.
    dict
        Pipeline summary statistics.
    """
    summary = {}

    # Step 1: Load and clean
    df = load_scanner_csv(scanner_data_path)
    summary['loaded_sessions'] = len(df)

    df, clean_summary = clean_sessions(df, max_duration_sec=max_duration_sec)
    summary['cleaning'] = clean_summary

    # Step 2: Merge bouts and compute features
    bouts = merge_bouts(df, gap_threshold=gap_threshold)
    summary['bouts'] = len(bouts)

    daily = compute_daily_features(bouts)
    summary['animal_days'] = len(daily)
    summary['animals'] = daily['tag_short'].nunique()
    summary['days'] = daily['date'].nunique()

    # Step 3: Score anomalies
    detector = AnomalyDetector()
    detector.load(model_dir)
    results = detector.score(daily)

    n_anomalies = results['ae_anomaly'].sum()
    summary['anomalies_flagged'] = int(n_anomalies)
    summary['anomaly_rate'] = float(n_anomalies / len(results))

    return results, summary
