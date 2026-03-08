"""
Preprocessing module: raw scanner data -> clean drinking sessions.

Handles:
- Timestamp conversion from epoch milliseconds
- Duration calculation
- Removal of invalid sessions (zero/negative duration, > 30 min)
- Partial day detection and removal
"""

import pandas as pd
import numpy as np


def epoch_ms_to_datetime(ts_str):
    """Convert epoch millisecond string to pandas Timestamp. Takes first 13 digits."""
    try:
        ts_ms = int(str(ts_str)[:13])
        return pd.Timestamp(ts_ms, unit='ms')
    except (ValueError, TypeError):
        return pd.NaT


def load_scanner_csv(filepath, tag_col='tag_id', start_col='start_timestamp',
                     end_col='end_timestamp'):
    """
    Load raw scanner data from CSV and convert timestamps.

    Parameters
    ----------
    filepath : str
        Path to CSV file with scanner records.
    tag_col : str
        Column name for animal RFID tag.
    start_col : str
        Column name for session start timestamp (epoch ms).
    end_col : str
        Column name for session end timestamp (epoch ms).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: tag_id, tag_short, start_dt, end_dt,
        duration_sec, date
    """
    df = pd.read_csv(filepath)

    if 'start_dt' in df.columns and 'end_dt' in df.columns:
        # Already processed (e.g., scanner_data_clean.csv)
        df['start_dt'] = pd.to_datetime(df['start_dt'])
        df['end_dt'] = pd.to_datetime(df['end_dt'])
        if 'duration_sec' not in df.columns:
            df['duration_sec'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()
        if 'date' not in df.columns:
            df['date'] = df['start_dt'].dt.date
        if 'tag_short' not in df.columns and tag_col in df.columns:
            df['tag_short'] = df[tag_col].astype(str).str[-4:]
        df['tag_short'] = df['tag_short'].astype(str).str.split('.').str[0]
        df['date'] = pd.to_datetime(df['date'])
        return df

    # Raw format: epoch millisecond timestamps
    df['start_dt'] = df[start_col].apply(epoch_ms_to_datetime)
    df['end_dt'] = df[end_col].apply(epoch_ms_to_datetime)
    df['duration_sec'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()
    df['date'] = df['start_dt'].dt.date
    df['tag_short'] = df[tag_col].astype(str).str[-4:].str.split('.').str[0]
    df['date'] = pd.to_datetime(df['date'])

    return df


def clean_sessions(df, max_duration_sec=1800, remove_partial_days=True,
                   min_last_hour=12):
    """
    Remove invalid sessions and partial collection days.

    Parameters
    ----------
    df : pd.DataFrame
        Scanner data with start_dt, end_dt, duration_sec, date columns.
    max_duration_sec : int
        Maximum valid session duration in seconds (default: 1800 = 30 min).
    remove_partial_days : bool
        Whether to detect and remove days where data collection was truncated.
    min_last_hour : int
        If the last record of a day is before this hour, the day is considered
        partial (default: 12 = noon).

    Returns
    -------
    pd.DataFrame
        Cleaned session data.
    dict
        Cleaning summary with counts.
    """
    n_original = len(df)
    summary = {'original_sessions': n_original}

    # Remove zero/negative durations
    df = df[df['duration_sec'] > 0].copy()
    summary['removed_zero_duration'] = n_original - len(df)

    # Remove excessively long sessions
    n_before = len(df)
    df = df[df['duration_sec'] <= max_duration_sec].copy()
    summary['removed_long_duration'] = n_before - len(df)

    # Detect and remove partial days
    if remove_partial_days:
        daily_coverage = df.groupby('date').agg(
            last_record=('start_dt', 'max')
        )
        partial_days = daily_coverage[
            daily_coverage['last_record'].dt.hour < min_last_hour
        ].index.tolist()

        if partial_days:
            n_before = len(df)
            df = df[~df['date'].isin(partial_days)].copy()
            summary['partial_days_removed'] = [str(d) for d in partial_days]
            summary['removed_partial_day_sessions'] = n_before - len(df)

    summary['clean_sessions'] = len(df)
    summary['unique_animals'] = df['tag_short'].nunique()
    summary['days'] = df['date'].nunique()

    return df, summary
