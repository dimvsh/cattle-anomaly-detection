"""
Feature engineering module: clean sessions -> bout-merged daily features.

Handles:
- Merging consecutive scanner sessions into drinking bouts (60s gap threshold)
- Computing 4 daily behavioral metrics per animal
"""

import pandas as pd
import numpy as np


def merge_bouts(df, gap_threshold=60):
    """
    Merge consecutive scanner sessions into drinking bouts.

    Sessions separated by less than gap_threshold seconds are considered
    part of the same drinking bout (the animal remained at the drinker).

    Parameters
    ----------
    df : pd.DataFrame
        Clean session data with columns: tag_short, start_dt, end_dt, duration_sec, date.
    gap_threshold : int
        Maximum gap (seconds) between sessions to merge into one bout.

    Returns
    -------
    pd.DataFrame
        Bout-level data with columns: tag_short, bout_id, start_dt, end_dt,
        duration_sec, n_sessions, date.
    """
    df = df.sort_values(['tag_short', 'start_dt']).reset_index(drop=True)

    df['prev_end'] = df.groupby('tag_short')['end_dt'].shift(1)
    df['gap_sec'] = (df['start_dt'] - df['prev_end']).dt.total_seconds()

    df['new_bout'] = (df['gap_sec'] > gap_threshold) | (df['gap_sec'].isna())
    df['bout_id'] = df.groupby('tag_short')['new_bout'].cumsum()

    bouts = df.groupby(['tag_short', 'bout_id']).agg(
        start_dt=('start_dt', 'min'),
        end_dt=('end_dt', 'max'),
        duration_sec=('duration_sec', 'sum'),
        n_sessions=('duration_sec', 'count'),
        date=('date', 'first')
    ).reset_index()

    return bouts


def compute_daily_features(bouts):
    """
    Compute 4 daily behavioral metrics from drinking bouts.

    Metrics:
        1. total_daily_duration — total seconds spent drinking
        2. visit_count — number of drinking bouts
        3. avg_visit_duration — mean bout duration (seconds)
        4. max_absence_hours — longest gap between consecutive bouts (hours)

    Parameters
    ----------
    bouts : pd.DataFrame
        Bout-level data from merge_bouts().

    Returns
    -------
    pd.DataFrame
        One row per animal per day with the 4 features.
    """
    bouts = bouts.sort_values(['tag_short', 'start_dt']).reset_index(drop=True)

    bouts['prev_bout_end'] = bouts.groupby('tag_short')['end_dt'].shift(1)
    bouts['bout_gap_sec'] = (bouts['start_dt'] - bouts['prev_bout_end']).dt.total_seconds()

    daily = bouts.groupby(['tag_short', 'date']).agg(
        total_daily_duration=('duration_sec', 'sum'),
        visit_count=('duration_sec', 'count'),
        avg_visit_duration=('duration_sec', 'mean'),
        max_absence_sec=('bout_gap_sec', 'max')
    ).reset_index()

    daily['max_absence_hours'] = daily['max_absence_sec'] / 3600

    return daily
