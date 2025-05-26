# core.py

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

########################################
# VIBRATO DETECTION AND ANALYSIS
########################################

def detect_vibrato_cycles(times, cents_raw, cents_filtered, prominence=5, distance=5):
    valid = ~np.isnan(cents_filtered)
    t_valid = times[valid]
    filt = cents_filtered[valid]
    raw = cents_raw[valid]

    peaks, _ = find_peaks(filt, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-1 * filt, prominence=prominence, distance=distance)
    all_idx = np.sort(np.concatenate((peaks, troughs)))

    cycle_times = []
    cycle_extents = []  # half-extents

    for i in range(len(all_idx) - 1):
        i1, i2 = all_idx[i], all_idx[i + 1]
        dt = t_valid[i2] - t_valid[i1]
        full_amp = abs(filt[i2] - filt[i1])
        half_amp = full_amp / 2.0
        cycle_times.append(dt)
        cycle_extents.append(half_amp)

    return peaks, troughs, np.array(cycle_times), np.array(cycle_extents), t_valid, raw, all_idx


def compute_cycle_parameters(times_valid, raw_values, extents, all_idx):
    cycle_params = []
    for i in range(len(extents)):
        idx1 = all_idx[i]
        idx2 = all_idx[i + 1]
        t1 = times_valid[idx1]
        t2 = times_valid[idx2]
        center_cents = (raw_values[idx1] + raw_values[idx2]) / 2.0
        center_pitch = 2 ** (center_cents / 1200) * 440
        cycle_params.append({
            'cycle_time': t2 - t1,
            'half_extent_cents': extents[i],
            'center_pitch': center_pitch
        })
    return cycle_params


def filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0):
    return [cp for cp in cycle_params if
            (cp['half_extent_cents'] >= min_half_extent and cp['half_extent_cents'] <= max_half_extent)]


def compute_jitter_metrics(periods):
    N = len(periods)
    if N < 2:
        return {
            'jitter_local_percent': np.nan,
            'jitter_local_abs_ms': np.nan,
            'jitter_rap_percent': np.nan,
            'jitter_ppq5_percent': np.nan,
            'jitter_ddp_percent': np.nan
        }
    mean_period = np.nanmean(periods)
    diffs = np.abs(np.diff(periods))
    jitter_local_percent = np.nanmean(diffs) / mean_period * 100
    jitter_local_abs_ms = np.nanmean(diffs) * 1000
    if N < 3:
        jitter_rap_percent = np.nan
    else:
        rap_diffs = []
        for i in range(1, N - 1):
            avg_three = np.nanmean(periods[i - 1:i + 2])
            rap_diffs.append(abs(periods[i] - avg_three))
        jitter_rap_percent = np.nanmean(rap_diffs) / mean_period * 100
    if N < 5:
        jitter_ppq5_percent = np.nan
    else:
        ppq5_diffs = []
        for i in range(2, N - 2):
            avg_five = np.nanmean(periods[i - 2:i + 3])
            ppq5_diffs.append(abs(periods[i] - avg_five))
        jitter_ppq5_percent = np.nanmean(ppq5_diffs) / mean_period * 100
    if N < 3:
        jitter_ddp_percent = np.nan
    else:
        ddp_diffs = []
        for i in range(1, N - 1):
            diff1 = periods[i] - periods[i - 1]
            diff2 = periods[i + 1] - periods[i]
            ddp_diffs.append(abs(diff1 - diff2))
        jitter_ddp_percent = np.nanmean(ddp_diffs) / mean_period * 100
    return {
        'jitter_local_percent': jitter_local_percent,
        'jitter_local_abs_ms': jitter_local_abs_ms,
        'jitter_rap_percent': jitter_rap_percent,
        'jitter_ppq5_percent': jitter_ppq5_percent,
        'jitter_ddp_percent': jitter_ddp_percent
    }


def compute_cv(cycle_times, cycle_extents):
    """
    Computes the Coefficient of Variation (CV) for vibrato rates and extents.

    Parameters:
        cycle_times (array-like): Array of half-cycle times (s).
        cycle_extents (array-like): Array of half-extent amplitudes (cents).

    Returns:
        tuple: (cv_rate, cv_extent) as percentages, or (np.nan, np.nan) if invalid.

    Notes:
        - Vibrato rate is computed as 1 / (2 * cycle_time) to account for two half-cycles per full cycle.
        - CV = (standard deviation / mean) * 100, using sample standard deviation (ddof=1).
        - Requires at least 2 valid points per array for computation.
    """
    cycle_times = np.asarray(cycle_times, dtype=float)
    cycle_extents = np.asarray(cycle_extents, dtype=float)

    valid_times = ~np.isnan(cycle_times) & (cycle_times > 0)
    valid_extents = ~np.isnan(cycle_extents) & (cycle_extents > 0)
    if np.sum(valid_times) < 2 or np.sum(valid_extents) < 2:
        return np.nan, np.nan

    vibrato_rates = 1 / (2 * cycle_times[valid_times])
    cv_rate = (np.nanstd(vibrato_rates, ddof=1) / np.nanmean(vibrato_rates)) * 100
    cv_extent = (np.nanstd(cycle_extents[valid_extents], ddof=1) / np.nanmean(cycle_extents[valid_extents])) * 100
    return cv_rate, cv_extent


def analyze_vibrato(cycle_times, cycle_extents):
    vibrato_rates = 1 / (2 * cycle_times)
    avg_rate = np.nanmean(vibrato_rates)
    stdev_rate = np.nanstd(vibrato_rates)
    median_rate = np.nanmedian(vibrato_rates)
    avg_extent = np.nanmean(cycle_extents)
    stdev_extent = np.nanstd(cycle_extents)
    median_extent = np.nanmedian(cycle_extents)
    jitter = np.nanstd(cycle_times) / np.nanmean(cycle_times)
    return {
        'mean_rate': avg_rate,
        'stdev_rate': stdev_rate,
        'median_rate': median_rate,
        'mean_extent': avg_extent,
        'stdev_extent': stdev_extent,
        'median_extent': median_extent,
        'jitter': jitter
    }


def create_vibrato_dataframe(cycle_params):
    return pd.DataFrame(cycle_params)


def smooth_vibrato_parameters(df, window_size=3):
    df_numeric = df.select_dtypes(include=[np.number])
    return df_numeric.rolling(window=window_size, center=True).agg(['mean', 'std'])
