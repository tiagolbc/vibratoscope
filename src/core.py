# core.py

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

########################################
# INTERNAL HELPERS
########################################

class _ExtremaIndexArray(np.ndarray):
    """
    Integer extrema indices carrying refined sub-sample metadata.

    The class behaves as a normal NumPy integer array, so the legacy GUI and
    existing callers remain compatible. The additional metadata allows
    compute_cycle_parameters() to use the same refined extrema times and raw
    contour values already calculated by detect_vibrato_cycles().
    """

    def __new__(cls, values, refined_times=None, refined_raw_values=None):
        obj = np.asarray(values, dtype=int).view(cls)
        obj.refined_times = (
            None if refined_times is None
            else np.asarray(refined_times, dtype=float)
        )
        obj.refined_raw_values = (
            None if refined_raw_values is None
            else np.asarray(refined_raw_values, dtype=float)
        )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.refined_times = getattr(obj, "refined_times", None)
        self.refined_raw_values = getattr(obj, "refined_raw_values", None)


def _quadratic_extremum_time(times, values, index):
    """
    Refine an extremum time by fitting a parabola to the point and its
    immediate neighbours.

    The calculation uses the actual time coordinates, so it remains valid even
    when the nominal 100 Hz grid is not represented by exactly 10 ms steps.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    index = int(index)

    if index <= 0 or index >= len(values) - 1:
        return float(times[index])

    x = times[index - 1:index + 2]
    y = values[index - 1:index + 2]

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return float(times[index])

    # Centre x values to improve numerical conditioning.
    x0 = float(x[1])
    xc = x - x0

    try:
        a, b, _ = np.polyfit(xc, y, 2)
    except Exception:
        return float(times[index])

    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-15:
        return float(times[index])

    vertex_offset = -b / (2.0 * a)
    lower = float(xc[0])
    upper = float(xc[-1])

    if not np.isfinite(vertex_offset) or vertex_offset < lower or vertex_offset > upper:
        return float(times[index])

    return x0 + float(vertex_offset)


def _quadratic_value_at_time(times, values, index, target_time):
    """
    Evaluate the local raw contour at a refined extremum time.

    A three-point quadratic interpolation is preferred. Linear interpolation
    over the finite contour is used as a safe fallback.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    index = int(index)
    target_time = float(target_time)

    if 0 < index < len(values) - 1:
        x = times[index - 1:index + 2]
        y = values[index - 1:index + 2]

        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            x0 = float(x[1])
            xc = x - x0
            try:
                coeffs = np.polyfit(xc, y, 2)
                value = np.polyval(coeffs, target_time - x0)
                if np.isfinite(value):
                    return float(value)
            except Exception:
                pass

    finite = np.isfinite(times) & np.isfinite(values)
    if np.sum(finite) >= 2:
        return float(np.interp(target_time, times[finite], values[finite]))

    if 0 <= index < len(values) and np.isfinite(values[index]):
        return float(values[index])

    return np.nan


########################################
# VIBRATO DETECTION AND ANALYSIS
########################################

def detect_vibrato_cycles(times, cents_raw, cents_filtered, prominence=5, distance=5):
    """
    Detect alternating extrema on the filtered contour.

    Rate:
        Extremum times are refined to sub-sample precision by local quadratic
        interpolation, removing integer-grid timing quantisation.

    Half-extent:
        Extrema are located on the filtered contour, but amplitude is measured
        on the raw centred contour at the refined extremum times.

    The public function signature and seven return values are unchanged.
    """
    times = np.asarray(times, dtype=float)
    cents_raw = np.asarray(cents_raw, dtype=float)
    cents_filtered = np.asarray(cents_filtered, dtype=float)

    if not (len(times) == len(cents_raw) == len(cents_filtered)):
        raise ValueError(
            "times, cents_raw, and cents_filtered must have the same length."
        )

    valid = np.isfinite(times) & np.isfinite(cents_filtered)
    t_valid = times[valid]
    filt = cents_filtered[valid]
    raw = cents_raw[valid]

    peaks, _ = find_peaks(filt, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-1 * filt, prominence=prominence, distance=distance)
    integer_indices = np.sort(np.concatenate((peaks, troughs))).astype(int)

    if len(integer_indices) == 0:
        all_idx = _ExtremaIndexArray(
            integer_indices,
            refined_times=np.array([], dtype=float),
            refined_raw_values=np.array([], dtype=float),
        )
        return (
            peaks,
            troughs,
            np.array([], dtype=float),
            np.array([], dtype=float),
            t_valid,
            raw,
            all_idx,
        )

    refined_times = np.asarray(
        [_quadratic_extremum_time(t_valid, filt, idx) for idx in integer_indices],
        dtype=float,
    )

    refined_raw_values = np.asarray(
        [
            _quadratic_value_at_time(t_valid, raw, idx, refined_time)
            for idx, refined_time in zip(integer_indices, refined_times)
        ],
        dtype=float,
    )

    all_idx = _ExtremaIndexArray(
        integer_indices,
        refined_times=refined_times,
        refined_raw_values=refined_raw_values,
    )

    if len(integer_indices) < 2:
        cycle_times = np.array([], dtype=float)
        cycle_extents = np.array([], dtype=float)
    else:
        cycle_times = np.diff(refined_times)
        cycle_extents = 0.5 * np.abs(np.diff(refined_raw_values))

    return (
        peaks,
        troughs,
        np.asarray(cycle_times, dtype=float),
        np.asarray(cycle_extents, dtype=float),
        t_valid,
        raw,
        all_idx,
    )


def compute_cycle_parameters(times_valid, raw_values, extents, all_idx):
    """
    Build the legacy per-half-cycle dictionaries.

    When all_idx comes from the corrected detect_vibrato_cycles(), refined
    times and raw contour values are used automatically. Ordinary integer
    arrays remain supported for backward compatibility.
    """
    times_valid = np.asarray(times_valid, dtype=float)
    raw_values = np.asarray(raw_values, dtype=float)
    extents = np.asarray(extents, dtype=float)

    refined_times = getattr(all_idx, "refined_times", None)
    refined_raw_values = getattr(all_idx, "refined_raw_values", None)
    index_array = np.asarray(all_idx, dtype=int)

    use_refined_times = (
        refined_times is not None
        and len(refined_times) == len(index_array)
        and np.all(np.isfinite(refined_times))
    )
    use_refined_raw = (
        refined_raw_values is not None
        and len(refined_raw_values) == len(index_array)
    )

    cycle_params = []

    for i in range(min(len(extents), max(0, len(index_array) - 1))):
        idx1 = index_array[i]
        idx2 = index_array[i + 1]

        if use_refined_times:
            t1 = float(refined_times[i])
            t2 = float(refined_times[i + 1])
        else:
            t1 = float(times_valid[idx1])
            t2 = float(times_valid[idx2])

        if use_refined_raw:
            raw1 = float(refined_raw_values[i])
            raw2 = float(refined_raw_values[i + 1])
        else:
            raw1 = float(raw_values[idx1])
            raw2 = float(raw_values[idx2])

        center_time_s = (t1 + t2) / 2.0
        center_cents = (raw1 + raw2) / 2.0
        center_pitch = 2 ** (center_cents / 1200) * 440

        cycle_params.append({
            'center_time_s': center_time_s,
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
    cycle_times = np.asarray(cycle_times, dtype=float)
    cycle_extents = np.asarray(cycle_extents, dtype=float)

    valid_times = np.isfinite(cycle_times) & (cycle_times > 0)
    valid_extents = np.isfinite(cycle_extents)

    vibrato_rates = 1 / (2 * cycle_times[valid_times])

    avg_rate = np.nanmean(vibrato_rates) if len(vibrato_rates) else np.nan
    stdev_rate = np.nanstd(vibrato_rates) if len(vibrato_rates) else np.nan
    median_rate = np.nanmedian(vibrato_rates) if len(vibrato_rates) else np.nan
    avg_extent = np.nanmean(cycle_extents[valid_extents]) if np.any(valid_extents) else np.nan
    stdev_extent = np.nanstd(cycle_extents[valid_extents]) if np.any(valid_extents) else np.nan
    median_extent = np.nanmedian(cycle_extents[valid_extents]) if np.any(valid_extents) else np.nan
    jitter = (
        np.nanstd(cycle_times[valid_times]) / np.nanmean(cycle_times[valid_times])
        if np.any(valid_times) and np.nanmean(cycle_times[valid_times]) != 0
        else np.nan
    )
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
