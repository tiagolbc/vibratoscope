# utils.py

import numpy as np
import math
from scipy.interpolate import interp1d

########################################
# UTILITY FUNCTIONS
########################################

def frequency_to_note_name(freq, ref_hz=440.0):
    """
    Converts a frequency in Hz to a musical note name.
    """
    if freq is None or np.isnan(freq) or freq <= 0:
        return "N/A"
    note_number = 12 * math.log2(freq / ref_hz) + 69
    note_number_int = int(round(note_number))
    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    octave = note_number_int // 12 - 1
    note = note_names[note_number_int % 12]
    return f"{note}{octave}"


def sample_entropy(time_series, m=2, r=None, distance='chebyshev'):
    """
    Computes the conventional Sample Entropy (SampEn) of a time series.

    Parameters:
        time_series (array-like): Input time series data.
        m (int, optional): Length of compared sequences (default: 2).
        r (float, optional): Tolerance for distance
            (default: 0.2 * standard deviation of the series).
        distance (str, optional): Distance metric
            ('chebyshev', 'euclidean', 'manhattan'; default: 'chebyshev').

    Returns:
        float: SampEn value, np.nan if data are insufficient or non-finite,
        and np.inf if no valid matches exist.

    Notes:
        - SampEn = -ln(A(m+1) / A(m)).
        - Self-matches are excluded.
        - No additional multiplicative bias factor is applied.
        - Data are not normalized internally; provide normalized input when
          scale invariance is required.
    """
    time_series = np.asarray(time_series, dtype=np.float64).ravel()
    N = len(time_series)

    if m < 1:
        raise ValueError("m must be at least 1")
    if N < m + 2:
        return np.nan
    if not np.all(np.isfinite(time_series)):
        return np.nan

    if r is None:
        series_std = np.std(time_series)
        if series_std == 0:
            return 0.0
        r = 0.2 * series_std

    r = float(r)
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r must be a positive finite number")

    if distance == 'chebyshev':
        dist_func = lambda x, y: np.max(np.abs(x - y), axis=1)
    elif distance == 'euclidean':
        dist_func = lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=1))
    elif distance == 'manhattan':
        dist_func = lambda x, y: np.sum(np.abs(x - y), axis=1)
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

    def _phi(m_len):
        patterns = np.array(
            [time_series[i:i + m_len] for i in range(N - m_len + 1)],
            dtype=np.float64
        )
        count = 0
        for i in range(len(patterns)):
            dists = dist_func(patterns, patterns[i])
            count += np.sum(dists <= r) - 1  # Exclude self-match.

        denom = len(patterns) * (len(patterns) - 1)
        return count / denom if denom > 0 else 0.0

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return np.inf

    return float(-np.log(phi_m1 / phi_m))


def convert_to_cents(pitch_hz, ref_hz=440.0):
    """
    Converts an array of pitch values (Hz) into cents relative to ref_hz.
    """
    pitch_hz = np.array(pitch_hz)
    cents = np.full_like(pitch_hz, np.nan, dtype=float)
    valid = (~np.isnan(pitch_hz)) & (pitch_hz > 0)
    cents[valid] = 1200 * np.log2(pitch_hz[valid] / ref_hz)
    return cents


def remove_mean_or_median(cents_array, use_median=True):
    """
    Centers the pitch contour by removing either the mean or the median.
    """
    valid = ~np.isnan(cents_array)
    if not np.any(valid):
        return cents_array
    center_value = np.median(cents_array[valid]) if use_median else np.mean(cents_array[valid])
    return cents_array - center_value


def resample_to_uniform_time(times, values, new_sr=100):
    """
    Resamples values onto an exact uniform time grid.

    The returned grid has a fixed interval of 1 / new_sr seconds. Linear
    interpolation is used to avoid cubic overshoot that could create artificial
    extrema in a pitch contour.
    """
    times = np.asarray(times, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()

    if len(times) != len(values):
        raise ValueError("times and values must have the same length")
    if not np.isfinite(new_sr) or new_sr <= 0:
        raise ValueError("new_sr must be a positive finite number")

    valid = np.isfinite(times) & np.isfinite(values)
    if np.sum(valid) < 2:
        return None, None

    times_valid = times[valid]
    values_valid = values[valid]

    order = np.argsort(times_valid)
    times_valid = times_valid[order]
    values_valid = values_valid[order]

    # interp1d requires strictly increasing time coordinates. When duplicate
    # times occur, retain the first corresponding value.
    times_valid, unique_indices = np.unique(times_valid, return_index=True)
    values_valid = values_valid[unique_indices]

    if len(times_valid) < 2:
        return None, None

    t_min = float(times_valid[0])
    t_max = float(times_valid[-1])
    step = 1.0 / float(new_sr)

    n_steps = int(np.floor((t_max - t_min) / step + 1e-12))
    if n_steps < 1:
        return None, None

    t_uniform = t_min + np.arange(n_steps + 1, dtype=float) * step

    interpolator = interp1d(
        times_valid,
        values_valid,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True
    )
    v_uniform = interpolator(t_uniform)

    return t_uniform, v_uniform
