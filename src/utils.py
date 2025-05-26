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
    Computes the Sample Entropy (SampEn) of a time series.

    Parameters:
        time_series (array-like): Input time series data.
        m (int, optional): Length of compared sequences (default: 2).
        r (float, optional): Tolerance for distance (default: 0.2 * std of series).
        distance (str, optional): Distance metric ('chebyshev', 'euclidean', 'manhattan'; default: 'chebyshev').

    Returns:
        float: SampEn value, np.nan if insufficient data, np.inf if no matches.

    Notes:
        - SampEn = -ln(A(m+1) / A(m)), where A(m) is the proportion of vector pairs of length m within r.
        - Data is not normalized internally; provide normalized input if scale invariance is needed.
        - Bias correction is applied for small sample sizes using an approximate factor (m+1)/(N-m).
        - Requires at least m+1 points; returns np.nan if unmet.
    """
    time_series = np.asarray(time_series, dtype=np.float64)
    N = len(time_series)

    if m < 1:
        raise ValueError("m must be at least 1")
    if N < m + 1:
        return np.nan  # Not enough data

    if r is None:
        r = 0.2 * np.std(time_series)
    if r <= 0:
        raise ValueError("r must be positive")

        # Define distance function
    if distance == 'chebyshev':
        dist_func = lambda x, y: np.max(np.abs(x - y), axis=1)
    elif distance == 'euclidean':
        dist_func = lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=1))
    elif distance == 'manhattan':
        dist_func = lambda x, y: np.sum(np.abs(x - y), axis=1)
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

    def _phi(m_len):
        patterns = np.array([time_series[i:i + m_len] for i in range(N - m_len + 1)])
        count = 0
        for i in range(len(patterns)):
            dists = dist_func(patterns, patterns[i])
            count += np.sum(dists <= r) - 1  # exclude self-match
        denom = (N - m_len + 1) * (N - m_len)
        return count / denom if denom > 0 else 0

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    print(f"[DEBUG SampEn] phi(m={m}): {phi_m:.6f}")
    print(f"[DEBUG SampEn] phi(m+1={m + 1}): {phi_m1:.6f}")

    if phi_m == 0 or phi_m1 == 0:
        return np.inf

    # Apply bias correction (approximate correction from Lake et al., 2002)
    correction = (m + 1) / (N - m)  # Simplified bias factor
    raw_sampen = -np.log(phi_m1 / phi_m)
    corrected_sampen = raw_sampen * correction
    return corrected_sampen


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
    Resamples values to a uniform time grid.
    """
    valid = ~np.isnan(values)
    if np.sum(valid) < 2:
        return None, None
    f = interp1d(times[valid], values[valid], kind='cubic', bounds_error=False, fill_value=np.nan)
    t_min = times[valid].min()
    t_max = times[valid].max()
    num_samples = int(np.ceil((t_max - t_min) * new_sr))
    t_uniform = np.linspace(t_min, t_max, num_samples)
    v_uniform = f(t_uniform)
    return t_uniform, v_uniform
