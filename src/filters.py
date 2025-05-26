# filters.py

import numpy as np
from scipy.signal import butter, filtfilt

########################################
# FILTERING FUNCTIONS
########################################

def filter_pitch_outliers(pitch_values, threshold=5):
    """
    Filters pitch values using the median and median absolute deviation (MAD).
    Values that deviate more than threshold*MAD from the median are replaced by NaN.
    """
    valid = ~np.isnan(pitch_values) & (pitch_values > 0)
    if np.sum(valid) < 5:  # Require at least 5 valid points
        print(f"Warning: Only {np.sum(valid)} valid pitch points before filtering. Returning original array.")
        return pitch_values.copy()

    median_val = np.median(pitch_values[valid])
    mad = np.median(np.abs(pitch_values[valid] - median_val))
    if mad == 0:
        print("Warning: MAD is zero. No filtering applied.")
        return pitch_values.copy()

    lower_bound = median_val - threshold * mad
    upper_bound = median_val + threshold * mad
    filtered = np.where((pitch_values >= lower_bound) & (pitch_values <= upper_bound), pitch_values, np.nan)

    valid_after = np.sum(~np.isnan(filtered))
    print(f"Pitch filtering: {np.sum(valid)} valid points before, {valid_after} after (threshold={threshold})")

    return filtered


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Designs a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a


def apply_bandpass_filter(signal, fs, lowcut=3.0, highcut=9.0, order=4):
    """
    Applies a Butterworth bandpass filter to the signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, signal, method='pad')
    return filtered
