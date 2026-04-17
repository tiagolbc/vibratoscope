import numpy as np

from filters import filter_pitch_outliers, apply_bandpass_filter
from utils import (
    convert_to_cents,
    remove_mean_or_median,
    resample_to_uniform_time,
    sample_entropy,
    frequency_to_note_name,
)


def test_filter_pitch_outliers_replaces_extreme_value_with_nan():
    pitch = np.array([100.0, 101.0, 99.0, 100.5, 1000.0])
    filtered = filter_pitch_outliers(pitch, threshold=3)

    assert np.isnan(filtered[-1])
    assert not np.isnan(filtered[0])
    assert not np.isnan(filtered[1])


def test_apply_bandpass_filter_preserves_length():
    sr = 100
    t = np.arange(0, 2, 1 / sr)
    signal = np.sin(2 * np.pi * 6 * t) + 0.2 * np.sin(2 * np.pi * 1 * t)

    filtered = apply_bandpass_filter(signal, fs=sr, lowcut=3.0, highcut=9.0, order=4)

    assert len(filtered) == len(signal)
    assert np.all(np.isfinite(filtered))


def test_convert_to_cents_known_values():
    hz = np.array([440.0, 880.0, 220.0])
    cents = convert_to_cents(hz)

    assert np.isclose(cents[0], 0.0, atol=1e-12)
    assert np.isclose(cents[1], 1200.0, atol=1e-9)
    assert np.isclose(cents[2], -1200.0, atol=1e-9)


def test_remove_mean_or_median_centers_data():
    cents = np.array([10.0, 20.0, 30.0])
    centered = remove_mean_or_median(cents, use_median=True)

    assert np.isclose(np.median(centered), 0.0, atol=1e-12)


def test_resample_to_uniform_time_returns_arrays():
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    values = np.array([0.0, 1.0, 0.0, -1.0, 0.0])

    t_uniform, v_uniform = resample_to_uniform_time(times, values, new_sr=50)

    assert t_uniform is not None
    assert v_uniform is not None
    assert len(t_uniform) == len(v_uniform)
    assert len(t_uniform) > 2


def test_sample_entropy_returns_nan_when_series_too_short():
    x = np.array([1.0, 2.0])
    result = sample_entropy(x, m=2, r=0.2)

    assert np.isnan(result)


def test_sample_entropy_raises_for_invalid_m():
    x = np.array([1.0, 2.0, 3.0, 4.0])

    try:
        sample_entropy(x, m=0, r=0.2)
        assert False, "Expected ValueError for m < 1"
    except ValueError:
        assert True


def test_frequency_to_note_name_for_a4():
    assert frequency_to_note_name(440.0) == "A4"