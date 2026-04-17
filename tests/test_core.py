import numpy as np
import pandas as pd

from core import (
    detect_vibrato_cycles,
    compute_cycle_parameters,
    filter_vibrato_cycles,
    compute_cv,
    analyze_vibrato,
    smooth_vibrato_parameters,
)


def make_sine_vibrato(duration=2.0, sr=100, rate_hz=6.0, amp_cents=30.0):
    t = np.arange(0, duration, 1 / sr)
    y = amp_cents * np.sin(2 * np.pi * rate_hz * t)
    return t, y


def test_detect_vibrato_cycles_on_clean_sine():
    times, cents = make_sine_vibrato(duration=2.0, sr=100, rate_hz=6.0, amp_cents=30.0)

    peaks, troughs, cycle_times, cycle_extents, t_valid, raw, all_idx = detect_vibrato_cycles(
        times, cents, cents, prominence=5, distance=5
    )

    assert len(peaks) > 5
    assert len(troughs) > 5
    assert len(cycle_times) == len(all_idx) - 1
    assert len(cycle_extents) == len(all_idx) - 1

    # Half-cycle time for 6 Hz vibrato is about 1 / (2*6) = 0.0833 s
    assert np.isclose(np.mean(cycle_times), 1 / (2 * 6.0), atol=0.02)

    # Half-extent should be close to the sine amplitude in cents
    assert np.isclose(np.mean(cycle_extents), 30.0, atol=5.0)


def test_compute_cycle_parameters_returns_expected_columns():
    times, cents = make_sine_vibrato(duration=2.0, sr=100, rate_hz=6.0, amp_cents=30.0)

    peaks, troughs, cycle_times, cycle_extents, t_valid, raw, all_idx = detect_vibrato_cycles(
        times, cents, cents, prominence=5, distance=5
    )

    cycle_params = compute_cycle_parameters(t_valid, raw, cycle_extents, all_idx)

    assert len(cycle_params) > 0
    first = cycle_params[0]

    assert "center_time_s" in first
    assert "cycle_time" in first
    assert "half_extent_cents" in first
    assert "center_pitch" in first

    assert first["cycle_time"] > 0
    assert first["half_extent_cents"] > 0
    assert first["center_pitch"] > 0


def test_filter_vibrato_cycles_removes_out_of_range_extents():
    cycle_params = [
        {"half_extent_cents": 5.0},
        {"half_extent_cents": 20.0},
        {"half_extent_cents": 50.0},
        {"half_extent_cents": 400.0},
    ]

    filtered = filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0)

    assert len(filtered) == 2
    assert filtered[0]["half_extent_cents"] == 20.0
    assert filtered[1]["half_extent_cents"] == 50.0


def test_compute_cv_zero_when_all_values_equal():
    cycle_times = np.array([0.1, 0.1, 0.1, 0.1])
    cycle_extents = np.array([25.0, 25.0, 25.0, 25.0])

    cv_rate, cv_extent = compute_cv(cycle_times, cycle_extents)

    assert np.isclose(cv_rate, 0.0, atol=1e-12)
    assert np.isclose(cv_extent, 0.0, atol=1e-12)


def test_analyze_vibrato_known_values():
    cycle_times = np.array([0.1, 0.1, 0.1, 0.1])
    cycle_extents = np.array([30.0, 30.0, 30.0, 30.0])

    result = analyze_vibrato(cycle_times, cycle_extents)

    assert np.isclose(result["mean_rate"], 5.0, atol=1e-12)
    assert np.isclose(result["median_rate"], 5.0, atol=1e-12)
    assert np.isclose(result["mean_extent"], 30.0, atol=1e-12)
    assert np.isclose(result["median_extent"], 30.0, atol=1e-12)
    assert np.isclose(result["jitter"], 0.0, atol=1e-12)


def test_smooth_vibrato_parameters_returns_multiindex_columns():
    df = pd.DataFrame(
        {
            "cycle_time": [0.10, 0.11, 0.09, 0.10, 0.10],
            "half_extent_cents": [20, 22, 21, 19, 20],
            "label": ["a", "b", "c", "d", "e"],
        }
    )

    smoothed = smooth_vibrato_parameters(df, window_size=3)

    assert ("cycle_time", "mean") in smoothed.columns
    assert ("cycle_time", "std") in smoothed.columns
    assert ("half_extent_cents", "mean") in smoothed.columns
    assert ("half_extent_cents", "std") in smoothed.columns