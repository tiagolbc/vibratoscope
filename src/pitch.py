# pitch.py

import os
import numpy as np
import pandas as pd
import librosa
import parselmouth
from scipy import fft
from sfeeds import extract_pitch_sfeeds as extract_pitch_with_sfeeds

try:
    import pyreaper
    PYREAPER_AVAILABLE = True
except ImportError:
    pyreaper = None
    PYREAPER_AVAILABLE = False


########################################
# LOAD AUDIO
########################################

def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr


########################################
# PITCH EXTRACTION FUNCTIONS
########################################

def _save_pitch_txt(wav_path, times, freqs):
    """Save a Praat pitch contour next to the source WAV."""
    out_txt = os.path.splitext(wav_path)[0] + "_pitch_praat.txt"
    df = pd.DataFrame({
        "time_s": np.asarray(times, dtype=float),
        "f0_Hz": np.asarray(freqs, dtype=float),
    })
    df.to_csv(out_txt, sep="\t", index=False, header=["time_s", "f0_Hz"])
    return out_txt


def extract_pitch_with_praat(
    wav_path,
    fmin=50.0,
    fmax=1500.0,
    time_step=0.005,
    save_txt=True,
):
    """
    Use Praat autocorrelation through Parselmouth.

    Native Praat frame times are returned unchanged. They must not be replaced
    by a linearly stretched 0-duration axis, because that would bias vibrato
    rate.
    """
    snd = parselmouth.Sound(wav_path)
    pitch_obj = snd.to_pitch_ac(
        time_step=float(time_step),
        pitch_floor=float(fmin),
        pitch_ceiling=float(fmax),
    )

    times = np.asarray(pitch_obj.xs(), dtype=float)
    freqs = np.asarray(pitch_obj.selected_array["frequency"], dtype=float)
    freqs[~np.isfinite(freqs) | (freqs <= 0)] = np.nan

    if save_txt:
        _save_pitch_txt(wav_path, times, freqs)

    return times, freqs


def extract_pitch_with_praat_adaptive(
    wav_path,
    fmin=50.0,
    fmax=1500.0,
    time_step=0.005,
    floor_factor=0.75,
    min_valid_points=20,
    return_info=False,
):
    """
    Two-pass Praat extraction for sustained-note vibrato analysis.

    First pass
    ----------
    Uses the supplied Fmin/Fmax values to obtain a preliminary contour.

    Second pass
    -----------
    Uses a pitch floor equal to 0.75 times the preliminary median F0, while
    never going below the user-supplied Fmin. This shortens Praat's
    autocorrelation window and reduces attenuation of fast F0 modulation,
    improving half-extent recovery.

    Safety
    ------
    If the second pass has too few valid frames, the first-pass contour is
    retained. Native Praat times are preserved in both passes.
    """
    fmin = float(fmin)
    fmax = float(fmax)
    time_step = float(time_step)
    floor_factor = float(floor_factor)
    min_valid_points = int(max(3, min_valid_points))

    times_first, freqs_first = extract_pitch_with_praat(
        wav_path,
        fmin=fmin,
        fmax=fmax,
        time_step=time_step,
        save_txt=False,
    )

    valid_first = freqs_first[np.isfinite(freqs_first) & (freqs_first > 0)]
    preliminary_median = (
        float(np.nanmedian(valid_first)) if valid_first.size else np.nan
    )

    info = {
        "adaptive_used": False,
        "preliminary_floor_hz": fmin,
        "final_floor_hz": fmin,
        "preliminary_median_f0_hz": preliminary_median,
    }

    if valid_first.size < min_valid_points or not (0.0 < floor_factor < 1.0):
        _save_pitch_txt(wav_path, times_first, freqs_first)
        return (
            (times_first, freqs_first, info)
            if return_info
            else (times_first, freqs_first)
        )

    adaptive_floor = max(fmin, floor_factor * preliminary_median)
    adaptive_floor = min(adaptive_floor, 0.90 * fmax)

    if not np.isfinite(adaptive_floor) or adaptive_floor <= fmin * 1.01:
        _save_pitch_txt(wav_path, times_first, freqs_first)
        return (
            (times_first, freqs_first, info)
            if return_info
            else (times_first, freqs_first)
        )

    try:
        times_second, freqs_second = extract_pitch_with_praat(
            wav_path,
            fmin=adaptive_floor,
            fmax=fmax,
            time_step=time_step,
            save_txt=False,
        )

        valid_second = int(
            np.sum(np.isfinite(freqs_second) & (freqs_second > 0))
        )
        required_second = max(
            min_valid_points,
            int(np.ceil(0.50 * valid_first.size)),
        )

        if valid_second >= required_second:
            info["adaptive_used"] = True
            info["final_floor_hz"] = float(adaptive_floor)
            _save_pitch_txt(wav_path, times_second, freqs_second)
            return (
                (times_second, freqs_second, info)
                if return_info
                else (times_second, freqs_second)
            )
    except Exception:
        pass

    _save_pitch_txt(wav_path, times_first, freqs_first)
    return (
        (times_first, freqs_first, info)
        if return_info
        else (times_first, freqs_first)
    )


def extract_pitch_with_yin(audio_data, sr, fmin=50.0, fmax=1500.0, file_path=None):
    """
    Extract pitch using YIN (librosa.pyin).
    If file_path is given, save TXT as <file_path>_pitch_yin.txt.
    """
    f0, _, _ = librosa.pyin(audio_data, sr=sr, fmin=fmin, fmax=fmax)
    times = librosa.times_like(f0, sr=sr)

    if file_path is not None:
        out_txt = os.path.splitext(file_path)[0] + "_pitch_yin.txt"
        df = pd.DataFrame({
            "time_s": times,
            "f0_Hz": f0
        })
        df.to_csv(out_txt, sep="\t", index=False, header=["time_s", "f0_Hz"])

    return times, f0


def extract_pitch_with_hps(audio_data, sr, file_path, fmin=50.0, fmax=1500.0, frame_length=0.04, hop_length=0.01):
    """
    Extract pitch using Harmonic Product Spectrum (HPS) with harmonic correction.
    """
    frame_samples = int(frame_length * sr)
    hop_samples = int(hop_length * sr)
    n_frames = int((len(audio_data) - frame_samples) / hop_samples) + 1
    f0 = np.zeros(n_frames)
    times = np.arange(n_frames) * hop_length

    for i in range(n_frames):
        start = i * hop_samples
        frame = audio_data[start:start + frame_samples]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)), mode='constant')

        window = np.hanning(frame_samples)
        frame_windowed = frame * window

        N = frame_samples * 4
        spectrum = np.abs(fft.fft(frame_windowed, N))
        freqs = fft.fftfreq(N, 1 / sr)[:N // 2]
        spectrum = spectrum[:N // 2]

        hps = spectrum.copy()
        num_harmonics = 5
        for h in range(2, num_harmonics + 1):
            decimated = np.interp(freqs, freqs[::h], spectrum[::h])
            hps *= decimated

        valid = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(valid):
            f0[i] = np.nan
            continue

        peak_freq = freqs[valid][np.argmax(hps[valid])]

        for h in range(num_harmonics, 1, -1):
            possible_f0 = peak_freq / h
            if fmin <= possible_f0 <= fmax:
                idx = np.argmin(np.abs(freqs - possible_f0))
                if idx > 0 and spectrum[idx] > 0.1 * spectrum.max():
                    peak_freq = possible_f0
                    break

        f0[i] = peak_freq if fmin <= peak_freq <= fmax else np.nan

    out_txt = os.path.splitext(file_path)[0] + "_pitch_hps.txt"
    df = pd.DataFrame({
        "time_s": times,
        "f0_Hz": f0
    })
    df.to_csv(out_txt, sep="\t", index=False, header=["time_s", "f0_Hz"])

    return times, f0


def extract_pitch_with_reaper(audio_data, sr, fmin=50.0, fmax=1500.0, file_path=None):
    """
    Extract pitch using REAPER (pyreaper wrapper).

    This method is optional. If pyreaper is not installed, a clear error is raised
    only when the REAPER method is explicitly requested.
    """
    if not PYREAPER_AVAILABLE:
        raise ImportError(
            "REAPER support is optional and requires pyreaper. "
            "Install pyreaper to use the REAPER pitch extraction method."
        )

    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            raise ValueError("Invalid audio data.")

        max_abs = np.max(np.abs(audio_data))
        if max_abs == 0:
            raise ValueError("Audio data is silent")

        audio_data_int16 = np.int16(audio_data / max_abs * 32767)

        reaper_output = pyreaper.reaper(audio_data_int16, sr, minf0=fmin, maxf0=fmax)

        if len(reaper_output) >= 4:
            pm_times, pm_epochs, f0_times, f0_values = reaper_output[:4]
        else:
            raise ValueError(f"Unexpected REAPER output format: got {len(reaper_output)} outputs")

        f0 = np.array(f0_values)
        times = np.array(f0_times)
        f0_filtered = np.where((f0 >= fmin) & (f0 <= fmax) & (f0 > 0), f0, np.nan)

        if file_path:
            out_txt = os.path.splitext(file_path)[0] + "_pitch_reaper.txt"
            df = pd.DataFrame({
                "time_s": times,
                "f0_Hz": f0_filtered
            })
            df.to_csv(out_txt, sep="\t", index=False, header=["time_s", "f0_Hz"])

        valid_count = np.sum(~np.isnan(f0_filtered))
        if valid_count < len(f0_filtered) * 0.1:
            raise ValueError(f"Too few valid pitch points extracted by REAPER: {valid_count}/{len(f0_filtered)}")

        return times, f0_filtered

    except Exception as e:
        print(f"REAPER pitch extraction failed: {str(e)}")
        raise


########################################
# HARMONIC PITCH EXTRACTION
########################################

def extract_harmonic_pitch(y, sr, f0_est=None):
    """
    Extracts harmonic-based pitch contour from the STFT.
    """
    n_fft = 2048
    hop_length = 512
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid = freqs >= 50
    if not np.any(valid):
        return None, None, None

    freqs_valid = freqs[valid]
    avg_amp = np.mean(D[valid, :], axis=1)
    idx_max = np.argmax(avg_amp)
    dominant_freq = freqs_valid[idx_max]

    if f0_est is None:
        f0_est = dominant_freq / 2

    harmonic_num = max(1, int(round(dominant_freq / f0_est)))
    t_frames = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    harmonic_contour = []
    expected_freq = harmonic_num * f0_est
    for i in range(D.shape[1]):
        spectrum = D[:, i]
        idx_center = np.argmin(np.abs(freqs - expected_freq))
        start_bin = max(0, idx_center - 3)
        end_bin = min(len(freqs), idx_center + 4)
        local_amp = spectrum[start_bin:end_bin]
        local_idx = np.argmax(local_amp)
        freq_est = freqs[start_bin + local_idx]
        harmonic_contour.append(freq_est)

    harmonic_contour = np.array(harmonic_contour)
    return t_frames, harmonic_contour, harmonic_num
