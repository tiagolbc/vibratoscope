import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Define o backend antes de importar pyplot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import sounddevice as sd
import shutil


########################################
# NEW FUNCTION: Convert frequency to musical note name
########################################
def frequency_to_note_name(freq, ref_hz=440.0):
    import math
    if freq <= 0:
        return "N/A"
    # Compute note number based on A4 = 440 Hz (MIDI 69)
    note_number = 12 * math.log2(freq / ref_hz) + 69
    note_number_int = int(round(note_number))
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = note_number_int // 12 - 1
    note = note_names[note_number_int % 12]
    return f"{note}{octave}"


########################################
# FUNCTION: Filter pitch outliers using median and MAD
########################################
def filter_pitch_outliers(pitch_values, threshold=3):
    """
    Filters pitch values using the median and median absolute deviation (MAD).
    Values that deviate more than threshold*MAD from the median are replaced by NaN.
    """
    valid = ~np.isnan(pitch_values) & (pitch_values > 0)
    if np.sum(valid) == 0:
        return pitch_values
    median_val = np.median(pitch_values[valid])
    mad = np.median(np.abs(pitch_values[valid] - median_val))
    if mad == 0:
        return pitch_values
    lower_bound = median_val - threshold * mad
    upper_bound = median_val + threshold * mad
    filtered = np.where((pitch_values >= lower_bound) & (pitch_values <= upper_bound), pitch_values, np.nan)
    return filtered


########################################
# FUNCTION: Sample Entropy
########################################
def sample_entropy(time_series, m=2, r=None):
    """
    Computes the sample entropy of a time series.

    Parameters:
      time_series: array-like, the time series (e.g., vibrato rate or extent)
      m: embedding dimension (usually 2)
      r: tolerance; if None, will be set to 0.2 times the standard deviation

    Returns:
      sampen: the value of the sample entropy
    """
    time_series = np.array(time_series)
    N = len(time_series)
    if r is None:
        r = 0.2 * np.std(time_series)

    def _phi(m):
        x = np.array([time_series[i: i + m] for i in range(N - m + 1)])
        C = []
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            count = np.sum(dist <= r) - 1  # discount self-match
            C.append(count)
        return np.sum(C) / ((N - m + 1) * (N - m))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return np.inf
    return -np.log(phi_m1 / phi_m)


########################################
# SPLASH SCREEN
########################################
def show_splash(root):
    splash = tk.Toplevel(root)
    splash.title("About Vibrato Scope")
    splash.geometry("800x640")
    splash.resizable(False, False)

    splash_text = (
        "Vibrato Scope\n\n"
        "Developed by: Dr. Tiago Lima Bicalho Cruz\n\n"
        "Ph.D. in Music (2020–2025)\n"
        "Federal University of Minas Gerais (UFMG)\n\n"
        "Additional Qualifications:\n"
        "M.A. in Music, Postgraduate Specialization in Voice Clinics, B.Sc. in Speech-Language Therapy and Audiology\n\n"
        "Teaching & Research:\n"
        "Vocal Coach, Lecturer, and Researcher in voice and acoustic analysis\n\n"
        "Disclaimer: This software is for personal use only. It is provided as-is without any warranty.\n\n"
        "License: Although this software is open source under the MIT license, the author kindly requests that users "
        "do not modify or redistribute altered versions. If you have suggestions please send an e-mail to me. Please use the tool as-is and cite it appropriately. Thank you!\n\n"
        "The analysis results are approximate and may not be 100% accurate.\n\n"
        "© 2025 Dr. Tiago Lima Bicalho Cruz\n\n"
        "tiagolbc@gmail.com\n\n"
        "Press 'Continue' to proceed."
    )

    label = tk.Label(splash, text=splash_text, justify="center", font=("Arial", 10), padx=20, pady=20, wraplength=760)
    label.pack(expand=True, fill="both")

    btn = tk.Button(splash, text="Continue", command=splash.destroy, font=("Arial", 12))
    btn.pack(pady=10)

    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (800 // 2)
    y = (root.winfo_screenheight() // 2) - (640 // 2)
    splash.geometry(f"+{x}+{y}")

    splash.grab_set()
    root.wait_window(splash)


########################################
# HELPER FUNCTIONS FOR PROCESSING
########################################
def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr


def extract_pitch_yin(y, sr, fmin=50, fmax=1500):
    f0, _, _ = librosa.pyin(y, sr=sr, fmin=fmin, fmax=fmax)
    times = librosa.times_like(f0, sr=sr)
    return times, f0


def convert_to_cents(pitch_hz, ref_hz=440.0):
    pitch_hz = np.array(pitch_hz)
    cents = np.full_like(pitch_hz, np.nan, dtype=float)
    valid = (~np.isnan(pitch_hz)) & (pitch_hz > 0)
    cents[valid] = 1200 * np.log2(pitch_hz[valid] / ref_hz)
    return cents


def remove_mean_or_median(cents_array, use_median=True):
    valid = ~np.isnan(cents_array)
    if not np.any(valid):
        return cents_array
    center_value = np.median(cents_array[valid]) if use_median else np.mean(cents_array[valid])
    return cents_array - center_value


def resample_to_uniform_time(times, values, new_sr=100):
    from scipy.interpolate import interp1d
    valid = ~np.isnan(values)
    if np.sum(valid) < 2:
        return None, None
    f = interp1d(times[valid], values[valid], kind='linear', bounds_error=False, fill_value=np.nan)
    t_min = times[valid].min()
    t_max = times[valid].max()
    num_samples = int(np.ceil((t_max - t_min) * new_sr))
    t_uniform = np.linspace(t_min, t_max, num_samples)
    v_uniform = f(t_uniform)
    return t_uniform, v_uniform


def butter_bandpass(lowcut, highcut, fs, order=4):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a


def apply_bandpass_filter(signal, fs, lowcut=3.0, highcut=9.0, order=4):
    from scipy.signal import filtfilt
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, signal, method='pad')
    return filtered


########################################
# VIBRATO DETECTION AND ANALYSIS
########################################
def detect_vibrato_cycles(times, cents_raw, cents_filtered, prominence=5, distance=5):
    from scipy.signal import find_peaks
    valid = ~np.isnan(cents_filtered)
    t_valid = times[valid]
    filt = cents_filtered[valid]
    raw = cents_raw[valid]
    peaks, _ = find_peaks(filt, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-filt, prominence=prominence, distance=distance)
    all_idx = np.sort(np.concatenate((peaks, troughs)))

    cycle_times = []
    cycle_extents = []
    for i in range(len(all_idx) - 1):
        i1, i2 = all_idx[i], all_idx[i + 1]
        dt = t_valid[i2] - t_valid[i1]
        amp_full = abs(filt[i2] - filt[i1])  # full extent em cents
        cycle_times.append(dt)
        cycle_extents.append(amp_full)

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
            'half_extent_cents': extents[i],  # aqui já é o full extent
            'center_pitch': center_pitch
        })
    return cycle_params


def filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0):
    return [cp for cp in cycle_params if
            (cp['half_extent_cents'] >= min_half_extent and cp['half_extent_cents'] <= max_half_extent)]


def compute_jitter_metrics(periods):
    import numpy as np
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
    import numpy as np
    vibrato_rates = 1 / (2 * cycle_times)
    cv_rate = (np.nanstd(vibrato_rates) / np.nanmean(vibrato_rates)) * 100
    cv_extent = (np.nanstd(cycle_extents) / np.nanmean(cycle_extents)) * 100
    return cv_rate, cv_extent


def analyze_vibrato(cycle_times, cycle_extents):
    import numpy as np
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
    import numpy as np
    df_numeric = df.select_dtypes(include=[np.number])
    return df_numeric.rolling(window=window_size, center=True).agg(['mean', 'std'])


########################################
# EXTRACT HARMONIC PITCH (from entire audio)
########################################
def extract_harmonic_pitch(y, sr, f0_est=None):
    """
    Extracts, from the STFT of the entire audio, a pitch contour based on the harmonic
    with the highest intensity (above 50 Hz). If f0_est is provided, it is used to estimate the harmonic number.

    Returns:
      t_frames: time vector (in seconds) for each STFT frame
      harmonic_contour: the extracted frequency contour (in Hz) for the harmonic
      harmonic_num: the harmonic number used (1 for fundamental, 2 for second, etc.)
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


########################################
# FIGURE 1: BEFORE AND AFTER FILTER
########################################
def plot_before_after_filter(t_uniform, cents_uniform, cents_filtered, save_path="Figure_1.png"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t_uniform, cents_uniform, label="Centered Pitch (cents)")
    axs[0].set_title("Centered Pitch (before filter)")
    axs[0].set_ylabel("Cents")
    axs[0].legend()
    axs[1].plot(t_uniform, cents_filtered, color='r', label="Filtered Signal (3-9 Hz)")
    axs[1].set_title("Vibrato Component (filtered)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Cents")
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


########################################
# FIGURE 2: PEAKS AND TROUGHS
########################################
def plot_peaks_troughs(t_valid, cents_valid, peaks, troughs, save_path="Figure_2.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_valid, cents_valid, label="Filtered Signal")
    ax.plot(t_valid[peaks], cents_valid[peaks], 'ro', label="Peaks")
    ax.plot(t_valid[troughs], cents_valid[troughs], 'go', label="Troughs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cents (centered)")
    ax.set_title("Peak and Trough Detection in Filtered Signal")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


########################################
# FIGURE 3: VIBRATO RATE PER HALF CYCLE
########################################
def plot_vibrato_rate(df, df_smoothed, save_path="Figure_3.png"):
    df['VibratoRate'] = 1 / (2 * df['cycle_time'])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['VibratoRate'], 'k.-', label="Rate")
    if ('cycle_time', 'mean') in df_smoothed.columns:
        smoothed_rate = 1 / (2 * df_smoothed[('cycle_time', 'mean')])
        ax.plot(smoothed_rate.index, smoothed_rate, 'b.-', label="Smoothed Rate")
    ax.set_title("Vibrato Rate per Half Cycle")
    ax.set_xlabel("Cycle #")
    ax.set_ylabel("Rate (Hz)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


########################################
# FINAL SUMMARY PLOT FUNCTION (3 on top, 3 on bottom)
########################################
def final_plot(df, df_smoothed, summary_data, jitter_metrics, cv_rate, cv_extent,
               cv_rate_smooth, cv_extent_smooth, filename,
               sampen_rate, sampen_extent, title="Vibrato Scope: Final Analysis", show_figure=True):
    from matplotlib.gridspec import GridSpec
    import os
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.8], figure=fig)
    fig.subplots_adjust(hspace=0.6, wspace=0.3, left=0.05, right=0.95, top=0.88, bottom=0.05)
    fig.suptitle(f"{title}\nFile: {os.path.basename(filename)}", fontsize=14, fontweight='bold')

    # Row 0
    ax1 = fig.add_subplot(gs[0, 0])
    df['VibratoRate'] = 1 / (2 * df['cycle_time'])
    ax1.plot(df.index, df['VibratoRate'], 'k.-')
    ax1.set_title("Raw Rate (Hz)", fontsize=10)
    ax1.set_xlabel("Cycle #", fontsize=9)
    ax1.set_ylabel("Rate (Hz)", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df.index, df['half_extent_cents'], 'm.-')
    ax2.set_title("Raw Extent (cents)", fontsize=10)
    ax2.set_xlabel("Cycle #", fontsize=9)
    ax2.set_ylabel("Extent (cents)", fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    if ('cycle_time', 'mean') in df_smoothed.columns:
        rate_smooth = 1 / (2 * df_smoothed[('cycle_time', 'mean')])
        ax3.plot(rate_smooth.index, rate_smooth, 'b.-')
    ax3.set_title("Smoothed Rate (Hz)", fontsize=10)
    ax3.set_xlabel("Cycle #", fontsize=9)
    ax3.set_ylabel("Rate (Hz)", fontsize=9)

    # Row 1
    ax4 = fig.add_subplot(gs[1, 0])
    if ('half_extent_cents', 'mean') in df_smoothed.columns:
        extent_smooth = df_smoothed[('half_extent_cents', 'mean')]
        ax4.plot(extent_smooth.index, extent_smooth, 'c.-')
    ax4.set_title("Smoothed Extent (cents)", fontsize=10)
    ax4.set_xlabel("Cycle #", fontsize=9)
    ax4.set_ylabel("Extent (cents)", fontsize=9)

    ax5 = fig.add_subplot(gs[1, 1])
    rate_raw = df['VibratoRate']
    extent_raw = df['half_extent_cents']
    rate_norm = (rate_raw - rate_raw.mean()) / rate_raw.std() if rate_raw.std() != 0 else rate_raw * 0
    extent_norm = (extent_raw - extent_raw.mean()) / extent_raw.std() if extent_raw.std() != 0 else extent_raw * 0
    ax5.plot(df.index, rate_norm, 'k-', label="Rate (norm)")
    ax5.plot(df.index, extent_norm, 'm-', label="Extent (norm)")
    ax5.set_title("CoV Time Series", fontsize=10)
    ax5.set_xlabel("Cycle #", fontsize=9)
    ax5.set_ylabel("Normalized", fontsize=9)
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(['Rate', 'Extent'], [sampen_rate, sampen_extent], color=['blue', 'purple'])
    ax6.set_title("Sample Entropy", fontsize=10)
    ax6.set_ylabel("SampEn Value", fontsize=9)

    # Row 2: Summary table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')
    col_titles = [
        "Smoothed Data",
        "Unsmoothed Data (Hz)",
        "Vibrato Jitter (Unsmoothed)",
        "Coefficient of Variability",
        "Other Data"
    ]
    vertical_title_offset = 0.9
    vertical_text_offset = 0.65

    mean_pitch = summary_data.get('mean_pitch_hz', 0)
    min_pitch = summary_data.get('min_pitch_hz', None)
    max_pitch = summary_data.get('max_pitch_hz', None)
    if min_pitch is not None and max_pitch is not None and abs(max_pitch - min_pitch) > 1e-6:
        note_min = frequency_to_note_name(min_pitch)
        note_max = frequency_to_note_name(max_pitch)
        additional_data = f"Pitch Range (Hz): {min_pitch:.2f} - {max_pitch:.2f}\nRange (Notes): {note_min} - {note_max}"
    else:
        note_name = frequency_to_note_name(mean_pitch)
        additional_data = f"Note: {note_name}"
    other_data_str = (
            f"Mean Pitch (Hz): {mean_pitch:.2f}\n"
            + additional_data + "\n"
            + f"Global Jitter: {summary_data.get('Global_Jitter', 0):.2f}\n"
            + f"SampEn Rate: {sampen_rate:.2f}\n"
            + f"SampEn Extent: {sampen_extent:.2f}\n"
            + f"Harmonic Used: {summary_data.get('HarmonicUsed', 'N/A')}\n"
            + f"Harmonic Frequency: {summary_data.get('HarmonicFrequency', 'N/A')}"
    )
    smoothed_data_str = (
        f"Mean Rate: {summary_data.get('mean_rate_smooth', np.nan):.2f}\n"
        f"StDev Rate: {summary_data.get('stdev_rate_smooth', np.nan):.2f}\n"
        f"Median Rate: {summary_data.get('median_rate_smooth', np.nan):.2f}\n"
        f"Mean Extent: {summary_data.get('mean_extent_smooth', np.nan):.2f}\n"
        f"StDev Extent: {summary_data.get('stdev_extent_smooth', np.nan):.2f}\n"
        f"Median Extent: {summary_data.get('median_extent_smooth', np.nan):.2f}"
    )
    unsmoothed_data_str = (
        f"Mean Rate: {summary_data.get('mean_rate_unsmoothed', np.nan):.2f}\n"
        f"StDev Rate: {summary_data.get('stdev_rate_unsmoothed', np.nan):.2f}\n"
        f"Median Rate: {summary_data.get('median_rate_unsmoothed', np.nan):.2f}\n"
        f"Mean Extent: {summary_data.get('mean_extent_unsmoothed', np.nan):.2f}\n"
        f"StDev Extent: {summary_data.get('stdev_extent_unsmoothed', np.nan):.2f}\n"
        f"Median Extent: {summary_data.get('median_extent_unsmoothed', np.nan):.2f}"
    )
    jitter_str = (
        f"Local: {jitter_metrics.get('jitter_local_percent', np.nan):.2f}%\n"
        f"Local Abs: {jitter_metrics.get('jitter_local_abs_ms', np.nan):.2f}ms\n"
        f"Rap: {jitter_metrics.get('jitter_rap_percent', np.nan):.2f}%\n"
        f"Ppq5: {jitter_metrics.get('jitter_ppq5_percent', np.nan):.2f}%\n"
        f"Ddp: {jitter_metrics.get('jitter_ddp_percent', np.nan):.2f}%"
    )
    cov_str = (
        f"Rate: {cv_rate:.2f}%\n"
        f"Extent: {cv_extent:.2f}%\n"
        f"Sm. Rate: {cv_rate_smooth:.2f}%\n"
        f"Sm. Extent: {cv_extent_smooth:.2f}%"
    )
    col_texts = [
        smoothed_data_str,
        unsmoothed_data_str,
        jitter_str,
        cov_str,
        other_data_str
    ]
    col_x = [0.02, 0.22, 0.42, 0.62, 0.82]
    for i in range(5):
        ax_table.text(col_x[i], vertical_title_offset, col_titles[i],
                      fontsize=9, fontweight='bold', ha='left', va='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
        ax_table.text(col_x[i], vertical_text_offset, col_texts[i],
                      fontsize=8, ha='left', va='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))

    fig.canvas.draw()
    fig.savefig("Figure_summary.png", dpi=300)
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    return fig


########################################
# TKINTER GUI CLASS
########################################
class VibratoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vibrato Scope - version 1.0")
        self.root.geometry("1200x800")
        # Initialize variables
        self.audio_data = None
        self.sr = None
        self.times = None
        self.pitch_hz = None
        self.t_uniform = None
        self.cents_uniform = None
        self.selected_regions = []
        self.file_path = ""
        self.df_detailed = None
        self.df_avg = None
        self.df_region = None
        self.summary_data = None
        self.last_fig = None
        self.generated_figures = []
        self.alternative_used = False
        self.harmonic_used = None
        self.harmonic_frequency = None

        # Top frame for buttons and controls
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.btn_select = tk.Button(top, text="Select WAV File", command=self.load_file)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        self.btn_play = tk.Button(top, text="Play Audio", command=self.play_audio, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        fminf = tk.Frame(top)
        tk.Label(fminf, text="Fmin (Hz):").pack(side=tk.LEFT)
        self.fmin = tk.StringVar(value="50.114")
        tk.Entry(fminf, textvariable=self.fmin, width=5).pack(side=tk.LEFT)
        fminf.pack(side=tk.LEFT, padx=5)
        fmaxf = tk.Frame(top)
        tk.Label(fmaxf, text="Fmax (Hz):").pack(side=tk.LEFT)
        self.fmax = tk.StringVar(value="1500")
        tk.Entry(fmaxf, textvariable=self.fmax, width=6).pack(side=tk.LEFT)
        fmaxf.pack(side=tk.LEFT, padx=5)
        self.btn_batch = tk.Button(top, text="Batch Process", command=self.batch_process)
        self.btn_batch.pack(side=tk.LEFT, padx=5)

        # Frame for spectrogram display
        specf = tk.Frame(root, relief="sunken", borderwidth=1)
        specf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=specf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb = tk.Frame(specf)
        tb.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, tb)
        self.ax.set_title("Load Audio File - Spectrogram will appear here")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")

        # Frame for region selection listbox
        lf = tk.Frame(root)
        lf.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(lf, text="Selected Regions:").pack(side=tk.LEFT)
        self.lb = tk.Listbox(lf, width=60, height=4)
        self.lb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        bf = tk.Frame(lf)
        self.btn_clear = tk.Button(bf, text="Clear Regions", command=self.clear_regions)
        self.btn_clear.pack(padx=5, pady=2)
        self.btn_analyze = tk.Button(bf, text="Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(padx=5, pady=2)
        bf.pack(side=tk.LEFT, padx=5)

        # Bottom frame for save and exit buttons
        bot = tk.Frame(root)
        bot.pack(fill=tk.X, padx=5, pady=5)
        self.btn_save = tk.Button(bot, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        self.btn_exit = tk.Button(bot, text="Exit", command=self.on_closing)
        self.btn_exit.pack(side=tk.RIGHT, padx=5)

        self.span = None
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def reset_analysis_state(self):
        # Reset all analysis-related variables and UI elements
        self.audio_data = None
        self.sr = None
        self.times = None
        self.pitch_hz = None
        self.cents_centered = None
        self.t_uniform = None
        self.cents_uniform = None
        self.selected_regions = []
        self.lb.delete(0, tk.END)
        self.file_path = ""
        self.df_detailed = None
        self.df_avg = None
        self.df_region = None
        self.summary_data = {}
        self.last_fig = None
        self.generated_figures = []
        self.alternative_used = False
        self.harmonic_used = None
        self.harmonic_frequency = None
        self.ax.clear()
        self.ax.set_title("Load Audio File - Spectrogram will appear here")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.canvas.draw()
        self.btn_play.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        if self.span:
            self.span.disconnect_events()
            self.span = None
        self.root.title("Vibrato Scope - version 1.0")

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def load_file(self):
        # Load a WAV file and display its spectrogram and pitch contour
        fp = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if not fp:
            return
        self.file_path = fp
        self.root.title(f"Vibrato Scope - {os.path.basename(fp)}")
        try:
            fmin, fmax = float(self.fmin.get()), float(self.fmax.get())
            if fmin >= fmax or fmin < 0:
                raise ValueError("Invalid Fmin/Fmax")
        except:
            messagebox.showerror("Error", "Invalid Fmin/Fmax.")
            return
        self.reset_analysis_state()
        try:
            self.audio_data, self.sr = load_audio(fp)
            self.times, self.pitch_hz = extract_pitch_yin(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
            self.pitch_hz = filter_pitch_outliers(self.pitch_hz, threshold=3)
            if np.all(np.isnan(self.pitch_hz)):
                messagebox.showerror("Error", "No valid pitch after filtering.")
                return
            cents = convert_to_cents(self.pitch_hz)
            self.cents_centered = remove_mean_or_median(cents, True)
            self.t_uniform, self.cents_uniform = resample_to_uniform_time(self.times, self.cents_centered, 100)
            if self.t_uniform is None or len(self.t_uniform) < 2:
                messagebox.showerror("Error", "Resampling failed.")
                return
            self.ax.clear()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
            librosa.display.specshow(D, sr=self.sr, hop_length=512, x_axis='time', y_axis='log', ax=self.ax)
            vm = ~np.isnan(self.pitch_hz)
            if np.any(vm):
                self.ax.plot(self.times[vm], self.pitch_hz[vm], color='cyan', linewidth=1.5, label='F0')
                self.ax.legend(loc='upper right')
            self.ax.set_title(f"Spectrogram & Detected F0 - {os.path.basename(fp)}")
            self.canvas.draw()
            self.btn_play.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            if self.span:
                self.span.disconnect_events()
            self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True,
                                     props=dict(alpha=0.3, facecolor='red'), interactive=True)
        except Exception as e:
            messagebox.showerror("Error Loading File", str(e))
            self.reset_analysis_state()

    def on_select(self, tmin, tmax):
        # Handle region selection using SpanSelector
        if self.times is not None:
            tmin = max(tmin, self.times[0])
            tmax = min(tmax, self.times[-1])
        if tmax > tmin:
            self.selected_regions.append((tmin, tmax))
            self.lb.insert(tk.END, f"Region {len(self.selected_regions)}: {tmin:.3f}-{tmax:.3f}s")
            self.lb.see(tk.END)

    def clear_regions(self):
        # Clear all selected regions
        self.selected_regions.clear()
        self.lb.delete(0, tk.END)

    def play_audio(self):
        if self.audio_data is not None:
            sd.play(self.audio_data, self.sr)

    def run_analysis(self):
        # Reset alternative analysis flags each time we run analysis.
        self.alternative_used = False
        self.harmonic_used = None
        self.harmonic_frequency = None

        if not self.selected_regions:
            messagebox.showerror("Error", "No regions selected!")
            return

        all_cycle_params = []
        region_list = []
        self.generated_figures = []

        # Figure 1: Global analysis using the fundamental-based (default) resampled data.
        global_filtered = apply_bandpass_filter(self.cents_uniform, fs=100, lowcut=3.0, highcut=9.0, order=4)
        fig1 = plot_before_after_filter(self.t_uniform, self.cents_uniform, global_filtered, save_path="Figure_1.png")
        self.generated_figures.append(fig1)

        for reg_idx, (t_start, t_end) in enumerate(self.selected_regions):
            mask = (self.t_uniform >= t_start) & (self.t_uniform <= t_end)
            t_region = self.t_uniform[mask]
            if len(t_region) == 0:
                continue
            cents_region = self.cents_uniform[mask]
            cents_filtered = apply_bandpass_filter(cents_region, fs=100, lowcut=3.0, highcut=9.0, order=4)
            peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                t_region, cents_region, cents_filtered, prominence=5, distance=5
            )
            if len(t_valid) > 0:
                fig2 = plot_peaks_troughs(t_valid, cents_region, peaks, troughs,
                                          save_path=f"Figure_2_region_{reg_idx + 1}.png")
                self.generated_figures.append(fig2)
            from numpy import isnan
            reg_mask = (self.times >= t_start) & (self.times <= t_end)
            region_pitch = self.pitch_hz[reg_mask]
            region_avg_F0 = np.nanmean(region_pitch) if np.any(~isnan(region_pitch)) else np.nan
            if len(all_idx) < 2:
                continue
            cycle_params = compute_cycle_parameters(t_valid, self.cents_uniform, half_cycle_extents, all_idx)
            filtered_params = filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0)
            if not filtered_params:
                continue
            df_file = pd.DataFrame(filtered_params)
            df_file['File_Name'] = os.path.basename(self.file_path)
            num_cycles = len(df_file)
            if num_cycles < 5:
                continue
            region_rate = 1 / (2 * df_file['cycle_time'])
            region_avg_rate = region_rate.mean()
            region_std_rate = region_rate.std()
            region_median_rate = np.median(region_rate)
            region_avg_extent = df_file['half_extent_cents'].mean()
            region_std_extent = df_file['half_extent_cents'].std()
            region_median_extent = np.median(df_file['half_extent_cents'])
            region_jitter = df_file['cycle_time'].std() / df_file['cycle_time'].mean()
            region_cv_rate = (region_rate.std() / region_rate.mean()) * 100
            region_cv_extent = (df_file['half_extent_cents'].std() / df_file['half_extent_cents'].mean()) * 100
            for cp in filtered_params:
                cp['Region_Start'] = t_start
                cp['Region_End'] = t_end
                cp['Region_Num_Cycles'] = num_cycles
                cp['Region_Avg_Rate_Hz'] = region_avg_rate
                cp['Region_StDev_Rate_Hz'] = region_std_rate
                cp['Region_Median_Rate_Hz'] = region_median_rate
                cp['Region_Avg_Extent_cents'] = region_avg_extent
                cp['Region_StDev_Extent_cents'] = region_std_extent
                cp['Region_Median_Extent_cents'] = region_median_extent
                cp['Region_Jitter'] = region_jitter
                cp['Region_CV_Rate_%'] = region_cv_rate
                cp['Region_CV_Extent_%'] = region_cv_extent
                cp['Region_Avg_F0_Hz'] = region_avg_F0
            all_cycle_params.extend(filtered_params)
            reg_dict = {
                'Region_Start': t_start,
                'Region_End': t_end,
                'Num_Cycles': num_cycles,
                'Avg_Rate_Hz': region_avg_rate,
                'StDev_Rate_Hz': region_std_rate,
                'Median_Rate_Hz': region_median_rate,
                'Avg_Extent_cents': region_avg_extent,
                'StDev_Extent_cents': region_std_extent,
                'Median_Extent_cents': region_median_extent,
                'Jitter': region_jitter,
                'CV_Rate_%': region_cv_rate,
                'CV_Extent_%': region_cv_extent,
                'Avg_F0_Hz': region_avg_F0,
                'File_Name': os.path.basename(self.file_path)
            }
            region_list.append(reg_dict)
            if not hasattr(self, "batch_detailed_list"):
                self.batch_detailed_list = []
            self.batch_detailed_list.append(df_file)

        # If no valid cycles were found using the fundamental-based analysis,
        # try alternative harmonic-based analysis.
        if not all_cycle_params:
            response = messagebox.askyesno(
                "Alternative Analysis",
                "No valid vibrato cycles were found using the fundamental-based analysis.\n"
                "Would you like to try analysis using the most intense harmonic?"
            )
            if response:
                from numpy import isnan
                valid_f0 = self.pitch_hz[~isnan(self.pitch_hz) & (self.pitch_hz > 0)]
                if len(valid_f0) == 0:
                    messagebox.showerror("Error", "No valid fundamental (f0) to estimate harmonic.")
                    return
                f0_est = np.median(valid_f0)
                t_h, harmonic_pitch, harmonic_num = extract_harmonic_pitch(self.audio_data, self.sr, f0_est)
                if t_h is None or harmonic_pitch is None:
                    messagebox.showerror("Error", "Failed to extract the harmonic pitch contour.")
                    return
                self.alternative_used = True
                self.harmonic_used = harmonic_num
                self.harmonic_frequency = np.nanmedian(harmonic_pitch)
                cents_h = convert_to_cents(harmonic_pitch)
                cents_h_centered = remove_mean_or_median(cents_h, use_median=True)
                cents_adjusted = cents_h_centered / harmonic_num

                new_t_uniform, new_cents_uniform = resample_to_uniform_time(t_h, cents_adjusted, new_sr=100)
                if new_t_uniform is None:
                    messagebox.showerror("Error", "Not enough valid points for resampling in alternative method.")
                    return

                alt_cycle_params = []
                alt_region_list = []
                # For each selected region, slice the resampled harmonic contour
                for reg_idx, (t_start_alt, t_end_alt) in enumerate(self.selected_regions):
                    region_mask = (new_t_uniform >= t_start_alt) & (new_t_uniform <= t_end_alt)
                    t_region_alt = new_t_uniform[region_mask]
                    c_region_alt = new_cents_uniform[region_mask]
                    if len(t_region_alt) < 2 or np.all(np.isnan(c_region_alt)):
                        continue
                    c_filtered_alt = apply_bandpass_filter(c_region_alt, fs=100, lowcut=3.0, highcut=9.0, order=4)
                    peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                        t_region_alt,
                        c_region_alt,  # raw cents
                        c_filtered_alt,
                        prominence=5,
                        distance=5
                    )
                    if len(t_valid) < 2 or len(all_idx) < 2:
                        continue
                    cycle_params = compute_cycle_parameters(t_valid, self.cents_uniform, half_cycle_extents, all_idx)
                    filtered_params = filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0)
                    if not filtered_params:
                        continue
                    df_file = pd.DataFrame(filtered_params)
                    df_file['File_Name'] = os.path.basename(self.file_path)
                    alt_cycle_params.extend(filtered_params)
                    num_cycles = len(df_file)
                    region_rate = 1 / (2 * df_file['cycle_time'])
                    region_avg_rate = region_rate.mean()
                    region_std_rate = region_rate.std()
                    region_median_rate = np.median(region_rate)
                    region_avg_extent = df_file['half_extent_cents'].mean()
                    region_std_extent = df_file['half_extent_cents'].std()
                    region_median_extent = np.median(df_file['half_extent_cents'])
                    region_jitter = df_file['cycle_time'].std() / df_file['cycle_time'].mean()
                    region_cv_rate = (region_rate.std() / region_rate.mean()) * 100
                    region_cv_extent = (df_file['half_extent_cents'].std() / df_file['half_extent_cents'].mean()) * 100
                    alt_region_item = {
                        'Region_Start': t_start_alt,
                        'Region_End': t_end_alt,
                        'Num_Cycles': num_cycles,
                        'Avg_Rate_Hz': region_avg_rate,
                        'StDev_Rate_Hz': region_std_rate,
                        'Median_Rate_Hz': region_median_rate,
                        'Avg_Extent_cents': region_avg_extent,
                        'StDev_Extent_cents': region_std_extent,
                        'Median_Extent_cents': region_median_extent,
                        'Jitter': region_jitter,
                        'CV_Rate_%': region_cv_rate,
                        'CV_Extent_%': region_cv_extent,
                        'Analysis': f"Harmonic {harmonic_num}",
                        'File_Name': os.path.basename(self.file_path)
                    }
                    alt_region_list.append(alt_region_item)
                    fig_alt = plot_before_after_filter(t_region_alt, c_region_alt, c_filtered_alt,
                                                       save_path=f"Figure_1_alt_region_{reg_idx + 1}.png")
                    self.generated_figures.append(fig_alt)
                    alt_peaks, alt_troughs = detect_vibrato_cycles(t_region_alt, c_region_alt, c_filtered_alt,
                                                                   prominence=5, distance=5)[:2]
                    fig2_alt = plot_peaks_troughs(t_region_alt, c_filtered_alt, alt_peaks, alt_troughs,
                                                  save_path=f"Figure_2_alt_region_{reg_idx + 1}.png")
                    self.generated_figures.append(fig2_alt)

                if not alt_cycle_params:
                    messagebox.showinfo("Analysis", "No valid vibrato cycles were found using the harmonic method.")
                    return
                all_cycle_params = alt_cycle_params
                region_list = alt_region_list
                # For plotting, override t_uniform and cents_uniform to reflect the first selected region
                mask_alt = (new_t_uniform >= self.selected_regions[0][0]) & (
                            new_t_uniform <= self.selected_regions[0][1])
                self.t_uniform = new_t_uniform[mask_alt]
                self.cents_uniform = new_cents_uniform[mask_alt]
            else:
                messagebox.showinfo("Analysis", "No valid vibrato cycles were found.")
                return

        df_detailed = create_vibrato_dataframe(all_cycle_params)
        df_detailed['File_Name'] = os.path.basename(self.file_path)
        df_detailed['VibratoRate'] = 1 / (2 * df_detailed['cycle_time'])
        cycle_times_all = df_detailed['cycle_time'].values
        cycle_extents_all = df_detailed['half_extent_cents'].values
        global_stats = analyze_vibrato(cycle_times_all, cycle_extents_all)
        cv_rate, cv_extent = compute_cv(cycle_times_all, cycle_extents_all)
        window_size = simpledialog.askinteger("Smoothing", "Enter number of cycles for smoothing (e.g., 3):",
                                              initialvalue=3)
        if window_size is None:
            window_size = 3
        df_smoothed = smooth_vibrato_parameters(df_detailed, window_size=window_size)
        smoothed_cycle_times = df_smoothed[('cycle_time', 'mean')].dropna().values
        smoothed_cycle_extents = df_smoothed[('half_extent_cents', 'mean')].dropna().values
        global_stats_smooth = analyze_vibrato(smoothed_cycle_times, smoothed_cycle_extents)
        cv_rate_smooth, cv_extent_smooth = compute_cv(smoothed_cycle_times, smoothed_cycle_extents)
        from numpy import nanmean

        self.df_detailed = df_detailed
        if self.alternative_used:
            avg_dict = {
                'Global_AvgRate_Hz': global_stats['mean_rate'],
                'Global_AvgExtent_cents': global_stats['mean_extent'],
                'Global_Jitter': global_stats['jitter'],
                'Global_CV_Rate_%': cv_rate,
                'Global_CV_Extent_%': cv_extent,
                'File_Name': os.path.basename(self.file_path),
                'HarmonicUsed': f"Harmonic {self.harmonic_used}",
                'HarmonicFrequency': self.harmonic_frequency
            }
        else:
            avg_dict = {
                'Global_AvgRate_Hz': global_stats['mean_rate'],
                'Global_AvgExtent_cents': global_stats['mean_extent'],
                'Global_Jitter': global_stats['jitter'],
                'Global_CV_Rate_%': cv_rate,
                'Global_CV_Extent_%': cv_extent,
                'File_Name': os.path.basename(self.file_path),
                'HarmonicUsed': "Fundamental",
                'HarmonicFrequency': np.nanmedian(self.pitch_hz)
            }
        self.df_avg = pd.DataFrame([avg_dict])

        if self.alternative_used:
            harmonic_used_str = f"Harmonic {self.harmonic_used}"
            harmonic_freq_val = self.harmonic_frequency
        else:
            harmonic_used_str = "Fundamental"
            harmonic_freq_val = np.nanmedian(self.pitch_hz)
        self.df_region = pd.DataFrame(region_list)
        summary_data_file = {
            'mean_rate_unsmoothed': global_stats['mean_rate'],
            'stdev_rate_unsmoothed': global_stats['stdev_rate'],
            'median_rate_unsmoothed': global_stats['median_rate'],
            'mean_extent_unsmoothed': global_stats['mean_extent'],
            'stdev_extent_unsmoothed': global_stats['stdev_extent'],
            'median_extent_unsmoothed': global_stats['median_extent'],
            'mean_rate_smooth': global_stats_smooth['mean_rate'] if len(smoothed_cycle_times) else np.nan,
            'stdev_rate_smooth': global_stats_smooth['stdev_rate'] if len(smoothed_cycle_times) else np.nan,
            'median_rate_smooth': global_stats_smooth['median_rate'] if len(smoothed_cycle_times) else np.nan,
            'mean_extent_smooth': global_stats_smooth['mean_extent'] if len(smoothed_cycle_times) else np.nan,
            'stdev_extent_smooth': global_stats_smooth['stdev_extent'] if len(smoothed_cycle_times) else np.nan,
            'median_extent_smooth': global_stats_smooth['median_extent'] if len(smoothed_cycle_times) else np.nan,
            'Global_Jitter': global_stats['jitter'],
            'Global_CV_Rate_%': cv_rate,
            'Global_CV_Extent_%': cv_extent,
            'mean_pitch_hz': nanmean(self.pitch_hz),
            'min_pitch_hz': np.nanmin(self.pitch_hz),
            'max_pitch_hz': np.nanmax(self.pitch_hz),
            'HarmonicUsed': harmonic_used_str,
            'HarmonicFrequency': harmonic_freq_val
        }
        self.summary_data = summary_data_file

        # Compute Sample Entropy
        vibrato_rate_series = df_detailed['VibratoRate'].dropna().values
        vibrato_extent_series = df_detailed['half_extent_cents'].dropna().values
        sampen_rate = sample_entropy(vibrato_rate_series, m=2, r=0.2 * np.std(vibrato_rate_series))
        sampen_extent = sample_entropy(vibrato_extent_series, m=2, r=0.2 * np.std(vibrato_extent_series))

        # Attach sample entropy values to summary_data
        self.summary_data['SampEn_Rate'] = sampen_rate
        self.summary_data['SampEn_Extent'] = sampen_extent

        fig3 = plot_vibrato_rate(df_detailed, df_smoothed, save_path="Figure_3.png")
        self.generated_figures.append(fig3)

        fig_summary = final_plot(
            df_detailed, df_smoothed, self.summary_data,
            jitter_metrics=compute_jitter_metrics(df_detailed['cycle_time'].values * 2),
            cv_rate=cv_rate, cv_extent=cv_extent,
            cv_rate_smooth=cv_rate_smooth, cv_extent_smooth=cv_extent_smooth,
            filename=self.file_path,
            sampen_rate=sampen_rate, sampen_extent=sampen_extent,
            title="Vibrato Scope: Final Analysis",
            show_figure=False
        )
        self.generated_figures.append("Figure_summary.png")
        self.last_fig = fig_summary

        # ONLY CHANGE #1: Update each region with all final-table parameters.
        for reg_item in region_list:
            reg_item.update(self.summary_data)
        self.df_region = pd.DataFrame(region_list)

        # ONLY CHANGE #2: Build averaged df by copying all summary_data fields.
        new_avg_dict = self.summary_data.copy()
        new_avg_dict['File_Name'] = os.path.basename(self.file_path)
        self.df_avg = pd.DataFrame([new_avg_dict])

        self.save_results()

    def batch_process(self):
        file_paths = filedialog.askopenfilenames(title="Select WAV Files for Batch Processing",
                                                 filetypes=[("WAV Files", "*.wav")])
        if not file_paths:
            return
        save_dir = filedialog.askdirectory(title="Select folder to save batch results")
        if not save_dir:
            return
        batch_avg_list = []
        batch_detailed_list = []
        batch_region_list = []
        all_cycle_params = []
        self.generated_figures = []
        for file_path in file_paths:
            y, sr = load_audio(file_path)
            times, pitch_hz = extract_pitch_yin(y, sr, fmin=50, fmax=1500)
            if pitch_hz is None or np.all(np.isnan(pitch_hz)):
                continue
            pitch_hz = filter_pitch_outliers(pitch_hz, threshold=3)
            cents = convert_to_cents(pitch_hz)
            cents_centered = remove_mean_or_median(cents, use_median=True)
            t_uniform, cents_uniform = resample_to_uniform_time(times, cents_centered, new_sr=100)
            if t_uniform is None:
                continue
            if (t_uniform[-1] - t_uniform[0]) > 1.0:
                onset_offset = 0.2
                mask = (t_uniform >= t_uniform[0] + onset_offset) & (t_uniform <= t_uniform[-1] - onset_offset)
                t_uniform = t_uniform[mask]
                cents_uniform = cents_uniform[mask]
            cents_filtered = apply_bandpass_filter(cents_uniform, fs=100, lowcut=3.0, highcut=9.0, order=4)
            peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                t_uniform, cents_uniform, cents_filtered, prominence=5, distance=5)
            if len(all_idx) < 2:
                continue
            cycle_params = compute_cycle_parameters(t_valid, cents_uniform, half_cycle_extents, all_idx)
            filtered_params = filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0)
            if not filtered_params:
                continue
            region_start = t_uniform[0]
            region_end = t_uniform[-1]
            from numpy import isnan
            reg_mask = (times >= region_start) & (times <= region_end)
            region_pitch = pitch_hz[reg_mask]
            region_avg_F0 = np.nanmean(region_pitch) if np.any(~isnan(region_pitch)) else np.nan
            df_file = pd.DataFrame(filtered_params)
            df_file['File_Name'] = os.path.basename(file_path)
            num_cycles = len(df_file)
            if num_cycles < 5:
                continue

            batch_detailed_list.append(df_file)
            region_rate = 1 / (2 * df_file['cycle_time'])
            region_avg_rate = region_rate.mean()
            region_std_rate = region_rate.std()
            region_median_rate = np.median(region_rate)
            region_avg_extent = df_file['half_extent_cents'].mean()
            region_std_extent = df_file['half_extent_cents'].std()
            region_median_extent = np.median(df_file['half_extent_cents'])
            region_jitter = df_file['cycle_time'].std() / df_file['cycle_time'].mean()
            region_cv_rate = (region_std_rate / region_avg_rate) * 100
            region_cv_extent = (df_file['half_extent_cents'].std() / df_file['half_extent_cents'].mean()) * 100

            region_item = {
                'Region_Start': region_start,
                'Region_End': region_end,
                'Num_Cycles': num_cycles,
                'Avg_Rate_Hz': region_avg_rate,
                'StDev_Rate_Hz': region_std_rate,
                'Median_Rate_Hz': region_median_rate,
                'Avg_Extent_cents': region_avg_extent,
                'StDev_Extent_cents': region_std_extent,
                'Median_Extent_cents': region_median_extent,
                'Jitter': region_jitter,
                'CV_Rate_%': region_cv_rate,
                'CV_Extent_%': region_cv_extent,
                'Avg_F0_Hz': region_avg_F0,
                'File_Name': os.path.basename(file_path)
            }

            all_cycle_params.extend(filtered_params)
            batch_region_list.append(pd.DataFrame([region_item]))

            cycle_times_all = df_file['cycle_time'].values
            cycle_extents_all = df_file['half_extent_cents'].values
            global_stats = analyze_vibrato(cycle_times_all, cycle_extents_all)
            cv_rate_file, cv_extent_file = compute_cv(cycle_times_all, cycle_extents_all)
            from numpy import nanmean
            vibrato_rate_series = (1 / (2 * df_file['cycle_time'])).dropna().values
            vibrato_extent_series = df_file['half_extent_cents'].dropna().values
            sampen_rate = sample_entropy(vibrato_rate_series, m=2, r=0.2 * np.std(vibrato_rate_series))
            sampen_extent = sample_entropy(vibrato_extent_series, m=2, r=0.2 * np.std(vibrato_extent_series))

            window_size = 3
            df_smoothed = smooth_vibrato_parameters(df_file, window_size=window_size)
            smoothed_cycle_times = df_smoothed[('cycle_time', 'mean')].dropna().values
            smoothed_cycle_extents = df_smoothed[('half_extent_cents', 'mean')].dropna().values
            global_stats_smooth = analyze_vibrato(smoothed_cycle_times, smoothed_cycle_extents)
            cv_rate_smooth, cv_extent_smooth = compute_cv(smoothed_cycle_times, smoothed_cycle_extents)

            summary_data_file = {
                'mean_rate_unsmoothed': global_stats['mean_rate'],
                'stdev_rate_unsmoothed': global_stats['stdev_rate'],
                'median_rate_unsmoothed': global_stats['median_rate'],
                'mean_extent_unsmoothed': global_stats['mean_extent'],
                'stdev_extent_unsmoothed': global_stats['stdev_extent'],
                'median_extent_unsmoothed': global_stats['median_extent'],
                'mean_rate_smooth': global_stats_smooth['mean_rate'] if len(smoothed_cycle_times) else np.nan,
                'stdev_rate_smooth': global_stats_smooth['stdev_rate'] if len(smoothed_cycle_times) else np.nan,
                'median_rate_smooth': global_stats_smooth['median_rate'] if len(smoothed_cycle_times) else np.nan,
                'mean_extent_smooth': global_stats_smooth['mean_extent'] if len(smoothed_cycle_times) else np.nan,
                'stdev_extent_smooth': global_stats_smooth['stdev_extent'] if len(smoothed_cycle_times) else np.nan,
                'median_extent_smooth': global_stats_smooth['median_extent'] if len(smoothed_cycle_times) else np.nan,
                'Global_Jitter': global_stats['jitter'],
                'Global_CV_Rate_%': cv_rate_file,
                'Global_CV_Extent_%': cv_extent_file,
                'mean_pitch_hz': nanmean(pitch_hz),
                'min_pitch_hz': np.nanmin(pitch_hz),
                'max_pitch_hz': np.nanmax(pitch_hz),
                'HarmonicUsed': "Fundamental",
                'HarmonicFrequency': np.nanmedian(pitch_hz),
                'SampEn_Rate': sampen_rate,
                'SampEn_Extent': sampen_extent
            }

            # ONLY CHANGE #1 (batch): Update region_item with summary_data_file
            region_item.update(summary_data_file)

            batch_region_list.append(pd.DataFrame([region_item]))

            avg_dict = summary_data_file.copy()
            avg_dict['File_Name'] = os.path.basename(file_path)
            # ONLY CHANGE #2 (batch): Append complete summary data for this file.
            batch_avg_list.append(avg_dict)

            fig_b = plot_before_after_filter(t_uniform, cents_uniform, cents_filtered,
                                             save_path=os.path.join(save_dir,
                                                                    f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_1.png"))
            self.generated_figures.append(fig_b)
            fig_p = plot_peaks_troughs(t_valid, cents_valid, peaks, troughs,
                                       save_path=os.path.join(save_dir,
                                                              f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_2.png"))
            self.generated_figures.append(fig_p)
            fig_r = plot_vibrato_rate(df_file, df_smoothed,
                                      save_path=os.path.join(save_dir,
                                                             f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_3.png"))
            self.generated_figures.append(fig_r)
            fig_sum = final_plot(
                df_file, df_smoothed, summary_data_file,
                jitter_metrics=compute_jitter_metrics(df_file['cycle_time'].values * 2),
                cv_rate=cv_rate_file, cv_extent=cv_extent_file,
                cv_rate_smooth=cv_rate_smooth, cv_extent_smooth=cv_extent_smooth,
                filename=file_path,
                sampen_rate=sampen_rate,
                sampen_extent=sampen_extent,
                title="Vibrato Scope: Final Analysis",
                show_figure=False
            )
            fig_sum_path = os.path.join(save_dir,
                                        f"{os.path.splitext(os.path.basename(file_path))[0]}_final_analysis.png")
            self.generated_figures.append(fig_sum_path)
            fig_sum.savefig(fig_sum_path, dpi=300)
            plt.close(fig_sum)

        if not batch_avg_list:
            messagebox.showinfo("Batch Processing", "No valid vibrato cycles found in the selected files.")
            return

        df_avg_all = pd.DataFrame(batch_avg_list)
        df_detailed_all = pd.concat(batch_detailed_list, ignore_index=True)
        df_region_all = pd.concat(batch_region_list, ignore_index=True)

        df_avg_all.to_csv(os.path.join(save_dir, "averaged_vibrato_data.csv"), index=False)
        df_detailed_all.to_csv(os.path.join(save_dir, "detailed_vibrato_data.csv"), index=False)
        df_region_all.to_csv(os.path.join(save_dir, "region_vibrato_data.csv"), index=False)

        messagebox.showinfo("Batch Processing", f"Batch processing complete.\nResults saved in:\n{save_dir}")

    def save_results(self):
        if self.last_fig is None or self.df_detailed is None:
            messagebox.showwarning("Save Results", "No analysis results to save. Please run analysis first.")
            return
        save_dir = filedialog.askdirectory(title="Select folder to save results")
        if not save_dir:
            return
        region_path = os.path.join(save_dir, "region_vibrato_data.csv")
        self.df_region.to_csv(region_path, index=False)
        avg_path = os.path.join(save_dir, "averaged_vibrato_data.csv")
        self.df_avg.to_csv(avg_path, index=False)
        detailed_path = os.path.join(save_dir, "detailed_vibrato_data.csv")
        if hasattr(self, "batch_detailed_list") and self.batch_detailed_list:
            df_detailed_all = pd.concat(self.batch_detailed_list, ignore_index=True)
            df_detailed_all.to_csv(detailed_path, index=False)
        else:
            self.df_detailed.to_csv(detailed_path, index=False)
        for fig_name in self.generated_figures:
            src = os.path.abspath(fig_name)
            dst = os.path.abspath(os.path.join(save_dir, fig_name))
            if src != dst:
                try:
                    shutil.copy(fig_name, dst)
                except shutil.SameFileError:
                    pass
        figure_path = os.path.join(save_dir, "final_analysis.png")
        self.last_fig.canvas.draw()
        self.last_fig.savefig(figure_path, dpi=300)
        messagebox.showinfo("Save Results", f"Results saved in:\n{save_dir}")


def main():
    root = tk.Tk()
    show_splash(root)
    app = VibratoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
