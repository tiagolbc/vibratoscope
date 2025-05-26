# spectrogram.py

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import parselmouth
from parselmouth import SpectralAnalysisWindowShape
from utils import frequency_to_note_name

########################################
# PLOTTING FUNCTIONS
########################################

def plot_spectrogram(ax, audio_data, sr, times, pitch_hz,
                     spectrogram_type, file_path, pitch_method,
                     vmin=-80, vmax=0):
    """
    Plots the selected type of spectrogram with optional pitch overlay.
    """
    ax.clear()
    valid = ~np.isnan(pitch_hz)
    max_freq = 5000  # Max frequency for spectrogram display

    if spectrogram_type == "standard":
        n_fft = 2048
        hop_length = 256
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        freq_mask = freqs <= max_freq
        D = D[freq_mask, :]
        librosa.display.specshow(
            D,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=ax,
            cmap='magma',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_ylim(0, max_freq)
        ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])

    elif spectrogram_type == "mel":
        n_mels = 256
        S = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_fft=2048,
            hop_length=256,
            n_mels=n_mels,
            fmax=max_freq
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=256,
            x_axis='time',
            y_axis='hz',
            fmax=max_freq,
            ax=ax,
            cmap='magma',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_ylim(0, max_freq)
        ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])

    elif spectrogram_type == "praat_narrow":
        snd = parselmouth.Sound(file_path)
        spec = snd.to_spectrogram(
            window_length=0.03,
            maximum_frequency=max_freq,
            window_shape=SpectralAnalysisWindowShape.GAUSSIAN
        )
        S = spec.values
        db = 10 * np.log10(S + np.finfo(float).eps)
        max_db = db.max()
        db = np.clip(db, max_db - 70, max_db)
        extent = [spec.xmin, spec.xmax, spec.ymin, spec.ymax]
        ax.imshow(
            db,
            origin='lower',
            extent=extent,
            aspect='auto',
            cmap='Greys_r'
        )
        ax.set_xlim(spec.xmin, spec.xmax)
        ax.set_ylim(0, max_freq)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Praat‐style Narrow‐band Spectrogram & F0 ({pitch_method.upper()})")

    else:  # cqt fallback
        n_fft = 2048
        hop_length = 256
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        freq_mask = freqs <= max_freq
        D = D[freq_mask, :]
        ts = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)
        ax.imshow(
            D,
            origin='lower',
            aspect='auto',
            extent=[ts[0], ts[-1], 0, max_freq],
            cmap='inferno',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_ylim(0, max_freq)
        ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])

    # Clear previous pitch axis if it exists
    if hasattr(ax, '_pitch_axis'):
        try:
            ax._pitch_axis.remove()
            del ax._pitch_axis
        except Exception:
            pass

    # Overlay pitch contour with secondary axis
    if valid.any():
        pitch_valid = pitch_hz[valid]
        times_valid = times[valid]

        ax2 = ax.twinx()  # Create new twin axis for pitch
        ax2.set_ylim(50, 1000)
        ax2.set_ylabel("F0 (Hz)", color='#37bdb4')
        ax2.tick_params(axis='y', colors='#37bdb4')
        ax2.spines['right'].set_color('#37bdb4')

        pitch_mask = (pitch_valid >= 50) & (pitch_valid <= 1000)
        if np.any(pitch_mask):
            ax2.plot(
                times_valid[pitch_mask],
                pitch_valid[pitch_mask],
                color='#37bdb4',
                linewidth=2.5,
            )

        # Plot an "invisible" curve just for legend on ax (main spectrogram axis)
        ax.plot([], [], color='#37bdb4', linewidth=2.5, label=f"F0 ({pitch_method.upper()})")
        ax.legend(loc='upper right', fontsize=10)

        # Save reference to ax2 to clear it next time
        ax._pitch_axis = ax2

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{spectrogram_type.replace('_', ' ').title()} Spectrogram & F0 ({pitch_method.upper()})")


def plot_before_after_filter(t_uniform, cents_uniform, cents_filtered, save_path="Figure_1.png"):
    """
    Plots the centered pitch and filtered pitch for comparison.
    """
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


def plot_peaks_troughs(t_valid, cents_filtered, peaks, troughs, save_path="Figure_2.png"):
    """
    Plots the filtered signal with detected peaks and troughs.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_valid, cents_filtered, label="Filtered Signal")
    ax.plot(t_valid[peaks], cents_filtered[peaks], 'ro', label="Peaks")
    ax.plot(t_valid[troughs], cents_filtered[troughs], 'go', label="Troughs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cents (centered)")
    ax.set_title("Peak and Trough Detection in Filtered Signal")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_vibrato_rate(df, df_smoothed, save_path="Figure_3.png"):
    """
    Plots the vibrato rate per half cycle.
    """
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


def final_plot(df, df_smoothed, summary_data, jitter_metrics, cv_rate, cv_extent,
               cv_rate_smooth, cv_extent_smooth, filename,
               sampen_rate, sampen_extent, sampen_cycle_time, sampen_target, sampen_distance,
               title="Vibrato Scope: Final Analysis",
               show_figure=True):

    """
    Creates a final summary plot with 6 subplots and a metrics table.
    """
    from matplotlib.gridspec import GridSpec
    import os
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.8], figure=fig)
    fig.subplots_adjust(hspace=0.6, wspace=0.3, left=0.05, right=0.95, top=0.88, bottom=0.05)
    fig.suptitle(f"{title}\nFile: {os.path.basename(filename)}\nPitch Method: {summary_data['PitchMethod'].upper()}",
                 fontsize=14, fontweight='bold')

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
    ax5.set_title("Coefficient of Variation", fontsize=10)
    ax5.set_xlabel("Cycle #", fontsize=9)
    ax5.set_ylabel("Normalized", fontsize=9)
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])

    labels = ["Rate", "Extent"]
    values = [
        sampen_rate if np.isfinite(sampen_rate) else 0,
        sampen_extent if np.isfinite(sampen_extent) else 0
    ]
    colors = ["blue", "purple"]

    ax6.bar(labels, values, color=colors)

    # Ajustar espaço vertical automaticamente
    max_val = max(values) if values else 1
    offset = max_val * 0.1 if max_val > 0 else 0.1

    ax6.text(0, values[0] + offset, f"{sampen_rate:.2f}" if np.isfinite(sampen_rate) else "∞", ha='center', fontsize=9)
    ax6.text(1, values[1] + offset, f"{sampen_extent:.2f}" if np.isfinite(sampen_extent) else "∞", ha='center',
             fontsize=9)

    ax6.set_ylim(0, max_val + 2 * offset)  # Evita sobreposição com o título
    ax6.set_title("Sample Entropy (Rate & Extent)", fontsize=10)

    # Row 2: Summary table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')
    col_titles = [
        "Smoothed Data",
        "Unsmoothed Data (Hz)",
        "Vibrato Jitter (Unsmoothed)",
        "Coefficient of Variation",
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
                         f"{additional_data}\n"
                         f"Global Jitter: {summary_data.get('Global_Jitter', 0):.2f}\n"
                         f"SampEn Rate: {sampen_rate:.2f}" if np.isfinite(sampen_rate) else "SampEn Rate: ∞\n"
                     ) + "\n" + (
                         f"SampEn Extent: {sampen_extent:.2f}" if np.isfinite(sampen_extent) else "SampEn Extent: ∞\n"
                     ) + "\n" + (
                         f"SampEn Cycle Time: {sampen_cycle_time:.2f}" if np.isfinite(
                             sampen_cycle_time) else "SampEn Cycle Time: ∞\n"
                     ) + "\n" + (
                         f"SampEn Distance: {sampen_distance}\n"
                         f"Harmonic Used: {summary_data.get('HarmonicUsed', 'N/A')}\n"
                         f"Harmonic Frequency: {summary_data.get('HarmonicFrequency', 'N/A')}"
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
    fig.savefig("Figure_Analysis_Vibrato.png", dpi=300)
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    return fig