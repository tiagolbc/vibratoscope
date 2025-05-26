import parselmouth
from parselmouth.praat import call
import numpy as np
import traceback


def normalize_intensity_and_get_global_peak(sound, target_dB=70):
    sound.scale_intensity(target_dB)
    # Use Praat call to create LTAS (Long-Term Average Spectrum)
    ltas = call(sound, "To Ltas", 100.0)  # Bandwidth = 100 Hz (or 10.0 if you prefer)
    global_peak_dB = call(ltas, "Get maximum", 0, 0, "Parabolic")
    return sound, global_peak_dB


from parselmouth import WindowShape

def extract_gaussian_frames(sound, frame_length=0.1, time_step=0.0033):
    total_duration = sound.get_total_duration()
    n_frames = int(round((total_duration - frame_length) / time_step)) + 2
    frames = []
    for i in range(n_frames):
        start = i * time_step
        end = start + frame_length
        part = sound.extract_part(
            from_time=start,
            to_time=end,
            window_shape=WindowShape.GAUSSIAN1,  # ✅ nome correto
            relative_width=1.0,
            preserve_times=True
        )
        frames.append(part)
    return frames


from parselmouth.praat import call

def analyze_frame_ltas(frame, max_freq=700):
    ltas = call(frame, "To Ltas", 6.0)  # ✅ correto
    full_max_dB = call(ltas, "Get maximum", 0, 0, "none")
    dominant_peak_dB = call(ltas, "Get maximum", 50, max_freq, "none")
    dominant_peak_Hz = call(ltas, "Get frequency of maximum", 50, max_freq, "none")

    subbands = []
    for z in range(2, 8):
        start = (z - 1) / 8 * dominant_peak_Hz
        end = z / 8 * dominant_peak_Hz
        dB = call(ltas, "Get maximum", start, end, "none")
        Hz = call(ltas, "Get frequency of maximum", start, end, "none")
        subbands.append((Hz, dB))

    return full_max_dB, dominant_peak_Hz, dominant_peak_dB, subbands, ltas


def dominant_spectrum_test(dominant_peak_Hz, dominant_peak_dB, subbands, diff1=10):
    for i in range(len(subbands) - 1):
        hz_i, dB_i = subbands[i]
        hz_next, dB_next = subbands[i + 1]
        if dB_i > dominant_peak_dB - diff1:
            if dB_i >= dB_next:
                return hz_i, dB_i
            elif dB_next > dominant_peak_dB - diff1:
                return hz_next, dB_next
    return dominant_peak_Hz, dominant_peak_dB

from parselmouth.praat import call

def sequential_spectrum_test(ltas, prev_f0_Hz, prev_f0_dB, full_frame_dB, diff2=10, noise_threshold=20):
    if prev_f0_Hz == 0 or prev_f0_dB == 0:
        return None

    lower = 0.9 * prev_f0_Hz
    upper = 1.1 * prev_f0_Hz

    dB = call(ltas, "Get maximum", lower, upper, "none")
    Hz = call(ltas, "Get frequency of maximum", lower, upper, "none")

    if abs(dB - prev_f0_dB) <= diff2 and dB > full_frame_dB - noise_threshold:
        return Hz, dB
    return None


from parselmouth.praat import call

def harmonic_correction(ltas, f0_Hz, f0_dB, full_frame_dB, diff3=10, noise_threshold=20):
    corrected_Hz, corrected_dB = f0_Hz, f0_dB
    for factor in [0.5, 2/3]:
        lower = factor * f0_Hz - 10
        upper = factor * f0_Hz + 10
        if lower > 0:
            dB = call(ltas, "Get maximum", lower, upper, "none")
            Hz = call(ltas, "Get frequency of maximum", lower, upper, "none")
            if dB > f0_dB - diff3 and dB > full_frame_dB - noise_threshold:
                corrected_Hz, corrected_dB = Hz, dB
    return corrected_Hz, corrected_dB


def finalize_frame_f0(f0_Hz, f0_dB, global_max_dB, full_frame_dB, silence_threshold=30, min_f0_Hz=50):
    if f0_dB < global_max_dB - silence_threshold:
        return 0.0
    if f0_Hz < min_f0_Hz:
        return 0.0
    return f0_Hz

def run_sfeeds(sound, frame_length=0.1, time_step=0.0033, avg_int=70, max_freq=700,
               diff1=10, diff2=10, diff3=10, silence_threshold=30, noise_threshold=20):
    try:
        sound, global_max_dB = normalize_intensity_and_get_global_peak(sound, avg_int)
        frames = extract_gaussian_frames(sound, frame_length, time_step)
        f0_contour = []
        prev_f0_Hz, prev_f0_dB = 0, 0

        for i, frame in enumerate(frames):
            time = i * time_step + frame_length / 2
            full_frame_dB, dom_Hz, dom_dB, subbands, ltas = analyze_frame_ltas(frame, max_freq)
            f0_Hz, f0_dB = dominant_spectrum_test(dom_Hz, dom_dB, subbands, diff1)
            seq_result = sequential_spectrum_test(ltas, prev_f0_Hz, prev_f0_dB, full_frame_dB, diff2, noise_threshold)
            if seq_result:
                f0_Hz, f0_dB = seq_result
            f0_Hz, f0_dB = harmonic_correction(ltas, f0_Hz, f0_dB, full_frame_dB, diff3, noise_threshold)
            final_f0 = finalize_frame_f0(f0_Hz, f0_dB, global_max_dB, full_frame_dB, silence_threshold)
            f0_contour.append((time, final_f0))
            prev_f0_Hz = final_f0
            prev_f0_dB = f0_dB if final_f0 > 0 else 0

        return np.array(f0_contour)

    except Exception as e:
        import traceback
        print("❌ Error in run_sfeeds():")
        traceback.print_exc()
        raise e


def extract_pitch_sfeeds(audio_data, sr, fmin=50.0, fmax=700.0, time_step=0.0033, **kwargs):
    import parselmouth
    import numpy as np

    if isinstance(audio_data, str):
        # Backup: load from path
        print("[SFEEDS] audio_data is path. Loading Sound from file.")
        sound = parselmouth.Sound(audio_data)
    elif isinstance(audio_data, np.ndarray):
        print("[SFEEDS] audio_data is array. Creating Sound from numpy.")
        sound = parselmouth.Sound(audio_data, sampling_frequency=sr)
    elif isinstance(audio_data, parselmouth.Sound):
        print("[SFEEDS] audio_data is already a Praat Sound.")
        sound = audio_data
    else:
        raise ValueError("[SFEEDS] Invalid audio input: expected file path, numpy array or Sound object.")

    if sound is None:
        raise ValueError("[SFEEDS] Sound object could not be created. Aborting.")

    f0_contour = run_sfeeds(sound, time_step=time_step, max_freq=fmax)
    times = f0_contour[:, 0]
    f0s = f0_contour[:, 1]
    return times, f0s




