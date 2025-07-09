# gui.py

import os
import sys
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from pitch import extract_pitch_with_yin, extract_pitch_with_praat, extract_pitch_with_hps, extract_pitch_with_reaper

import sounddevice as sd
import pyreaper
import shutil

# Import from your own modules (without dot!)
from pitch import load_audio, extract_pitch_with_praat, extract_pitch_with_yin, extract_pitch_with_hps, extract_pitch_with_reaper, extract_pitch_with_sfeeds
from filters import filter_pitch_outliers, apply_bandpass_filter
from utils import convert_to_cents, remove_mean_or_median, resample_to_uniform_time, sample_entropy
from spectrogram import plot_spectrogram, plot_before_after_filter, plot_peaks_troughs, final_plot, plot_vibrato_rate
from core import detect_vibrato_cycles, compute_cycle_parameters, filter_vibrato_cycles, compute_cv, analyze_vibrato, create_vibrato_dataframe, smooth_vibrato_parameters, compute_jitter_metrics
from audio_recorder import AudioRecorder
from pathlib import Path


########################################
# MAIN GUI CLASS
########################################
class VibratoGUI:
    def __init__(self, root):
        from PIL import Image, ImageTk
        import tkinter as tk
        from tkinter import font as tkFont
        from tkinter import ttk  # Import ttk from tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt

        self.root = root
        self.root.title("Vibrato Scope - version 1.1.0")
        self.root.configure(bg="#b0afa6")

        # Fonts
        cal_font = tkFont.Font(family="Calibri", size=11)
        cal_bold = tkFont.Font(family="Calibri", size=12, weight="bold")

        # Control variables
        self.audio_data = None
        self.recording_stopped = False
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
        self.spectrogram_type = tk.StringVar(value="standard")
        self.pitch_method = tk.StringVar(value="praat")
        self.actual_pitch_method = tk.StringVar(value="praat")
        self.sampen_target = tk.StringVar(value="rate")
        self.pitch_method.trace_add("write", self.on_pitch_method_change)
        self.sampen_distance = tk.StringVar(value="chebyshev")

        # --- TOPO: Logo + controls ---
        self.top = tk.Frame(root, bg="#b0afa6")
        self.top.grid(row=0, column=0, sticky="ew", padx=10, pady=2)

        try:
            logo_image = Image.open("logo.png")
            logo_image = logo_image.resize((180, 180))
            self.logo = ImageTk.PhotoImage(logo_image)
            tk.Label(self.top, image=self.logo, bg="#b0afa6").pack(side=tk.LEFT, padx=(5, 20), anchor="center")
        except Exception:
            tk.Label(self.top, text="[Logo not found]", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=(5, 20),
                                                                                          anchor="center")

        btn_frame = tk.Frame(self.top, bg="#b0afa6")
        btn_frame.pack(side=tk.LEFT, expand=True)

        top_row = tk.Frame(btn_frame, bg="#b0afa6")
        top_row.pack(side=tk.TOP, anchor="center")

        self.btn_select = tk.Button(top_row, text="Select WAV File", command=self.load_file, font=cal_font)
        self.btn_select.pack(side=tk.LEFT, padx=3, pady=2, ipadx=6, ipady=2)

        self.btn_play = tk.Button(top_row, text="Play Audio", command=self.play_audio, state=tk.DISABLED, font=cal_font)
        self.btn_play.pack(side=tk.LEFT, padx=3, pady=2, ipadx=6, ipady=2)

        tk.Label(top_row, text="Fmin:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=(5, 0))
        self.fmin = tk.StringVar(value="50.114")
        tk.Entry(top_row, textvariable=self.fmin, width=5, font=cal_font).pack(side=tk.LEFT, padx=(0, 5))

        tk.Label(top_row, text="Fmax:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT)
        self.fmax = tk.StringVar(value="1500")
        tk.Entry(top_row, textvariable=self.fmax, width=5, font=cal_font).pack(side=tk.LEFT, padx=(0, 5))

        self.btn_batch = tk.Button(top_row, text="Batch Process", command=self.batch_process, font=cal_font)
        self.btn_batch.pack(side=tk.LEFT, padx=3, pady=2, ipadx=6, ipady=2)

        tk.Label(top_row, text="Spec Type:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT)
        spec_options = ["standard", "mel", "cqt", "praat_narrow"]
        spec_menu = ttk.Combobox(top_row, textvariable=self.spectrogram_type, values=spec_options,
                                 width=10, state="readonly", font=cal_font)  # Use ttk.Combobox
        spec_menu.pack(side=tk.LEFT, padx=(2, 5))

        self.btn_update_spec = tk.Button(top_row, text="Update", command=self.update_spectrogram,
                                         state=tk.DISABLED, font=cal_font)
        self.btn_update_spec.pack(side=tk.LEFT, padx=3, pady=2, ipadx=6, ipady=2)

        # Pitch method row
        pitch_frame = tk.Frame(btn_frame, bg="#b0afa6")
        pitch_frame.pack(side=tk.TOP, pady=(5, 0), anchor="center")

        tk.Label(pitch_frame, text="fo Tracking Method:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=(5, 5))
        for method in ["praat", "yin", "hps", "reaper", "sfeeds"]:
            tk.Radiobutton(pitch_frame, text=method.upper(), variable=self.pitch_method, value=method,
                           bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=2)

        # --- ESPECTROGRAMA ---
        self.specf = tk.Frame(root, relief="sunken", borderwidth=1)
        self.specf.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.specf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.specf)

        self.ax.set_title("Load Audio File - Spectrogram will appear here")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")

        # --- INPUT/OUTPUT SELECTION ---
        audio_frame = tk.Frame(root, bg="#b0afa6")
        audio_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        tk.Label(audio_frame, text="Sound Input:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=(5, 2))
        self.input_device_var = tk.StringVar()
        self.input_combo = ttk.Combobox(audio_frame, textvariable=self.input_device_var, width=40, state="readonly",
                                        font=cal_font)  # Use ttk.Combobox
        self.input_combo.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(audio_frame, text="Sound Output:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=(5, 2))
        self.output_device_var = tk.StringVar()
        self.output_combo = ttk.Combobox(audio_frame, textvariable=self.output_device_var, width=40, state="readonly",
                                         font=cal_font)  # Use ttk.Combobox
        self.output_combo.pack(side=tk.LEFT)

        self.populate_audio_devices()

        # --- BASE: Botões + Regiões ---
        self.bot = tk.Frame(root, bg="#b0afa6")
        self.bot.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        # Left: main buttons
        left = tk.Frame(self.bot, bg="#b0afa6")
        left.pack(side=tk.LEFT)

        self.btn_record = tk.Button(left, text="Record Audio", command=self.open_audio_recorder, font=cal_font)
        self.btn_record.pack(side=tk.LEFT, padx=5)

        self.btn_analyze = tk.Button(left, text="Run Analysis", command=self.run_analysis, state=tk.DISABLED,
                                     font=cal_font)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(left, text="Save Results", command=self.save_results, state=tk.DISABLED,
                                  font=cal_font)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_exit = tk.Button(left, text="Exit", command=self.on_closing, font=cal_font)
        self.btn_exit.pack(side=tk.LEFT, padx=5)

        # Center: donate button
        center = tk.Frame(self.bot, bg="#b0afa6")
        center.pack(side=tk.LEFT, expand=True)
        self.btn_donate = tk.Button(center, text="  D O N A T E  ", command=self.on_donate,
                                    font=cal_font, padx=10, pady=6, relief="raised", borderwidth=3)
        self.btn_donate.pack()

        # Right: regions + clear
        right = tk.Frame(self.bot, bg="#b0afa6")
        right.pack(side=tk.RIGHT)

        self.lb = tk.Listbox(right, width=40, height=10, font=cal_font)
        self.lb.pack(side=tk.LEFT)

        self.btn_clear = tk.Button(right, text="Clear Regions", command=self.clear_regions, font=cal_font)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # Choose SampEn Metric
        sampen_frame = tk.Frame(right, bg="#b0afa6")
        sampen_frame.pack(side=tk.TOP, pady=5)
        tk.Label(sampen_frame, text="SampEn Distance:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        sampen_menu = ttk.Combobox(sampen_frame, textvariable=self.sampen_distance,
                                   values=["chebyshev", "euclidean", "manhattan"],
                                   width=10, state="readonly", font=cal_font)  # Use ttk.Combobox
        sampen_menu.pack(side=tk.LEFT)

        # SampEn Target Selection
        target_frame = tk.Frame(right, bg="#b0afa6")
        target_frame.pack(side=tk.TOP, pady=5)
        tk.Label(target_frame, text="SampEn Target:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        target_menu = ttk.Combobox(target_frame, textvariable=self.sampen_target,
                                   values=["rate", "cycle_time", "extent"],
                                   width=12, state="readonly", font=cal_font)  # Use ttk.Combobox
        target_menu.pack(side=tk.LEFT)

        # Add to SampEn Target Selection frame
        tk.Label(target_frame, text="SampEn m:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.sampen_m = tk.StringVar(value="2")
        tk.Entry(target_frame, textvariable=self.sampen_m, width=5, font=cal_font).pack(side=tk.LEFT, padx=5)

        tk.Label(target_frame, text="SampEn r:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.sampen_r = tk.StringVar(value="0.2")
        tk.Entry(target_frame, textvariable=self.sampen_r, width=5, font=cal_font).pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        root.grid_rowconfigure(1, weight=1)  # specf row expands
        root.grid_rowconfigure(0, weight=0)  # top row fixed
        root.grid_rowconfigure(2, weight=0)  # audio_frame row fixed
        root.grid_rowconfigure(3, weight=0)  # bot row fixed
        root.grid_columnconfigure(0, weight=1)  # column expands

        self.span = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Configure>", self.on_resize)
        self.resize_after_id = None
        self.last_width = None
        self.last_height = None
        self.is_resizing = False

    def on_resize(self, event):
        if self.is_resizing:
            return

        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)

        self.resize_after_id = self.root.after(200, self._do_resize, event)

    def _do_resize(self, event):
        self.is_resizing = True

        try:
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()

            if self.last_width == window_width and self.last_height == window_height:
                return

            self.last_width = window_width
            self.last_height = window_height

            top_height = int(window_height * 0.15)
            specf_height = int(window_height * 0.65)
            bot_height = int(window_height * 0.20)

            self.top.config(height=top_height)
            self.specf.config(height=specf_height)
            self.bot.config(height=bot_height)

            self.fig.set_size_inches(window_width / 100, specf_height / 100)
            self.canvas.draw()

            self.lb.config(height=int(bot_height / 30))

        finally:
            self.is_resizing = False


    def on_donate(self):
        import webbrowser
        webbrowser.open("https://buymeacoffee.com/vibratoscope")

    def recompute_pitch(self):
        """
        Recompute pitch contour for the currently loaded audio using the selected method.
        """
        if self.audio_data is None:
            messagebox.showerror("Error", "No audio loaded.")
            return

        if not self.file_path or not os.path.isfile(self.file_path):
            messagebox.showerror("Error", "Invalid or missing file path. Please load a valid WAV file.")
            return

        try:
            method = self.pitch_method.get() if hasattr(self, 'pitch_method') else "yin"
            fmin = float(self.fmin.get())
            fmax = float(self.fmax.get())

            print(f"Recomputing pitch with method: {method}")
            print(f"fmin: {fmin}, fmax: {fmax}")
            print(f"audio_data shape: {self.audio_data.shape if self.audio_data is not None else None}")
            print(f"sr: {self.sr}")
            print(f"file_path: {self.file_path}")

            if method.lower() == "praat":
                self.times, self.pitch_hz = extract_pitch_with_praat(self.file_path, fmin=fmin, fmax=fmax)
            elif method.lower() == "yin":
                self.times, self.pitch_hz = extract_pitch_with_yin(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
            elif method.lower() == "hps":
                print("Calling extract_pitch_with_hps...")
                try:
                    self.times, self.pitch_hz = extract_pitch_with_hps(
                        self.audio_data, self.sr, self.file_path, fmin=fmin, fmax=fmax
                    )
                    print("HPS pitch extraction completed.")
                except Exception as hps_error:
                    print(f"HPS error: {str(hps_error)}")
                    raise
            elif method.lower() == "reaper":
                self.times, self.pitch_hz = extract_pitch_with_reaper(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
            elif method.lower() == "sfeeds":
                self.times, self.pitch_hz = extract_pitch_with_sfeeds(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
            else:
                self.times, self.pitch_hz = extract_pitch_with_yin(self.audio_data, self.sr, fmin=fmin, fmax=fmax)

            self.pitch_hz = filter_pitch_outliers(self.pitch_hz, threshold=3)
            cents = convert_to_cents(self.pitch_hz)
            self.cents_centered = remove_mean_or_median(cents, use_median=True)
            self.t_uniform, self.cents_uniform = resample_to_uniform_time(self.times, self.cents_centered, new_sr=100)

            # Update the method used for plotting
            self.last_method_used = method.capitalize()  # Corrects to "Praat", "Yin", "Hps", "Reaper"
            self.actual_pitch_method.set(method)  # for updating the legend dynamically

            self.update_spectrogram_plot()

            messagebox.showinfo("Done", f"Pitch recomputed successfully using {self.last_method_used}.")

        except Exception as e:
            print(f"Recompute pitch error: {str(e)}")
            messagebox.showerror("Error", f"Failed to recompute pitch:\n{str(e)}")

    def update_spectrogram_plot(self):
        """
        Update the spectrogram and pitch plot according to the current selected spectrogram type.
        """
        if self.audio_data is None or self.pitch_hz is None:
            return

        self.ax.clear()

        plot_spectrogram(
            self.ax,
            self.audio_data,
            self.sr,
            self.times,
            self.pitch_hz,
            self.spectrogram_type.get(),  # ✅ Use spectrogram_type properly
            self.file_path,
            self.actual_pitch_method.get()
        )

        self.canvas.draw()


    def populate_audio_devices(self):
        """
        Populate input and output device comboboxes with only active and valid devices.
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            output_devices = []

            for idx, device in enumerate(devices):
                name = device['name']
                hostapi = device.get('hostapi', None)
                max_in = device.get('max_input_channels', 0)
                max_out = device.get('max_output_channels', 0)
                samplerate = device.get('default_samplerate', 0)

                if hostapi is not None and samplerate > 0:
                    if max_in > 0:
                        input_devices.append(f"{idx}: {name}")
                    if max_out > 0:
                        output_devices.append(f"{idx}: {name}")

            # Remove duplicates
            input_devices = list(dict.fromkeys(input_devices))
            output_devices = list(dict.fromkeys(output_devices))

            # Set in combobox
            self.input_combo['values'] = input_devices
            self.output_combo['values'] = output_devices

            # Automatically select the system default devices
            default_input_idx, default_output_idx = sd.default.device

            if input_devices:
                for i, dev in enumerate(input_devices):
                    if dev.startswith(f"{default_input_idx}:"):
                        self.input_combo.current(i)
                        break
                else:
                    self.input_combo.current(0)

            if output_devices:
                for i, dev in enumerate(output_devices):
                    if dev.startswith(f"{default_output_idx}:"):
                        self.output_combo.current(i)
                        break
                else:
                    self.output_combo.current(0)

        except Exception as e:
            print(f"Error loading audio devices: {e}")

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
        self.actual_pitch_method.set("praat")  # Reset actual pitch method
        self.ax.clear()
        self.ax.set_title("Load Audio File - Spectrogram will appear here")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.canvas.draw()
        self.btn_play.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_update_spec.config(state=tk.DISABLED)
        if self.span:
            self.span.disconnect_events()
            self.span = None
        self.root.title("Vibrato Scope - version 1.1.0")

    def on_closing(self):
        """
        Handle window closing to ensure proper cleanup.
        """
        try:
            self.vu_monitoring = False
            if hasattr(self, 'after_ids'):
                for after_id in self.after_ids:
                    self.root.after_cancel(after_id)
                self.after_ids.clear()
            if hasattr(self, 'stream') and self.stream.active:
                self.stream.stop()
                self.stream.close()
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error during closing: {e}")
        sys.exit(0)

    def load_file(self):
        # Ask for WAV file
        fp = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if not fp:
            messagebox.showinfo("Warning", "No file selected.")
            return

        # Validate Fmin/Fmax
        try:
            fmin = float(self.fmin.get())
            fmax = float(self.fmax.get())
            if fmin < 0 or fmin >= fmax:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid Fmin/Fmax. Please enter valid numeric bounds.")
            return

        # Reset state
        self.reset_analysis_state()
        self.file_path = fp
        self.root.title(f"Vibrato Scope - {os.path.basename(fp)}")

        try:
            # Load audio
            self.audio_data, self.sr = load_audio(fp)

            # Extract pitch based on selected method
            self.extract_pitch(fmin, fmax)

            # Ensure pitch is a numeric array and filter outliers
            self.pitch_hz = np.array(self.pitch_hz, dtype=float)
            self.pitch_hz = filter_pitch_outliers(self.pitch_hz, threshold=3)

            # Plot spectrogram with pitch contour
            plot_spectrogram(self.ax, self.audio_data, self.sr, self.times, self.pitch_hz,
                             self.spectrogram_type.get(), fp, self.actual_pitch_method.get())
            self.canvas.draw()

            # ADD THIS:
            self.canvas.toolbar.update()
            self.canvas.toolbar.push_current()

            # Enable controls
            self.btn_play.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            self.btn_update_spec.config(state=tk.NORMAL)

            # Install selection tool
            if self.span:
                self.span.disconnect_events()
            self.span = SpanSelector(self.ax, self.on_select, 'horizontal',
                                     useblit=True,
                                     props=dict(alpha=0.3, facecolor='red'),
                                     interactive=True)

        except Exception as e:
            messagebox.showerror("Error Loading File", str(e))
            self.reset_analysis_state()

    def compare_mean_f0_methods(audio_data, sr, file_path, fmin=50.0, fmax=1500.0):
        """
        Extracts pitch using Praat, YIN, HPS, and REAPER and compares mean F0 for each method.
        """
        results = []

        # Praat
        try:
            times_praat, f0_praat = extract_pitch_with_praat(file_path, fmin=fmin, fmax=fmax)
            mean_f0_praat = np.nanmean(f0_praat)
            results.append(('Praat', mean_f0_praat))
        except Exception as e:
            print(f"Error extracting pitch with Praat: {e}")
            results.append(('Praat', np.nan))

        # YIN
        try:
            times_yin, f0_yin = extract_pitch_with_yin(audio_data, sr, fmin=fmin, fmax=fmax)
            mean_f0_yin = np.nanmean(f0_yin)
            results.append(('YIN', mean_f0_yin))
        except Exception as e:
            print(f"Error extracting pitch with YIN: {e}")
            results.append(('YIN', np.nan))

        # HPS
        try:
            times_hps, f0_hps = extract_pitch_with_hps(audio_data, sr, file_path, fmin=fmin, fmax=fmax)
            mean_f0_hps = np.nanmean(f0_hps)
            results.append(('HPS', mean_f0_hps))
        except Exception as e:
            print(f"Error extracting pitch with HPS: {e}")
            results.append(('HPS', np.nan))

        # REAPER
        try:
            times_reaper, f0_reaper = extract_pitch_with_reaper(audio_data, sr, fmin=fmin, fmax=fmax)
            mean_f0_reaper = np.nanmean(f0_reaper)
            results.append(('REAPER', mean_f0_reaper))
        except Exception as e:
            print(f"Error extracting pitch with REAPER: {e}")
            results.append(('REAPER', np.nan))

        df_compare = pd.DataFrame(results, columns=['Method', 'Mean_F0_Hz'])
        return df_compare

    def extract_pitch_with_fallback(self, wav_path, sr=22050, fmin=50, fmax=1500):
        """
        Attempts Praat pitch extraction, falls back to YIN or HPS if Praat fails.
        Updates self.actual_pitch_method to reflect the method used.
        """
        try:
            times, freqs = extract_pitch_with_praat(wav_path, fmin=float(fmin), fmax=float(fmax))
            self.actual_pitch_method.set("praat")
            print("Pitch extracted with Praat successfully.")
            messagebox.showinfo("Pitch Extraction", "Pitch extracted with Praat successfully.")
            return times, freqs
        except Exception as e:
            print(f"Praat failed: {e}. Falling back to YIN.")
            self.actual_pitch_method.set("yin")
            try:
                y, sr = librosa.load(wav_path, sr=sr)
                f0, _, _ = librosa.pyin(y, sr=sr, fmin=fmin, fmax=fmax)
                times = librosa.times_like(f0, sr=sr)
                print("Pitch extracted with YIN successfully.")
                messagebox.showinfo("Pitch Extraction", "Pitch extracted with YIN successfully.")
                return times, f0
            except Exception as e2:
                print(f"YIN failed: {e2}. Falling back to HPS.")
                self.actual_pitch_method.set("hps")
                f0, times = extract_pitch_with_hps(y, sr, wav_path, fmin=fmin, fmax=fmax)  # Pass wav_path
                print("Pitch extracted with HPS successfully.")
                messagebox.showinfo("Pitch Extraction", "Pitch extracted with HPS successfully.")
                return times, f0

    def extract_pitch(self, fmin, fmax):
        """
        Extracts pitch based on the selected pitch method and updates self.actual_pitch_method.
        """
        if self.pitch_method.get() == "praat":
            self.times, self.pitch_hz = self.extract_pitch_with_fallback(
                wav_path=self.file_path, sr=self.sr, fmin=fmin, fmax=fmax
            )
        elif self.pitch_method.get() == "yin":
            self.actual_pitch_method.set("yin")
            self.times, self.pitch_hz = extract_pitch_with_yin(
                self.audio_data, self.sr, fmin=fmin, fmax=fmax, file_path=self.file_path
            )
            print("Pitch extracted with YIN successfully.")
            messagebox.showinfo("Pitch Extraction", "Pitch extracted with YIN successfully.")
        elif self.pitch_method.get() == "reaper":
            self.actual_pitch_method.set("reaper")
            self.times, self.pitch_hz = extract_pitch_with_reaper(
                self.audio_data, self.sr, fmin=fmin, fmax=fmax, file_path=self.file_path
            )
            print("Pitch extracted with REAPER successfully.")
            messagebox.showinfo("Pitch Extraction", "Pitch extracted with REAPER successfully.")
        elif self.pitch_method.get() == "sfeeds":
            self.actual_pitch_method.set("sfeeds")
            from pitch import extract_pitch_with_sfeeds
            self.times, self.pitch_hz = extract_pitch_with_sfeeds(
                self.audio_data, self.sr, fmin=fmin, fmax=fmax, file_path=self.file_path
            )
            print("Pitch extracted with SFEEDS successfully.")
            messagebox.showinfo("Pitch Extraction", "Pitch extracted with SFEEDS successfully.")
        else:  # HPS case
            self.actual_pitch_method.set("hps")
            self.times, self.pitch_hz = extract_pitch_with_hps(
                self.audio_data, self.sr, self.file_path, fmin=fmin, fmax=fmax  # Pass self.file_path
            )
            print("Pitch extracted with HPS successfully.")
            messagebox.showinfo("Pitch Extraction", "Pitch extracted with HPS successfully.")
        print(f"Pitch extracted using {self.actual_pitch_method.get().upper()}")

    def on_pitch_method_change(self, *args):
        """
        Updates spectrogram and pitch data when pitch method changes.
        """
        if self.audio_data is not None and self.file_path != "":
            print(f"Pitch method changed to {self.pitch_method.get()}")
            self.update_spectrogram()

    def update_spectrogram(self):
        """
        Updates the spectrogram and pitch contour based on current settings.
        """
        if self.audio_data is None or self.sr is None or self.file_path == "":
            messagebox.showwarning("Warning", "No audio data loaded. Please load a WAV file first.")
            return
        try:
            # Get Fmin/Fmax
            fmin = float(self.fmin.get())
            fmax = float(self.fmax.get())
            if fmin < 0 or fmin >= fmax:
                raise ValueError("Invalid Fmin/Fmax")

            # Re-extract pitch based on current method
            self.extract_pitch(fmin, fmax)

            # Filter outliers
            self.pitch_hz = np.array(self.pitch_hz, dtype=float)
            self.pitch_hz = filter_pitch_outliers(self.pitch_hz, threshold=3)

            # Plot spectrogram with updated pitch contour
            plot_spectrogram(self.ax, self.audio_data, self.sr, self.times, self.pitch_hz,
                             self.spectrogram_type.get(), self.file_path, self.actual_pitch_method.get())
            self.canvas.draw()

            # ADD THIS:
            self.canvas.toolbar.update()
            self.canvas.toolbar.push_current()

        except ValueError as ve:
            messagebox.showerror("Error", "Invalid Fmin/Fmax. Please enter valid numeric bounds.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update spectrogram: {str(e)}")

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

    def open_audio_recorder(self):
        AudioRecorder(self.root, on_save=self.load_audio_after_recording)

    def load_audio_after_recording(self, filepath):
        """
        Load audio after recording and update GUI.
        """
        try:
            if not filepath or not os.path.exists(filepath):
                messagebox.showerror("Error", "Invalid file path.")
                return

            # Clear previous region selections, listbox, spectrogram, and SpanSelector
            self.selected_regions = []
            self.lb.delete(0, tk.END)
            if self.span:
                self.span.disconnect_events()
                self.span = None
            self.ax.clear()
            self.ax.set_title("Loading Recorded Audio")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Frequency (Hz)")
            self.canvas.draw()

            self.file_path = filepath
            self.audio_data, self.sr = load_audio(filepath)

            fmin = float(self.fmin.get())
            fmax = float(self.fmax.get())

            method = self.pitch_method.get()
            if method.lower() == "praat":
                self.times, self.pitch_hz = extract_pitch_with_praat(self.file_path, fmin=fmin, fmax=fmax)
            elif method.lower() == "yin":
                self.times, self.pitch_hz = extract_pitch_with_yin(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
            elif method.lower() == "hps":
                self.times, self.pitch_hz = extract_pitch_with_hps(self.audio_data, self.sr, self.file_path, fmin=fmin,
                                                                   fmax=fmax)
            elif method.lower() == "reaper":
                self.times, self.pitch_hz = extract_pitch_with_reaper(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
            elif method.lower() == "sfeeds":
                self.times, self.pitch_hz = extract_pitch_with_sfeeds(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
            else:
                self.times, self.pitch_hz = extract_pitch_with_yin(self.audio_data, self.sr, fmin=fmin, fmax=fmax)

            # Apply outlier filtering
            self.pitch_hz = filter_pitch_outliers(self.pitch_hz, threshold=3)

            # Plot updated spectrogram
            plot_spectrogram(self.ax, self.audio_data, self.sr, self.times, self.pitch_hz,
                             self.spectrogram_type.get(), self.file_path, self.actual_pitch_method.get())
            self.canvas.draw()

            self.btn_play.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            self.btn_update_spec.config(state=tk.NORMAL)
            self.last_method_used = method.capitalize()

            if self.span:
                self.span.disconnect_events()
            self.span = SpanSelector(self.ax, self.on_select, 'horizontal',
                                     useblit=True,
                                     props=dict(alpha=0.3, facecolor='red'),
                                     interactive=True)

            self.root.title(f"Vibrato Scope - {os.path.basename(self.file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load recorded audio: {str(e)}")

    def play_audio(self):
        """
        Play the loaded audio using the selected output device.
        """
        if self.audio_data is not None:
            try:
                selected_output = self.output_device_var.get()
                if selected_output:
                    device_id = int(selected_output.split(":")[0])
                    sd.play(self.audio_data, self.sr, device=device_id)
                else:
                    sd.play(self.audio_data, self.sr)  # fallback: use default
            except Exception as e:
                messagebox.showerror("Error", f"Failed to play audio:\n{str(e)}")

    def run_analysis(self):
        """
        Runs vibrato analysis, ensuring pitch data matches the selected pitch method.
        """
        self.generated_figures = []  # Reset figures
        # Reset alternative analysis flags
        self.alternative_used = False
        self.harmonic_used = None
        self.harmonic_frequency = None

        if not self.selected_regions:
            messagebox.showerror("Error", "No regions selected!")
            return

        if self.audio_data is None or self.file_path == "":
            messagebox.showerror("Error", "No audio data loaded!")
            return

        # Prompt user for output directory
        from pathlib import Path
        import os

        default_dir = Path.home() / "Documents" / "VibratoScope"
        default_dir.mkdir(parents=True, exist_ok=True)

        save_dir = filedialog.askdirectory(title="Select folder to save analysis results", initialdir=str(default_dir))
        if not save_dir or not os.access(save_dir, os.W_OK):
            messagebox.showerror("Error", "Invalid or unwritable directory selected!")
            return

        # Get Fmin/Fmax
        try:
            fmin = float(self.fmin.get())
            fmax = float(self.fmax.get())
            if fmin < 0 or fmin >= fmax:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid Fmin/Fmax. Please enter valid numeric bounds.")
            return

        # Explicitly re-extract pitch to ensure it matches the current method
        try:
            self.extract_pitch(fmin, fmax)
        except Exception as e:
            messagebox.showerror("Error", f"Pitch extraction failed: {str(e)}")
            return

        # Process pitch data
        self.pitch_hz = np.array(self.pitch_hz, dtype=float)
        self.pitch_hz = filter_pitch_outliers(self.pitch_hz, threshold=5)
        valid_mask = ~np.isnan(self.pitch_hz) & (self.pitch_hz > 0)

        if not np.any(valid_mask):
            messagebox.showerror("Error",
                                 "No valid pitch detected after outlier filtering. Try adjusting Fmin/Fmax or using a different pitch method.")
            return

        cents = convert_to_cents(self.pitch_hz)
        self.t_uniform, self.cents_uniform = resample_to_uniform_time(
            self.times, remove_mean_or_median(cents, use_median=True), new_sr=100
        )
        if self.t_uniform is None or len(self.t_uniform) < 2:
            messagebox.showerror("Error", "Resampling to uniform time grid failed.")
            return

        # Initialize lists for collecting cycle parameters and region data
        all_cycle_params = []
        region_list = []

        # Apply bandpass filter and detect vibrato cycles (global)
        cents_filtered = apply_bandpass_filter(self.cents_uniform, fs=100, lowcut=3.0, highcut=9.0, order=4)
        peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
            self.t_uniform, self.cents_uniform, cents_filtered, prominence=5, distance=5
        )

        # Apply bandpass filter and detect vibrato cycles (global)
        cents_filtered = apply_bandpass_filter(self.cents_uniform, fs=100, lowcut=3.0, highcut=9.0, order=4)

        # ==== NEW: Save vibrato component CSV ====
        try:
            import pandas as pd
            vib_df = pd.DataFrame({
                "time_s": self.t_uniform,
                "centered_cents": self.cents_uniform,
                "vibrato_component_cents": cents_filtered
            })
            base = os.path.splitext(os.path.basename(self.file_path))[0]
            vib_df.to_csv(
                os.path.join(save_dir, f"{base}_vibrato_component.csv"),
                index=False
            )
        except Exception as e:
            print(f"[WARN] Could not save vibrato_component.csv: {e}")
        # ==== END NEW ====

        # Plot peaks and troughs (filtered signal, global, Figure_2)
        save_path_global = os.path.join(save_dir,
                                        f"{os.path.splitext(os.path.basename(self.file_path))[0]}_Figure_2_global.png")
        fig_p = plot_peaks_troughs(t_valid, cents_filtered, peaks, troughs, save_path=save_path_global)
        if fig_p:
            self.generated_figures.append(fig_p)
        else:
            print("Warning: Failed to generate global Figure_2 plot.")

        # Figure 1: Global analysis (using original plot_before_after_filter)
        save_path_fig1 = os.path.join(save_dir, "Figure_1.png")
        fig1 = plot_before_after_filter(self.t_uniform, self.cents_uniform, cents_filtered, save_path=save_path_fig1)
        self.generated_figures.append(fig1)

        # Process each selected region
        for reg_idx, (t_start, t_end) in enumerate(self.selected_regions):
            mask = (self.t_uniform >= t_start) & (self.t_uniform <= t_end)
            t_region = self.t_uniform[mask]
            if len(t_region) == 0:
                print(f"Warning: Region {reg_idx + 1} is empty. Skipping.")
                continue
            cents_region = self.cents_uniform[mask]
            cents_filtered_region = apply_bandpass_filter(cents_region, fs=100, lowcut=3.0, highcut=9.0, order=4)
            peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                t_region, cents_region, cents_filtered_region, prominence=5, distance=5
            )
            # Plot peaks and troughs (filtered signal, region-specific, Figure_2)
            save_path_region = os.path.join(save_dir,
                                            f"{os.path.splitext(os.path.basename(self.file_path))[0]}_Figure_2_region_{reg_idx + 1}.png")
            fig_p = plot_peaks_troughs(t_valid, cents_filtered_region, peaks, troughs, save_path=save_path_region)
            if fig_p:
                self.generated_figures.append(fig_p)
            else:
                print(
                    f"Warning: Failed to generate Figure_2_region_{reg_idx + 1} for {os.path.basename(self.file_path)}")

            if len(all_idx) < 2:
                print(f"Warning: Insufficient vibrato cycles in region {reg_idx + 1}. Skipping.")
                continue
            # Calculate region_avg_F0 before using it
            reg_mask = (self.times >= t_start) & (self.times <= t_end)
            region_pitch = self.pitch_hz[reg_mask]
            region_avg_F0 = np.nanmean(region_pitch) if np.any(~np.isnan(region_pitch)) else np.nan
            cycle_params = compute_cycle_parameters(t_valid, cents_valid, half_cycle_extents, all_idx)
            filtered_params = filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0)
            if not filtered_params:
                print(f"Warning: No valid vibrato cycles after filtering in region {reg_idx + 1}. Skipping.")
                continue
            df_file = pd.DataFrame(filtered_params)
            df_file['File_Name'] = os.path.basename(self.file_path)
            num_cycles = len(df_file)
            if num_cycles < 5:
                print(f"Warning: Too few cycles ({num_cycles}) in region {reg_idx + 1}. Skipping.")
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

        # Alternative harmonic-based analysis
        if not all_cycle_params:
            response = messagebox.askyesno(
                "Alternative Analysis",
                "No valid vibrato cycles were found using the fundamental-based analysis.\n"
                "Would you like to try analysis using the most intense harmonic?"
            )
            if response:
                valid_f0 = self.pitch_hz[~np.isnan(self.pitch_hz) & (self.pitch_hz > 0)]
                if len(valid_f0) == 0:
                    messagebox.showerror("Error", "No valid fundamental (f0) to estimate harmonic.")
                    return
                f0_est = np.median(valid_f0)
                t_h, harmonic_pitch, harmonic_num = extract_harmonic_pitch(self.audio_data, self.sr, f0_est)
                if t_h is None or harmonic_pitch is None or np.all(np.isnan(harmonic_pitch)):
                    messagebox.showerror("Error", "Failed to extract valid harmonic pitch contour.")
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
                for reg_idx, (t_start_alt, t_end_alt) in enumerate(self.selected_regions):
                    region_mask = (new_t_uniform >= t_start_alt) & (new_t_uniform <= t_end_alt)
                    t_region_alt = new_t_uniform[region_mask]
                    c_region_alt = new_cents_uniform[region_mask]
                    if len(t_region_alt) < 2 or np.all(np.isnan(c_region_alt)):
                        continue
                    c_filtered_alt = apply_bandpass_filter(c_region_alt, fs=100, lowcut=3.0, highcut=9.0, order=4)
                    peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                        t_region_alt, c_region_alt, c_filtered_alt, prominence=5, distance=5
                    )
                    if len(t_valid) < 2 or len(all_idx) < 2:
                        continue
                    cycle_params = compute_cycle_parameters(t_valid, cents_valid, half_cycle_extents, all_idx)
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
                                                       save_path=os.path.join(save_dir,
                                                                              f"Figure_1_alt_region_{reg_idx + 1}.png"))
                    self.generated_figures.append(fig_alt)
                    fig2_alt = plot_peaks_troughs(t_valid, c_filtered_alt, peaks, troughs,
                                                  save_path=os.path.join(save_dir,
                                                                         f"Figure_2_alt_region_{reg_idx + 1}.png"))
                    self.generated_figures.append(fig2_alt)

                if not alt_cycle_params:
                    messagebox.showinfo("Analysis", "No valid vibrato cycles were found using the harmonic method.")
                    return
                all_cycle_params = alt_cycle_params
                region_list = alt_region_list

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
        # === Export smoothed (rolling mean/std) vibrato parameters as CSV ===
        try:
            smoothed_csv_path = os.path.join(save_dir,
                                             f"{os.path.splitext(os.path.basename(self.file_path))[0]}_detailed_vibrato_data_smoothed.csv")
            df_smoothed.to_csv(smoothed_csv_path)
            print(f"[INFO] Smoothed vibrato data saved: {smoothed_csv_path}")
        except Exception as e:
            print(f"[WARN] Could not save smoothed vibrato CSV: {e}")
        smoothed_cycle_times = df_smoothed[('cycle_time', 'mean')].dropna().values
        smoothed_cycle_extents = df_smoothed[('half_extent_cents', 'mean')].dropna().values
        global_stats_smooth = analyze_vibrato(smoothed_cycle_times, smoothed_cycle_extents)
        cv_rate_smooth, cv_extent_smooth = compute_cv(smoothed_cycle_times, smoothed_cycle_extents)

        # SampEn Target Selection and Computation
        target = self.sampen_target.get()
        dist = self.sampen_distance.get()

        def safe_sampen(series, label):
            if len(series) >= 5 and np.std(series) > 0:
                norm = (series - np.mean(series)) / np.std(series)
                m = int(float(self.sampen_m.get()))  # Convert to int
                r = float(self.sampen_r.get())  # Convert to float
                value = sample_entropy(norm, m=m, r=r, distance=self.sampen_distance.get())
                print(f"[DEBUG SampEn] Target: {label} | SampEn = {value:.6f}")
                return value
            else:
                print(f"[DEBUG SampEn] Target: {label} | Series too short or flat.")
                return np.nan

        sampen_rate = safe_sampen(df_detailed['VibratoRate'].dropna().values, "rate")
        sampen_extent = safe_sampen(df_detailed['half_extent_cents'].dropna().values, "extent")
        sampen_cycle_time = safe_sampen(df_detailed['cycle_time'].dropna().values, "cycle_time")

        # Valor para o gráfico/tabela principal
        if target == "rate":
            sampen_display = sampen_rate
        elif target == "extent":
            sampen_display = sampen_extent
        elif target == "cycle_time":
            sampen_display = sampen_cycle_time
        else:
            sampen_display = np.nan

        if self.alternative_used:
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
                'mean_pitch_hz': np.nanmean(self.pitch_hz),
                'min_pitch_hz': np.nanmin(self.pitch_hz),
                'max_pitch_hz': np.nanmax(self.pitch_hz),
                'HarmonicUsed': f"Harmonic {self.harmonic_used}",
                'HarmonicFrequency': self.harmonic_frequency,
                'SampEn_Rate': sampen_final if target == "rate" else np.nan,
                'SampEn_Extent': sampen_final if target == "extent" else np.nan,
                'SampEn_CycleTime': sampen_final if target == "cycle_time" else np.nan,
                'SampEn_Target': target,
                'SampEn_Distance': dist,
                'PitchMethod': self.actual_pitch_method.get()
            }
        else:
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
                'mean_pitch_hz': np.nanmean(self.pitch_hz),
                'min_pitch_hz': np.nanmin(self.pitch_hz),
                'max_pitch_hz': np.nanmax(self.pitch_hz),
                'HarmonicUsed': "Fundamental",
                'HarmonicFrequency': np.nanmedian(self.pitch_hz),
                'SampEn_Rate': sampen_rate,
                'SampEn_Extent': sampen_extent,
                'SampEn_CycleTime': sampen_cycle_time,
                'SampEn_Target': target,
                'SampEn_Distance': dist,
                'PitchMethod': self.actual_pitch_method.get()
            }

        self.summary_data = summary_data_file
        self.df_detailed = df_detailed
        self.df_region = pd.DataFrame(region_list)
        new_avg_dict = summary_data_file.copy()
        new_avg_dict['File_Name'] = os.path.basename(self.file_path)
        self.df_avg = pd.DataFrame([new_avg_dict])

        for reg_item in region_list:
            reg_item.update(self.summary_data)

        fig3 = plot_vibrato_rate(df_detailed, df_smoothed, save_path=os.path.join(save_dir, "Figure_3.png"))
        self.generated_figures.append(fig3)

        print(f"[FINAL SampEn] SampEn Rate: {sampen_rate}, Extent: {sampen_extent}, Cycle Time: {sampen_cycle_time}")

        fig_summary = final_plot(
            df_detailed, df_smoothed, self.summary_data,
            jitter_metrics=compute_jitter_metrics(df_detailed['cycle_time'].values * 2),
            cv_rate=cv_rate, cv_extent=cv_extent,
            cv_rate_smooth=cv_rate_smooth, cv_extent_smooth=cv_extent_smooth,
            filename=self.file_path,
            sampen_rate=sampen_rate,
            sampen_extent=sampen_extent,
            sampen_cycle_time=sampen_cycle_time,
            sampen_target=target,
            sampen_distance=dist,
            title="Vibrato Scope: Final Analysis",
            show_figure=False
        )

        fig_summary_path = os.path.join(save_dir, "final_analysis.png")
        fig_summary.savefig(fig_summary_path, dpi=300)
        plt.close(fig_summary)
        self.generated_figures.append(fig_summary_path)
        self.last_fig = fig_summary

        self.save_results()

    def batch_process(self):
        """
        Performs vibrato analysis on multiple WAV files, respecting the selected pitch method.
        Saves results and figures in the selected directory.
        """
        # Select input files
        file_paths = filedialog.askopenfilenames(title="Select WAV Files for Batch Processing",
                                                 filetypes=[("WAV Files", "*.wav")])
        if not file_paths:
            messagebox.showinfo("Batch Processing", "No files selected.")
            return

        # Select output directory
        default_dir = Path.home() / "Documents" / "VibratoScope"
        default_dir.mkdir(parents=True, exist_ok=True)

        save_dir = filedialog.askdirectory(title="Select folder to save batch results", initialdir=str(default_dir))
        if not save_dir or not os.access(save_dir, os.W_OK):
            messagebox.showerror("Batch Processing", "Invalid or unwritable directory selected!")
            return

        # Initialize lists for aggregated results
        batch_avg_list = []
        batch_detailed_list = []
        batch_region_list = []
        self.generated_figures = []

        # Get Fmin/Fmax from GUI inputs
        try:
            fmin = float(self.fmin.get())
            fmax = float(self.fmax.get())
            if fmin < 0 or fmin >= fmax:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid Fmin/Fmax. Please enter valid numeric bounds.")
            return

        target = self.sampen_target.get()
        dist = self.sampen_distance.get()
        selected_method = self.pitch_method.get()  # Get the user-selected pitch method

        for file_path in file_paths:
            try:
                # Reset alternative analysis flags
                self.alternative_used = False
                self.harmonic_used = None
                self.harmonic_frequency = None
                self.file_path = file_path  # Set file_path for consistency
                self.generated_figures = []  # Reset figures for this file

                # Load audio
                self.audio_data, self.sr = load_audio(file_path)
                print(f"Batch: Loaded audio for {os.path.basename(file_path)}")

                # Extract pitch based on the selected method
                if selected_method.lower() == "praat":
                    try:
                        self.times, self.pitch_hz = extract_pitch_with_praat(file_path, fmin=fmin, fmax=fmax)
                        self.actual_pitch_method.set("praat")
                        print(f"Batch: Pitch extracted with Praat for {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Batch: Praat failed for {os.path.basename(file_path)}: {e}. Falling back to YIN.")
                        try:
                            f0, _, _ = librosa.pyin(self.audio_data, sr=self.sr, fmin=fmin, fmax=fmax)
                            self.times = librosa.times_like(f0, sr=self.sr)
                            self.pitch_hz = f0
                            self.actual_pitch_method.set("yin")
                            print(f"Batch: Pitch extracted with YIN for {os.path.basename(file_path)}")
                        except Exception as e2:
                            print(f"Batch: YIN failed: {e2}. Falling back to REAPER.")
                            try:
                                self.times, self.pitch_hz = extract_pitch_with_reaper(self.audio_data, self.sr,
                                                                                      fmin=fmin,
                                                                                      fmax=fmax, file_path=file_path)
                                self.actual_pitch_method.set("reaper")
                                print(f"Batch: Pitch extracted with REAPER for {os.path.basename(file_path)}")
                            except Exception as e3:
                                print(f"Batch: REAPER failed: {e3}. Falling back to HPS.")
                                self.times, self.pitch_hz = extract_pitch_with_hps(self.audio_data, self.sr, file_path,
                                                                                   fmin=fmin, fmax=fmax)
                                self.actual_pitch_method.set("hps")
                                print(f"Batch: Pitch extracted with HPS for {os.path.basename(file_path)}")
                elif selected_method.lower() == "yin":
                    self.times, self.pitch_hz = extract_pitch_with_yin(self.audio_data, self.sr, fmin=fmin, fmax=fmax)
                    self.actual_pitch_method.set("yin")
                    print(f"Batch: Pitch extracted with YIN for {os.path.basename(file_path)}")
                elif selected_method.lower() == "reaper":
                    self.times, self.pitch_hz = extract_pitch_with_reaper(self.audio_data, self.sr, fmin=fmin,
                                                                          fmax=fmax,
                                                                          file_path=file_path)
                    self.actual_pitch_method.set("reaper")
                    print(f"Batch: Pitch extracted with REAPER for {os.path.basename(file_path)}")
                elif selected_method.lower() == "hps":
                    self.times, self.pitch_hz = extract_pitch_with_hps(self.audio_data, self.sr, file_path,
                                                                       fmin=fmin, fmax=fmax)
                    self.actual_pitch_method.set("hps")
                    print(f"Batch: Pitch extracted with HPS for {os.path.basename(file_path)}")

                # Validate pitch data
                if self.pitch_hz is None or np.all(np.isnan(self.pitch_hz)):
                    print(f"Batch: No valid pitch data for {os.path.basename(file_path)}. Skipping.")
                    continue

                # Process pitch data
                self.pitch_hz = np.array(self.pitch_hz, dtype=float)
                valid_points_before = np.sum(~np.isnan(self.pitch_hz))
                self.pitch_hz = filter_pitch_outliers(self.pitch_hz, threshold=5)
                valid_points_after = np.sum(~np.isnan(self.pitch_hz))
                print(
                    f"Batch: Pitch filtering for {os.path.basename(file_path)}: {valid_points_before} valid points before, {valid_points_after} after (threshold=5)")
                valid_mask = ~np.isnan(self.pitch_hz) & (self.pitch_hz > 0)

                if not np.any(valid_mask):
                    print(
                        f"Batch: No valid pitch detected after filtering for {os.path.basename(file_path)}. Skipping.")
                    continue

                cents = convert_to_cents(self.pitch_hz)
                self.t_uniform, self.cents_uniform = resample_to_uniform_time(
                    self.times, remove_mean_or_median(cents, use_median=True), new_sr=100
                )
                if self.t_uniform is None or len(self.t_uniform) < 2:
                    print(f"Batch: Resampling failed for {os.path.basename(file_path)}. Skipping.")
                    continue

                # Initialize lists for cycle parameters and region data
                all_cycle_params = []
                region_list = []

                # Apply bandpass filter and detect vibrato cycles (global)
                cents_filtered = apply_bandpass_filter(self.cents_uniform, fs=100, lowcut=3.0, highcut=9.0, order=4)
                peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                    self.t_uniform, self.cents_uniform, cents_filtered, prominence=5, distance=5
                )

                # Plot global Figure_2 (peaks and troughs)
                save_path_global = os.path.join(save_dir,
                                                f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_2_global.png")
                fig_p = plot_peaks_troughs(t_valid, cents_filtered, peaks, troughs, save_path=save_path_global)
                if fig_p:
                    self.generated_figures.append(fig_p)
                else:
                    print(f"Batch: Warning: Failed to generate global Figure_2 for {os.path.basename(file_path)}")

                # Plot Figure_1 (before and after filter)
                save_path_fig1 = os.path.join(save_dir,
                                              f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_1.png")
                fig1 = plot_before_after_filter(self.t_uniform, self.cents_uniform, cents_filtered,
                                                save_path=save_path_fig1)
                self.generated_figures.append(fig1)

                # Define regions (use full duration or selected regions)
                if not self.selected_regions:
                    # Use full duration as a single region
                    t_start = self.t_uniform[0]
                    t_end = self.t_uniform[-1]
                    self.selected_regions = [(t_start, t_end)]
                else:
                    # Use user-defined regions (ensure they are valid)
                    self.selected_regions = [(max(t_s, self.t_uniform[0]), min(t_e, self.t_uniform[-1]))
                                             for t_s, t_e in self.selected_regions]

                # Process each region
                for reg_idx, (t_start, t_end) in enumerate(self.selected_regions):
                    mask = (self.t_uniform >= t_start) & (self.t_uniform <= t_end)
                    t_region = self.t_uniform[mask]
                    if len(t_region) == 0:
                        print(
                            f"Batch: Warning: Region {reg_idx + 1} is empty for {os.path.basename(file_path)}. Skipping.")
                        continue
                    cents_region = self.cents_uniform[mask]
                    cents_filtered_region = apply_bandpass_filter(cents_region, fs=100, lowcut=3.0, highcut=9.0,
                                                                  order=4)
                    peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                        t_region, cents_region, cents_filtered_region, prominence=5, distance=5
                    )

                    # Plot region-specific Figure_2
                    save_path_region = os.path.join(save_dir,
                                                    f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_2_region_{reg_idx + 1}.png")
                    fig_p = plot_peaks_troughs(t_valid, cents_filtered_region, peaks, troughs,
                                               save_path=save_path_region)
                    if fig_p:
                        self.generated_figures.append(fig_p)
                    else:
                        print(
                            f"Batch: Warning: Failed to generate Figure_2_region_{reg_idx + 1} for {os.path.basename(file_path)}")

                    if len(all_idx) < 2:
                        print(
                            f"Batch: Warning: Insufficient vibrato cycles in region {reg_idx + 1} for {os.path.basename(file_path)}. Skipping.")
                        continue

                    # Calculate region_avg_F0
                    reg_mask = (self.times >= t_start) & (self.times <= t_end)
                    region_pitch = self.pitch_hz[reg_mask]
                    region_avg_F0 = np.nanmean(region_pitch) if np.any(~np.isnan(region_pitch)) else np.nan

                    cycle_params = compute_cycle_parameters(t_valid, cents_valid, half_cycle_extents, all_idx)
                    filtered_params = filter_vibrato_cycles(cycle_params, min_half_extent=10.0, max_half_extent=300.0)
                    if not filtered_params:
                        print(
                            f"Batch: Warning: No valid vibrato cycles after filtering in region {reg_idx + 1} for {os.path.basename(file_path)}. Skipping.")
                        continue

                    # Create DataFrame and validate
                    df_file = pd.DataFrame(filtered_params)
                    df_file['File_Name'] = os.path.basename(file_path)
                    num_cycles = len(df_file)
                    if num_cycles < 5:
                        print(
                            f"Batch: Warning: Too few cycles ({num_cycles}) in region {reg_idx + 1} for {os.path.basename(file_path)}. Skipping.")
                        continue
                    if df_file['cycle_time'].isna().any() or df_file['cycle_time'].isnull().any():
                        print(
                            f"Batch: Warning: Invalid cycle_time values in region {reg_idx + 1} for {os.path.basename(file_path)}. Skipping.")
                        continue
                    if not np.all(df_file['cycle_time'] > 0):
                        print(
                            f"Batch: Warning: Non-positive cycle_time values in region {reg_idx + 1} for {os.path.basename(file_path)}. Skipping.")
                        continue

                    # Calculate region metrics
                    region_rate = 1 / (2 * df_file['cycle_time'])
                    region_avg_rate = region_rate.mean()
                    region_std_rate = region_rate.std()
                    region_median_rate = np.median(region_rate)
                    region_avg_extent = df_file['half_extent_cents'].mean()
                    region_std_extent = df_file['half_extent_cents'].std()
                    region_median_extent = np.median(df_file['half_extent_cents'])
                    region_jitter = df_file['cycle_time'].std() / df_file['cycle_time'].mean()
                    region_cv_rate = (region_rate.std() / region_rate.mean()) * 100 if region_avg_rate != 0 else np.nan
                    region_cv_extent = (df_file['half_extent_cents'].std() / df_file[
                        'half_extent_cents'].mean()) * 100 if region_avg_extent != 0 else np.nan

                    # Add region metadata to cycle parameters
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
                        'File_Name': os.path.basename(file_path),
                        'PitchMethod': self.actual_pitch_method.get()
                    }
                    region_list.append(reg_dict)

                # Alternative harmonic-based analysis if no valid cycles
                if not all_cycle_params:
                    print(
                        f"Batch: No valid vibrato cycles found for {os.path.basename(file_path)}. Trying harmonic-based analysis.")
                    valid_f0 = self.pitch_hz[~np.isnan(self.pitch_hz) & (self.pitch_hz > 0)]
                    if len(valid_f0) == 0:
                        print(
                            f"Batch: No valid fundamental (f0) to estimate harmonic for {os.path.basename(file_path)}. Skipping.")
                        continue
                    f0_est = np.median(valid_f0)
                    t_h, harmonic_pitch, harmonic_num = extract_harmonic_pitch(self.audio_data, self.sr, f0_est)
                    if t_h is None or harmonic_pitch is None or np.all(np.isnan(harmonic_pitch)):
                        print(
                            f"Batch: Failed to extract valid harmonic pitch contour for {os.path.basename(file_path)}. Skipping.")
                        continue
                    self.alternative_used = True
                    self.harmonic_used = harmonic_num
                    self.harmonic_frequency = np.nanmedian(harmonic_pitch)
                    cents_h = convert_to_cents(harmonic_pitch)
                    cents_h_centered = remove_mean_or_median(cents_h, use_median=True)
                    cents_adjusted = cents_h_centered / harmonic_num
                    new_t_uniform, new_cents_uniform = resample_to_uniform_time(t_h, cents_adjusted, new_sr=100)
                    if new_t_uniform is None:
                        print(
                            f"Batch: Resampling failed in alternative method for {os.path.basename(file_path)}. Skipping.")
                        continue

                    alt_cycle_params = []
                    alt_region_list = []
                    for reg_idx, (t_start_alt, t_end_alt) in enumerate(self.selected_regions):
                        region_mask = (new_t_uniform >= t_start_alt) & (new_t_uniform <= t_end_alt)
                        t_region_alt = new_t_uniform[region_mask]
                        c_region_alt = new_cents_uniform[region_mask]
                        if len(t_region_alt) < 2 or np.all(np.isnan(c_region_alt)):
                            print(
                                f"Batch: Warning: Region {reg_idx + 1} empty or invalid in alternative analysis for {os.path.basename(file_path)}. Skipping.")
                            continue
                        c_filtered_alt = apply_bandpass_filter(c_region_alt, fs=100, lowcut=3.0, highcut=9.0, order=4)
                        peaks, troughs, half_cycle_times, half_cycle_extents, t_valid, cents_valid, all_idx = detect_vibrato_cycles(
                            t_region_alt, c_region_alt, c_filtered_alt, prominence=5, distance=5
                        )
                        if len(t_valid) < 2 or len(all_idx) < 2:
                            print(
                                f"Batch: Warning: Insufficient vibrato cycles in region {reg_idx + 1} (alternative) for {os.path.basename(file_path)}. Skipping.")
                            continue
                        cycle_params = compute_cycle_parameters(t_valid, cents_valid, half_cycle_extents, all_idx)
                        filtered_params = filter_vibrato_cycles(cycle_params, min_half_extent=10.0,
                                                                max_half_extent=300.0)
                        if not filtered_params:
                            print(
                                f"Batch: Warning: No valid vibrato cycles after filtering in region {reg_idx + 1} (alternative) for {os.path.basename(file_path)}. Skipping.")
                            continue
                        df_file = pd.DataFrame(filtered_params)
                        df_file['File_Name'] = os.path.basename(file_path)
                        num_cycles = len(df_file)
                        if num_cycles < 5:
                            print(
                                f"Batch: Warning: Too few cycles ({num_cycles}) in region {reg_idx + 1} (alternative) for {os.path.basename(file_path)}. Skipping.")
                            continue
                        if df_file['cycle_time'].isna().any() or df_file['cycle_time'].isnull().any():
                            print(
                                f"Batch: Warning: Invalid cycle_time values in region {reg_idx + 1} (alternative) for {os.path.basename(file_path)}. Skipping.")
                            continue
                        if not np.all(df_file['cycle_time'] > 0):
                            print(
                                f"Batch: Warning: Non-positive cycle_time values in region {reg_idx + 1} (alternative) for {os.path.basename(file_path)}. Skipping.")
                            continue

                        region_rate = 1 / (2 * df_file['cycle_time'])
                        region_avg_rate = region_rate.mean()
                        region_std_rate = region_rate.std()
                        region_median_rate = np.median(region_rate)
                        region_avg_extent = df_file['half_extent_cents'].mean()
                        region_std_extent = df_file['half_extent_cents'].std()
                        region_median_extent = np.median(df_file['half_extent_cents'])
                        region_jitter = df_file['cycle_time'].std() / df_file['cycle_time'].mean()
                        region_cv_rate = (
                                                 region_rate.std() / region_rate.mean()) * 100 if region_avg_rate != 0 else np.nan
                        region_cv_extent = (df_file['half_extent_cents'].std() / df_file[
                            'half_extent_cents'].mean()) * 100 if region_avg_extent != 0 else np.nan

                        for cp in filtered_params:
                            cp['Region_Start'] = t_start_alt
                            cp['Region_End'] = t_end_alt
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
                            cp['Analysis'] = f"Harmonic {harmonic_num}"

                        alt_cycle_params.extend(filtered_params)
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
                            'Avg_F0_Hz': np.nanmean(harmonic_pitch) if np.any(~np.isnan(harmonic_pitch)) else np.nan,
                            'File_Name': os.path.basename(file_path),
                            'PitchMethod': self.actual_pitch_method.get(),
                            'Analysis': f"Harmonic {harmonic_num}"
                        }
                        alt_region_list.append(alt_region_item)

                        # Plot alternative figures
                        fig_alt = plot_before_after_filter(t_region_alt, c_region_alt, c_filtered_alt,
                                                           save_path=os.path.join(save_dir,
                                                                                  f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_1_alt_region_{reg_idx + 1}.png"))
                        self.generated_figures.append(fig_alt)
                        fig2_alt = plot_peaks_troughs(t_valid, c_filtered_alt, peaks, troughs,
                                                      save_path=os.path.join(save_dir,
                                                                             f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_2_alt_region_{reg_idx + 1}.png"))
                        self.generated_figures.append(fig2_alt)

                    if not alt_cycle_params:
                        print(
                            f"Batch: No valid vibrato cycles found in alternative analysis for {os.path.basename(file_path)}. Skipping.")
                        continue
                    all_cycle_params = alt_cycle_params
                    region_list = alt_region_list

                # Process aggregated cycle parameters
                if not all_cycle_params:
                    print(f"Batch: No valid vibrato cycles found for {os.path.basename(file_path)}. Skipping.")
                    continue

                df_detailed = create_vibrato_dataframe(all_cycle_params)
                df_detailed['File_Name'] = os.path.basename(file_path)
                df_detailed['VibratoRate'] = 1 / (2 * df_detailed['cycle_time'])
                batch_detailed_list.append(df_detailed)

                cycle_times_all = df_detailed['cycle_time'].values
                cycle_extents_all = df_detailed['half_extent_cents'].values
                global_stats = analyze_vibrato(cycle_times_all, cycle_extents_all)
                cv_rate, cv_extent = compute_cv(cycle_times_all, cycle_extents_all)

                # Smoothing
                window_size = 3  # Default value, as run_analysis uses a dialog (not suitable for batch)
                df_smoothed = smooth_vibrato_parameters(df_detailed, window_size=window_size)
                # === Export smoothed (rolling mean/std) vibrato parameters as CSV ===
                try:
                    smoothed_csv_path = os.path.join(save_dir,
                                                     f"{os.path.splitext(os.path.basename(self.file_path))[0]}_detailed_vibrato_data_smoothed.csv")
                    df_smoothed.to_csv(smoothed_csv_path)
                    print(f"[INFO] Smoothed vibrato data saved: {smoothed_csv_path}")
                except Exception as e:
                    print(f"[WARN] Could not save smoothed vibrato CSV: {e}")
                smoothed_cycle_times = df_smoothed[('cycle_time', 'mean')].dropna().values
                smoothed_cycle_extents = df_smoothed[('half_extent_cents', 'mean')].dropna().values
                global_stats_smooth = analyze_vibrato(smoothed_cycle_times, smoothed_cycle_extents)
                cv_rate_smooth, cv_extent_smooth = compute_cv(smoothed_cycle_times, smoothed_cycle_extents)

                # Sample Entropy calculations
                def safe_sampen(series, label):
                    if len(series) >= 5 and np.std(series) > 0:
                        norm = (series - np.mean(series)) / np.std(series)
                        m = int(float(self.sampen_m.get()))  # Convert to int
                        r = float(self.sampen_r.get())  # Convert to float
                        value = sample_entropy(norm, m=m, r=r, distance=self.sampen_distance.get())
                        print(f"[DEBUG SampEn] Target: {label} | SampEn = {value:.6f}")
                        return value
                    else:
                        print(f"[DEBUG SampEn] Target: {label} | Series too short or flat.")
                        return np.nan

                sampen_rate = safe_sampen(df_detailed['VibratoRate'].dropna().values, "rate")
                sampen_extent = safe_sampen(df_detailed['half_extent_cents'].dropna().values, "extent")
                sampen_cycle_time = safe_sampen(df_detailed['cycle_time'].dropna().values, "cycle_time")

                # Select SampEn display value
                if target == "rate":
                    sampen_display = sampen_rate
                elif target == "extent":
                    sampen_display = sampen_extent
                elif target == "cycle_time":
                    sampen_display = sampen_cycle_time
                else:
                    sampen_display = np.nan

                # Create summary data
                if self.alternative_used:
                    summary_data_file = {
                        'mean_rate_unsmoothed': global_stats['mean_rate'],
                        'stdev_rate_unsmoothed': global_stats['stdev_rate'],
                        'median_rate_unsmoothed': global_stats['median_rate'],
                        'mean_extent_unsmoothed': global_stats['mean_extent'],
                        'stdev_extent_unsmoothed': global_stats['stdev_extent'],
                        'median_extent_unsmoothed': global_stats['median_extent'],
                        'mean_rate_smooth': global_stats_smooth['mean_rate'] if len(smoothed_cycle_times) else np.nan,
                        'stdev_rate_smooth': global_stats_smooth['stdev_rate'] if len(smoothed_cycle_times) else np.nan,
                        'median_rate_smooth': global_stats_smooth['median_rate'] if len(
                            smoothed_cycle_times) else np.nan,
                        'mean_extent_smooth': global_stats_smooth['mean_extent'] if len(
                            smoothed_cycle_times) else np.nan,
                        'stdev_extent_smooth': global_stats_smooth['stdev_extent'] if len(
                            smoothed_cycle_times) else np.nan,
                        'median_extent_smooth': global_stats_smooth['median_extent'] if len(
                            smoothed_cycle_times) else np.nan,
                        'Global_Jitter': global_stats['jitter'],
                        'Global_CV_Rate_%': cv_rate,
                        'Global_CV_Extent_%': cv_extent,
                        'mean_pitch_hz': np.nanmean(self.pitch_hz),
                        'min_pitch_hz': np.nanmin(self.pitch_hz),
                        'max_pitch_hz': np.nanmax(self.pitch_hz),
                        'HarmonicUsed': f"Harmonic {self.harmonic_used}",
                        'HarmonicFrequency': self.harmonic_frequency,
                        'SampEn_Rate': sampen_rate,
                        'SampEn_Extent': sampen_extent,
                        'SampEn_CycleTime': sampen_cycle_time,
                        'SampEn_Target': target,
                        'SampEn_Distance': dist,
                        'PitchMethod': self.actual_pitch_method.get()
                    }
                else:
                    summary_data_file = {
                        'mean_rate_unsmoothed': global_stats['mean_rate'],
                        'stdev_rate_unsmoothed': global_stats['stdev_rate'],
                        'median_rate_unsmoothed': global_stats['median_rate'],
                        'mean_extent_unsmoothed': global_stats['mean_extent'],
                        'stdev_extent_unsmoothed': global_stats['stdev_extent'],
                        'median_extent_unsmoothed': global_stats['median_extent'],
                        'mean_rate_smooth': global_stats_smooth['mean_rate'] if len(smoothed_cycle_times) else np.nan,
                        'stdev_rate_smooth': global_stats_smooth['stdev_rate'] if len(smoothed_cycle_times) else np.nan,
                        'median_rate_smooth': global_stats_smooth['median_rate'] if len(
                            smoothed_cycle_times) else np.nan,
                        'mean_extent_smooth': global_stats_smooth['mean_extent'] if len(
                            smoothed_cycle_times) else np.nan,
                        'stdev_extent_smooth': global_stats_smooth['stdev_extent'] if len(
                            smoothed_cycle_times) else np.nan,
                        'median_extent_smooth': global_stats_smooth['median_extent'] if len(
                            smoothed_cycle_times) else np.nan,
                        'Global_Jitter': global_stats['jitter'],
                        'Global_CV_Rate_%': cv_rate,
                        'Global_CV_Extent_%': cv_extent,
                        'mean_pitch_hz': np.nanmean(self.pitch_hz),
                        'min_pitch_hz': np.nanmin(self.pitch_hz),
                        'max_pitch_hz': np.nanmax(self.pitch_hz),
                        'HarmonicUsed': "Fundamental",
                        'HarmonicFrequency': np.nanmedian(self.pitch_hz),
                        'SampEn_Rate': sampen_rate,
                        'SampEn_Extent': sampen_extent,
                        'SampEn_CycleTime': sampen_cycle_time,
                        'SampEn_Target': target,
                        'SampEn_Distance': dist,
                        'PitchMethod': self.actual_pitch_method.get()
                    }

                # Store results
                avg_dict = summary_data_file.copy()
                avg_dict['File_Name'] = os.path.basename(file_path)
                batch_avg_list.append(avg_dict)
                batch_region_list.extend(region_list)

                for reg_item in region_list:
                    reg_item.update(summary_data_file)

                # Plot Figure_3 (vibrato rate)
                save_path_fig3 = os.path.join(save_dir,
                                              f"{os.path.splitext(os.path.basename(file_path))[0]}_Figure_3.png")
                fig_r = plot_vibrato_rate(df_detailed, df_smoothed, save_path=save_path_fig3)
                self.generated_figures.append(fig_r)

                # Plot final analysis summary
                save_path_sum = os.path.join(save_dir,
                                             f"{os.path.splitext(os.path.basename(file_path))[0]}_final_analysis.png")
                print(
                    f"[FINAL SampEn] {os.path.basename(file_path)} | SampEn Rate = {sampen_rate:.6f}, Extent = {sampen_extent:.6f}, Cycle Time = {sampen_cycle_time:.6f}")
                fig_summary = final_plot(
                    df_detailed, df_smoothed, summary_data_file,
                    jitter_metrics=compute_jitter_metrics(df_detailed['cycle_time'].values * 2),
                    cv_rate=cv_rate, cv_extent=cv_extent,
                    cv_rate_smooth=cv_rate_smooth, cv_extent_smooth=cv_extent_smooth,
                    filename=file_path,
                    sampen_rate=sampen_rate,
                    sampen_extent=sampen_extent,
                    sampen_cycle_time=sampen_cycle_time,
                    sampen_target=target,
                    sampen_distance=dist,
                    title=f"Vibrato Scope: Final Analysis ({os.path.basename(file_path)})",
                    show_figure=False
                )
                fig_summary.savefig(save_path_sum, dpi=300)
                plt.close(fig_summary)
                self.generated_figures.append(save_path_sum)

            except Exception as e:
                print(f"Batch: Error processing {os.path.basename(file_path)}: {str(e)}")
                continue

        # Save aggregated results
        if not batch_avg_list:
            messagebox.showinfo("Batch Processing", "No valid vibrato cycles found in the selected files.")
            return

        df_avg_all = pd.DataFrame(batch_avg_list)
        df_detailed_all = pd.concat(batch_detailed_list, ignore_index=True) if batch_detailed_list else pd.DataFrame()
        df_region_all = pd.DataFrame(batch_region_list) if batch_region_list else pd.DataFrame()

        df_avg_all.to_csv(os.path.join(save_dir, "averaged_vibrato_data.csv"), index=False)
        df_detailed_all.to_csv(os.path.join(save_dir, "detailed_vibrato_data.csv"), index=False)
        df_region_all.to_csv(os.path.join(save_dir, "region_vibrato_data.csv"), index=False)

        messagebox.showinfo("Batch Processing", f"Batch processing complete.\nResults saved in:\n{save_dir}")

    def save_results(self):
        if self.last_fig is None or self.df_detailed is None:
            messagebox.showwarning("Save Results", "No analysis results to save. Please run analysis first.")
            return
        default_dir = Path.home() / "Documents" / "VibratoScope"
        default_dir.mkdir(parents=True, exist_ok=True)

        save_dir = filedialog.askdirectory(title="Select folder to save results", initialdir=str(default_dir))
        if not save_dir or not os.access(save_dir, os.W_OK):
            messagebox.showerror("Save Results", "Invalid or unwritable directory selected!")
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

        # Save figures
        for idx, fig_item in enumerate(self.generated_figures):
            if isinstance(fig_item, str):
                src = os.path.abspath(fig_item)
                dst = os.path.join(save_dir, os.path.basename(fig_item))
                if src != dst and os.path.exists(src):
                    try:
                        shutil.copy(src, dst)
                    except shutil.SameFileError:
                        pass
            else:
                fig_path = os.path.join(save_dir, f"figure_{idx}.png")
                fig_item.savefig(fig_path, dpi=300)
                plt.close(fig_item)

        # Save the final summary figure
        figure_path = os.path.join(save_dir, "final_analysis.png")
        self.last_fig.savefig(figure_path, dpi=300)
        plt.close(self.last_fig)

        messagebox.showinfo("Save Results", f"Results saved in:\n{save_dir}")
