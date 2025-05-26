# audio_recorder.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pyaudio
import wave
import threading
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import psutil
import shutil
import os
import queue

class AudioRecorder(tk.Toplevel):
    def __init__(self, parent, on_save=None):
        super().__init__(parent)
        self.title("Record Audio")
        self.geometry("1280x720")
        self.on_save = on_save

        self.recording = False
        self.tmp_file = "temp_recording.wav"
        self.sr = None
        self.duration = 5
        self.chunk = 2048  # Increased for stability
        self.rec_thread = None
        self.rec_exit_event = None
        self.live_plot = False
        self.device_index = None
        self.plot_mode = "Raw"
        self.db_range = 80  # Default to ±80 dB for voice
        self.audio_queue = queue.Queue()
        self.plot_data = []

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.init_gui()
        self.populate_devices()

    def init_gui(self):
        from tkinter import font as tkFont

        self.configure(bg="#b0afa6")
        cal_font = tkFont.Font(family="Calibri", size=11) if "Calibri" in tkFont.families() else tkFont.Font(
            family="Arial", size=11)

        # Header
        tk.Label(self, text="Before recording select the options below.",
                 bg="#b0afa6", font=(cal_font.actual("family"), 14, "bold")).pack(pady=(20, 5))

        tk.Label(self, text="Don't forget to increase Max Duration if you want to sing a song", bg="#b0afa6", font=(cal_font.actual("family"), 12, "italic")).pack()

        # Input device
        device_frame = tk.Frame(self, bg="#b0afa6")
        device_frame.pack(pady=5)
        tk.Label(device_frame, text="Input Device:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.device_var = tk.StringVar()
        self.device_menu = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly", width=50)
        self.device_menu.pack(side=tk.LEFT)

        # Sample Rate
        sr_frame = tk.Frame(self, bg="#b0afa6")
        sr_frame.pack(pady=2)
        tk.Label(sr_frame, text="Sample Rate (Hz):", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.sr_var = tk.StringVar(value="44100")
        self.sr_menu = ttk.Combobox(sr_frame, textvariable=self.sr_var, state="readonly", width=10)
        self.sr_menu.pack(side=tk.LEFT)

        # Buffer Size
        buf_frame = tk.Frame(self, bg="#b0afa6")
        buf_frame.pack(pady=2)
        tk.Label(buf_frame, text="Buffer Size:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.buffer_var = tk.StringVar(value="2048")
        self.buffer_menu = ttk.Combobox(buf_frame, textvariable=self.buffer_var, state="readonly", width=10)
        self.buffer_menu["values"] = ["512", "1024", "2048", "4096"]
        self.buffer_menu.pack(side=tk.LEFT)

        # Max Duration
        dur_frame = tk.Frame(self, bg="#b0afa6")
        dur_frame.pack(pady=2)
        tk.Label(dur_frame, text="Max Duration (s):", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.duration_entry = tk.Entry(dur_frame, width=10)
        self.duration_entry.insert(0, "5")
        self.duration_entry.pack(side=tk.LEFT)

        # Plot mode
        plot_frame = tk.Frame(self, bg="#b0afa6")
        plot_frame.pack(pady=2)
        tk.Label(plot_frame, text="Plot Mode:", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.plot_mode_var = tk.StringVar(value="Raw")
        self.plot_mode_menu = ttk.Combobox(plot_frame, textvariable=self.plot_mode_var, state="readonly", width=10)
        self.plot_mode_menu["values"] = ["Raw", "dB"]
        self.plot_mode_menu.pack(side=tk.LEFT)
        self.plot_mode_menu.bind("<<ComboboxSelected>>", self.update_plot_mode)

        # dB Range
        db_frame = tk.Frame(self, bg="#b0afa6")
        db_frame.pack(pady=2)
        tk.Label(db_frame, text="dB Range (±):", bg="#b0afa6", font=cal_font).pack(side=tk.LEFT, padx=5)
        self.db_range_var = tk.StringVar(value="80")
        self.db_range_menu = ttk.Combobox(db_frame, textvariable=self.db_range_var, state="readonly", width=10)
        self.db_range_menu["values"] = ["60", "80", "100", "120", "140"]
        self.db_range_menu.pack(side=tk.LEFT)
        self.db_range_menu.bind("<<ComboboxSelected>>", self.update_db_range)

        # Enable Live Plotting
        self.live_plot_var = tk.BooleanVar(value=False)
        self.live_plot_check = tk.Checkbutton(self, text="Enable Live Plotting", variable=self.live_plot_var,
                                              command=self.update_live_plot, bg="#b0afa6", font=cal_font)
        self.live_plot_check.pack(pady=5)

        # Plot canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(padx=20, pady=10, fill=tk.BOTH, expand=False)

        # Buttons
        button_frame = tk.Frame(self, bg="#b0afa6")
        button_frame.pack(pady=15)

        self.start_btn = tk.Button(button_frame, text="Record", width=12, font=cal_font, command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(button_frame, text="Stop", width=12, font=cal_font, command=self.stop_recording,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        self.save_btn = tk.Button(button_frame, text="Save", width=12, font=cal_font, command=self.save_recording,
                                  state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=10)

        self.cancel_btn = tk.Button(button_frame, text="Cancel", width=12, font=cal_font, command=self.destroy)
        self.cancel_btn.pack(side=tk.LEFT, padx=10)

    def update_plot_mode(self, event=None):
        self.plot_mode = self.plot_mode_var.get()
        self.update_axes()

    def update_db_range(self, event=None):
        self.db_range = int(self.db_range_var.get())
        self.update_axes()

    def update_axes(self):
        self.ax.clear()
        if self.plot_mode == "Raw":
            self.ax.set_ylim(-32768, 32768)
            self.ax.set_ylabel("Amplitude")
        else:
            self.ax.set_ylim(-self.db_range, self.db_range)
            self.ax.set_ylabel("Amplitude (dB)")
        self.ax.set_xlabel("Time (s)")
        self.canvas.draw()

    def update_live_plot(self):
        self.live_plot = self.live_plot_var.get()
        self.logger.info(f"Live plotting {'enabled' if self.live_plot else 'disabled'}")

    def populate_devices(self):
        pa = pyaudio.PyAudio()
        devices = []
        self.device_indices = []
        seen_names = set()
        default_index = pa.get_default_input_device_info().get("index", None)

        for i in range(pa.get_device_count()):
            dev_info = pa.get_device_info_by_index(i)
            if dev_info["maxInputChannels"] > 0:
                name = dev_info["name"]
                if name not in seen_names:
                    seen_names.add(name)
                    display_name = f"{name} (Index: {i})"
                    devices.append(display_name)
                    self.device_indices.append(i)
                    if i == default_index:
                        self.sr = int(dev_info["defaultSampleRate"])
                        self.sr_var.set(str(self.sr))

        pa.terminate()
        self.device_menu["values"] = devices
        self.sr_menu["values"] = ["16000", "22050", "44100", "48000"]
        if self.sr:
            self.sr_menu.set(str(self.sr))
        if devices:
            for i, idx in enumerate(self.device_indices):
                if idx == default_index:
                    self.device_menu.current(i)
                    self.device_index = idx
                    break
            else:
                self.device_menu.current(0)
                self.device_index = self.device_indices[0]
        else:
            messagebox.showwarning("Warning", "No active input devices found.")
            self.device_menu["values"] = ["No devices"]
            self.device_menu.current(0)
            self.start_btn.config(state=tk.DISABLED)

    def start_recording(self):
        if os.path.exists(self.tmp_file):
            try:
                os.remove(self.tmp_file)
            except Exception as e:
                self.logger.error(f"Error removing existing temp file: {e}")
        self.rec_exit_event = threading.Event()
        self.ax.clear()
        self.update_axes()
        self.plot_data = []
        self.audio_queue.queue.clear()  # Clear queue for new recording
        try:
            self.sr = int(self.sr_var.get())
            self.duration = float(self.duration_entry.get())
            self.chunk = int(self.buffer_var.get())
            self.device_index = self.device_indices[self.device_menu.current()]
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Invalid sample rate, duration, buffer size, or device selection.")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)

        self.recording = True
        self.rec_thread = threading.Thread(target=self.recorder)
        self.logger.info(f"Starting recording: SR={self.sr}, Buffer={self.chunk}, CPU={psutil.cpu_percent()}%")
        self.rec_thread.start()
        if self.live_plot:
            self.update_plot()

    def recorder(self):
        pa = pyaudio.PyAudio()
        wavefile = None
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sr,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.device_index
            )
            wavefile = wave.open(self.tmp_file, 'wb')
            wavefile.setnchannels(1)
            wavefile.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wavefile.setframerate(self.sr)
        except Exception as e:
            self.logger.error(f"Failed to open stream or WAV file: {e}")
            messagebox.showerror("Error", f"Failed to open audio stream or WAV file: {e}")
            self.after_recording()
            return

        frame_count = 0
        max_frames = int((self.sr / self.chunk) * self.duration)
        start_time = time.time()

        while frame_count < max_frames and not self.rec_exit_event.is_set():
            try:
                audio = stream.read(self.chunk, exception_on_overflow=False)
                wavefile.writeframes(audio)
                frame_count += 1
                if self.live_plot:
                    audio_data = np.frombuffer(audio, dtype=np.int16)
                    self.audio_queue.put(audio_data)
                if frame_count % 10 == 0:  # Log every 10 frames
                    cpu_usage = psutil.cpu_percent()
                    self.logger.info(f"Frame {frame_count}/{max_frames}, CPU={cpu_usage}%")
                    if cpu_usage > 80:
                        self.logger.warning("High CPU usage detected, may affect recording")
            except IOError as e:
                self.logger.error(f"Stream read error: {e}")
                if "Input overflowed" in str(e):
                    self.logger.warning("Input buffer overflowed. Try increasing buffer size.")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                break

        elapsed_time = time.time() - start_time
        self.logger.info(f"Recording complete: {frame_count}/{max_frames} frames, {elapsed_time:.2f}s, CPU={psutil.cpu_percent()}%")
        try:
            wavefile.close()
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as e:
            self.logger.error(f"Error closing resources: {e}")
        self.after_recording()

    def update_plot(self):
        if not self.recording or self.rec_exit_event.is_set():
            return
        try:
            while not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                if self.plot_mode == "dB":
                    abs_data = np.abs(audio_data)
                    # Threshold for low-amplitude noise
                    mask = abs_data < 100  # ~0.3% of full scale
                    db_data = np.zeros_like(audio_data, dtype=float)
                    if np.any(~mask):
                        valid_data = abs_data[~mask] / 32768
                        valid_data = np.maximum(valid_data, 1e-10)
                        db_data[~mask] = 20 * np.log10(valid_data) * np.sign(audio_data[~mask])
                    db_data = np.clip(db_data, -self.db_range, self.db_range)
                    plot_data = db_data
                else:
                    plot_data = audio_data
                self.plot_data.extend(plot_data)
                # Limit plot data to last 0.5 seconds for performance
                max_samples = self.sr // 2
                if len(self.plot_data) > max_samples:
                    self.plot_data = self.plot_data[-max_samples:]

            if self.plot_data:
                time_axis = np.linspace(0, len(self.plot_data) / self.sr, len(self.plot_data))
                self.ax.clear()
                self.ax.plot(time_axis, self.plot_data, color='blue', alpha=0.5)
                self.ax.set_xlabel("Time (s)")
                if self.plot_mode == "Raw":
                    self.ax.set_ylim(-32768, 32768)
                    self.ax.set_ylabel("Amplitude")
                else:
                    self.ax.set_ylim(-self.db_range, self.db_range)
                    self.ax.set_ylabel("Amplitude (dB)")
                self.canvas.draw()
        except Exception as e:
            self.logger.error(f"Error updating plot: {e}")
        if self.recording:
            self.after(100, self.update_plot)  # 100ms interval for stability

    def after_recording(self):
        self.recording = False
        self.rec_thread = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if os.path.exists(self.tmp_file):
            self.save_btn.config(state=tk.NORMAL)
        else:
            self.logger.error("Temporary file missing after recording")
            messagebox.showerror("Error", "Recording failed: Temporary file not created")

        if os.path.exists(self.tmp_file):
            try:
                with wave.open(self.tmp_file, 'rb') as wavefile:
                    audio_data = np.frombuffer(wavefile.readframes(wavefile.getnframes()), dtype=np.int16)
                if self.plot_mode == "dB":
                    abs_data = np.abs(audio_data)
                    mask = abs_data < 100  # ~0.3% of full scale
                    db_data = np.zeros_like(audio_data, dtype=float)
                    if np.any(~mask):
                        valid_data = abs_data[~mask] / 32768
                        valid_data = np.maximum(valid_data, 1e-10)
                        db_data[~mask] = 20 * np.log10(valid_data) * np.sign(audio_data[~mask])
                    db_data = np.clip(db_data, -self.db_range, self.db_range)
                    plot_data = db_data
                    y_label = "Amplitude (dB)"
                    y_lim = (-self.db_range, self.db_range)
                else:
                    plot_data = audio_data
                    y_label = "Amplitude"
                    y_lim = (-32768, 32768)
                time_axis = np.linspace(0, len(plot_data) / self.sr, len(plot_data))
                self.ax.clear()
                self.ax.plot(time_axis, plot_data, color='blue')
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel(y_label)
                self.ax.set_ylim(y_lim)
                self.canvas.draw()
            except Exception as e:
                self.logger.error(f"Error plotting final waveform: {e}")

    def stop_recording(self):
        if self.rec_exit_event:
            self.rec_exit_event.set()

    def save_recording(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if save_path:
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
                shutil.move(self.tmp_file, save_path)
                messagebox.showinfo("Saved", f"Recording saved to {save_path}")
                if self.on_save:
                    self.on_save(save_path)
                self.destroy()
            except Exception as e:
                self.logger.error(f"Failed to save recording: {e}")
                messagebox.showerror("Error", f"Failed to save recording: {e}")

    def destroy(self):
        if self.recording:
            self.stop_recording()
        if os.path.exists(self.tmp_file):
            try:
                os.remove(self.tmp_file)
            except Exception as e:
                self.logger.error(f"Error removing temp file: {e}")
        self.audio_queue.queue.clear()
        self.plot_data = []
        super().destroy()