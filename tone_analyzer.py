"""
Tone Generator & FFT Spectrum Analyzer
Generates a user-selected tone via the speaker and displays
the real-time FFT spectrum from the microphone.
"""

import collections
import datetime
import json
import os
import pickle
import threading
import wave
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pyaudio

# ── Preferences file ────────────────────────────────────────────
PREFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preferences.json")
DEFAULT_PREFS = {
    "x_min": 20,
    "x_max": 20000,
    "frequency": 440.0,
    "volume": 0.5,
    "waveform": "sine",
    "pulse_f_start": 200,
    "pulse_f_end": 2000,
    "pulse_length_ms": 50,
    "pulse_gap_ms": 100,
    "time_window_ms": 20,
    "time_amplitude": "auto",
    "pulse_fft": False,
    "classify": False,
}


def load_prefs():
    try:
        with open(PREFS_PATH, "r") as f:
            saved = json.load(f)
        # Merge with defaults so new keys are always present
        prefs = {**DEFAULT_PREFS, **saved}
        return prefs
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_PREFS)


def save_prefs(prefs):
    with open(PREFS_PATH, "w") as f:
        json.dump(prefs, f, indent=2)

# ── Audio constants ──────────────────────────────────────────────
SAMPLE_RATE = 48000
CHUNK = 1024          # smaller chunks → lower latency
FORMAT = pyaudio.paFloat32
CHANNELS = 1
FFT_SIZE = 8192       # ~5 Hz resolution for sharp tone peaks

# ── PyAudio instance (shared) ───────────────────────────────────
pa = pyaudio.PyAudio()


class ToneGenerator:
    """Continuously streams a sine wave to the default output device."""

    def __init__(self):
        self.frequency = 440.0
        self.amplitude = 0.5
        self.playing = False
        self.phase = 0.0
        self.stream = None
        self._lock = threading.Lock()
        # Gaussian pulse parameters
        self.pulse_f_start = 200
        self.pulse_f_end = 2000
        self.pulse_length_ms = 50
        self.pulse_gap_ms = 100
        self._pulse_pos = 0  # running sample counter for pulse+gap cycle

    # callback runs on a separate audio thread
    def _callback(self, in_data, frame_count, time_info, status):
        with self._lock:
            freq = self.frequency
            amp = self.amplitude
        t = np.arange(frame_count) / SAMPLE_RATE
        samples = (amp * np.sin(2 * np.pi * freq * t + self.phase)).astype(np.float32)
        self.phase += 2 * np.pi * freq * frame_count / SAMPLE_RATE
        self.phase %= 2 * np.pi
        return (samples.tobytes(), pyaudio.paContinue)

    def start(self):
        if self.playing:
            return
        self.phase = 0.0
        self.stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._callback,
        )
        self.stream.start_stream()
        self.playing = True

    def stop(self):
        if not self.playing:
            return
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        self.playing = False

    def set_frequency(self, freq):
        with self._lock:
            self.frequency = freq

    def set_amplitude(self, amp):
        with self._lock:
            self.amplitude = amp

    def set_pulse_params(self, f_start, f_end, length_ms, gap_ms):
        with self._lock:
            self.pulse_f_start = f_start
            self.pulse_f_end = f_end
            self.pulse_length_ms = length_ms
            self.pulse_gap_ms = gap_ms


class MicAnalyzer:
    """Non-blocking mic capture using a PyAudio callback thread."""

    def __init__(self):
        self.stream = None
        self.running = False
        self._lock = threading.Lock()
        # Ring buffer for FFT
        self._buf = collections.deque(maxlen=FFT_SIZE)
        # Separate ring buffer for time-domain display (up to 5s)
        self._time_buf = collections.deque(maxlen=SAMPLE_RATE * 5)
        self._new_data = False
        self._recording = False
        self._rec_frames = []

    def _callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.float32)
        with self._lock:
            self._buf.extend(samples)
            self._time_buf.extend(samples)
            self._new_data = True
            if self._recording:
                self._rec_frames.append(in_data)
        return (None, pyaudio.paContinue)

    def start(self):
        if self.running:
            return
        self.stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._callback,
        )
        self.stream.start_stream()
        self.running = True

    def get_snapshot(self):
        """Return the latest FFT_SIZE samples (non-blocking). Returns None if no new data."""
        with self._lock:
            if not self._new_data:
                return None
            self._new_data = False
            arr = np.array(self._buf, dtype=np.float32)
        # Pad to FFT_SIZE if not enough samples yet
        if len(arr) < FFT_SIZE:
            arr = np.pad(arr, (FFT_SIZE - len(arr), 0))
        return arr

    def get_time_samples(self, n):
        """Return the last n samples from the time-domain buffer."""
        with self._lock:
            buf_len = len(self._time_buf)
            if buf_len >= n:
                # Slice the last n items from the deque
                arr = np.array(list(self._time_buf)[-n:], dtype=np.float32)
            else:
                arr = np.array(self._time_buf, dtype=np.float32)
                arr = np.pad(arr, (n - buf_len, 0))
        return arr

    def start_recording(self):
        with self._lock:
            self._rec_frames = []
            self._recording = True

    def stop_recording(self, filepath):
        with self._lock:
            self._recording = False
            frames = list(self._rec_frames)
            self._rec_frames = []
        # Convert float32 to int16 for WAV compatibility
        raw = b"".join(frames) if frames else b""
        float_samples = np.frombuffer(raw, dtype=np.float32) if raw else np.array([], dtype=np.float32)
        int_samples = np.clip(float_samples * 32767, -32768, 32767).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(int_samples.tobytes())

    def stop(self):
        if not self.running:
            return
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        self.running = False


# ── Pacman maze template ─────────────────────────────────────────
# 0 = open path, 1 = wall (dots placed on open cells at runtime)
_PACMAN_MAZE_TEMPLATE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,0,1,0,1,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,0,1,1,0,1,0,1,1,0,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,0,1,1,0,1,0,1,1,0,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,0,1,0,1,0,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.int8)


class App(tk.Tk):
    def __init__(self, game=None):
        super().__init__()
        self.title("Tone Generator & FFT Spectrum Analyzer")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._game = game  # None, "pong", "pacman", or "tron"

        self.prefs = load_prefs()
        self.tone = ToneGenerator()
        self.tone.set_frequency(self.prefs["frequency"])
        self.tone.set_amplitude(self.prefs["volume"])
        self.mic = MicAnalyzer()

        # Load classifier model early (needed by controls and plot)
        self._clf_model = None
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "classifier", "model.pkl")
        if os.path.isfile(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._clf_model = pickle.load(f)
            except Exception:
                self._clf_model = None

        self._build_controls()
        self._build_plot()

        # Start the mic stream and kick off the update loop
        self._closing = False
        self._after_id = None
        self.mic.start()
        self._update_plot()
        self.bind("<space>", lambda e: self._toggle_tone())
        self._toggle_tone()  # Start playing immediately

    @staticmethod
    def _scale_click_jump(event):
        """Click anywhere on a ttk.Scale trough to jump to that position."""
        scale = event.widget
        # Only intercept trough clicks; let thumb clicks pass through for dragging
        if scale.identify(event.x, event.y) != "trough":
            return
        if scale.cget("orient") == "horizontal":
            frac = event.x / scale.winfo_width()
        else:
            frac = event.y / scale.winfo_height()
        lo = float(scale.cget("from"))
        hi = float(scale.cget("to"))
        scale.set(lo + frac * (hi - lo))
        return "break"

    # ── GUI layout ───────────────────────────────────────────────
    def _build_controls(self):
        ctrl = ttk.Frame(self, padding=10)
        ctrl.pack(fill=tk.X)

        # Frequency
        ttk.Label(ctrl, text="Frequency (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.freq_var = tk.DoubleVar(value=self.prefs["frequency"])
        self.freq_slider = ttk.Scale(
            ctrl, from_=20, to=20000, variable=self.freq_var,
            orient=tk.HORIZONTAL, command=self._on_freq_change,
        )
        self.freq_slider.bind("<Button-1>", self._scale_click_jump)
        self.freq_slider.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.freq_label = ttk.Label(ctrl, text=f"{self.prefs['frequency']:.0f} Hz", width=10)
        self.freq_label.grid(row=0, column=2)

        # Manual entry
        self.freq_entry = ttk.Entry(ctrl, width=8)
        self.freq_entry.insert(0, f"{self.prefs['frequency']:.0f}")
        self.freq_entry.grid(row=0, column=3, padx=5)
        self.freq_entry.bind("<Return>", self._on_freq_entry)
        ttk.Label(ctrl, text="Hz").grid(row=0, column=4)

        # Volume
        ttk.Label(ctrl, text="Volume:").grid(row=1, column=0, sticky=tk.W)
        self.vol_var = tk.DoubleVar(value=self.prefs["volume"])
        self.vol_slider = ttk.Scale(
            ctrl, from_=0.0, to=1.0, variable=self.vol_var,
            orient=tk.HORIZONTAL, command=self._on_vol_change,
        )
        self.vol_slider.bind("<Button-1>", self._scale_click_jump)
        self.vol_slider.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.vol_label = ttk.Label(ctrl, text=f"{self.prefs['volume'] * 100:.0f}%", width=10)
        self.vol_label.grid(row=1, column=2)

        # Play / Stop
        self.play_btn = ttk.Button(ctrl, text="▶  Play Tone", command=self._toggle_tone)
        self.play_btn.grid(row=0, column=5, rowspan=2, padx=15, sticky=tk.NS)

        # Record
        self.rec_btn = ttk.Button(ctrl, text="●  Record", command=self._toggle_record)
        self.rec_btn.grid(row=0, column=6, rowspan=2, padx=5, sticky=tk.NS)

        # Waveform selector
        ttk.Label(ctrl, text="Waveform:").grid(row=2, column=0, sticky=tk.W)
        self.wave_var = tk.StringVar(value=self.prefs["waveform"])
        wave_frame = ttk.Frame(ctrl)
        wave_frame.grid(row=2, column=1, columnspan=4, sticky=tk.W, padx=5)
        for w in ("sine", "square", "sawtooth", "triangle", "gaussian"):
            ttk.Radiobutton(wave_frame, text=w.capitalize(), variable=self.wave_var,
                            value=w, command=self._on_wave_change).pack(side=tk.LEFT, padx=4)

        # Gaussian pulse parameters (row 3) - shown/hidden based on waveform
        self.gauss_frame = ttk.Frame(ctrl)
        self.gauss_frame.grid(row=3, column=0, columnspan=6, sticky=tk.EW, pady=(4, 0))

        ttk.Label(self.gauss_frame, text="Pulse:").pack(side=tk.LEFT)

        ttk.Label(self.gauss_frame, text="Start Freq").pack(side=tk.LEFT, padx=(8, 2))
        self.pulse_f_start_entry = ttk.Entry(self.gauss_frame, width=6)
        self.pulse_f_start_entry.insert(0, str(self.prefs["pulse_f_start"]))
        self.pulse_f_start_entry.bind("<Return>", lambda e: self._apply_pulse_params())
        self.pulse_f_start_entry.pack(side=tk.LEFT)
        ttk.Label(self.gauss_frame, text="Hz").pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(self.gauss_frame, text="End Freq").pack(side=tk.LEFT, padx=(0, 2))
        self.pulse_f_end_entry = ttk.Entry(self.gauss_frame, width=6)
        self.pulse_f_end_entry.insert(0, str(self.prefs["pulse_f_end"]))
        self.pulse_f_end_entry.bind("<Return>", lambda e: self._apply_pulse_params())
        self.pulse_f_end_entry.pack(side=tk.LEFT)
        ttk.Label(self.gauss_frame, text="Hz").pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(self.gauss_frame, text="Pulse").pack(side=tk.LEFT, padx=(0, 2))
        self.pulse_len_entry = ttk.Entry(self.gauss_frame, width=5)
        self.pulse_len_entry.insert(0, str(self.prefs["pulse_length_ms"]))
        self.pulse_len_entry.bind("<Return>", lambda e: self._apply_pulse_params())
        self.pulse_len_entry.pack(side=tk.LEFT)
        ttk.Label(self.gauss_frame, text="ms").pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(self.gauss_frame, text="Gap").pack(side=tk.LEFT, padx=(0, 2))
        self.pulse_gap_entry = ttk.Entry(self.gauss_frame, width=5)
        self.pulse_gap_entry.insert(0, str(self.prefs["pulse_gap_ms"]))
        self.pulse_gap_entry.bind("<Return>", lambda e: self._apply_pulse_params())
        self.pulse_gap_entry.pack(side=tk.LEFT)
        ttk.Label(self.gauss_frame, text="ms").pack(side=tk.LEFT, padx=(2, 8))

        ttk.Button(self.gauss_frame, text="Apply", command=self._apply_pulse_params).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self.gauss_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.pulse_fft_var = tk.BooleanVar(value=self.prefs.get("pulse_fft", False))
        ttk.Checkbutton(self.gauss_frame, text="Pulse FFT",
                        variable=self.pulse_fft_var).pack(side=tk.LEFT, padx=4)

        self.classify_var = tk.BooleanVar(value=self.prefs.get("classify", False))
        if self._clf_model is not None:
            ttk.Checkbutton(self.gauss_frame, text="Classify",
                            variable=self.classify_var).pack(side=tk.LEFT, padx=4)

        # Initialize visibility
        if self.prefs["waveform"] != "gaussian":
            self.gauss_frame.grid_remove()

        # Initialize ToneGenerator pulse params from prefs
        self.tone.set_pulse_params(
            self.prefs["pulse_f_start"],
            self.prefs["pulse_f_end"],
            self.prefs["pulse_length_ms"],
            self.prefs["pulse_gap_ms"],
        )

        # X-axis range
        xrange_frame = ttk.Frame(ctrl)
        xrange_frame.grid(row=4, column=0, columnspan=6, sticky=tk.EW, pady=(6, 0))

        ttk.Label(xrange_frame, text="X-axis:").pack(side=tk.LEFT)
        ttk.Label(xrange_frame, text="Min").pack(side=tk.LEFT, padx=(8, 2))
        self.xmin_entry = ttk.Entry(xrange_frame, width=7)
        self.xmin_entry.insert(0, str(self.prefs["x_min"]))
        self.xmin_entry.bind("<Return>", lambda e: self._apply_xrange())
        self.xmin_entry.pack(side=tk.LEFT)
        ttk.Label(xrange_frame, text="Hz").pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(xrange_frame, text="Max").pack(side=tk.LEFT, padx=(0, 2))
        self.xmax_entry = ttk.Entry(xrange_frame, width=7)
        self.xmax_entry.insert(0, str(self.prefs["x_max"]))
        self.xmax_entry.bind("<Return>", lambda e: self._apply_xrange())
        self.xmax_entry.pack(side=tk.LEFT)
        ttk.Label(xrange_frame, text="Hz").pack(side=tk.LEFT, padx=(2, 10))

        ttk.Button(xrange_frame, text="Apply", command=self._apply_xrange).pack(side=tk.LEFT, padx=4)

        # Preset buttons
        for label, lo, hi in [("Voice", 50, 4000), ("Full", 20, 20000), ("Bass", 20, 500), ("Mid", 200, 6000)]:
            ttk.Button(
                xrange_frame, text=label, width=5,
                command=lambda l=lo, h=hi: self._set_xrange(l, h),
            ).pack(side=tk.LEFT, padx=2)

        # Time-domain window
        trange_frame = ttk.Frame(ctrl)
        trange_frame.grid(row=5, column=0, columnspan=6, sticky=tk.EW, pady=(4, 0))

        ttk.Label(trange_frame, text="Time window:").pack(side=tk.LEFT)
        self.time_win_entry = ttk.Entry(trange_frame, width=6)
        self.time_win_entry.insert(0, str(self.prefs["time_window_ms"]))
        self.time_win_entry.bind("<Return>", lambda e: self._apply_time_window())
        self.time_win_entry.pack(side=tk.LEFT, padx=(8, 2))
        ttk.Label(trange_frame, text="ms").pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(trange_frame, text="Apply", command=self._apply_time_window).pack(side=tk.LEFT, padx=4)

        for label, ms in [("1ms", 1), ("5ms", 5), ("20ms", 20), ("100ms", 100), ("500ms", 500)]:
            ttk.Button(
                trange_frame, text=label, width=5,
                command=lambda m=ms: self._set_time_window(m),
            ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(trange_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(trange_frame, text="Y amp:").pack(side=tk.LEFT)
        self.time_amp_entry = ttk.Entry(trange_frame, width=6)
        self.time_amp_entry.insert(0, self.prefs.get("time_amplitude", "auto"))
        self.time_amp_entry.bind("<Return>", lambda e: self._apply_time_amp())
        self.time_amp_entry.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Button(trange_frame, text="Apply", command=self._apply_time_amp).pack(side=tk.LEFT, padx=2)
        ttk.Button(trange_frame, text="Auto", width=4,
                   command=lambda: self._set_time_amp("auto")).pack(side=tk.LEFT, padx=2)

        self._time_amp_fixed = None  # None means auto
        amp_pref = self.prefs.get("time_amplitude", "auto")
        if amp_pref != "auto":
            try:
                self._time_amp_fixed = float(amp_pref)
            except ValueError:
                pass

        ctrl.columnconfigure(1, weight=1)

    def _apply_xrange(self):
        try:
            xmin = int(self.xmin_entry.get())
            xmax = int(self.xmax_entry.get())
        except ValueError:
            return
        xmin = max(0, xmin)
        xmax = min(SAMPLE_RATE // 2, xmax)
        if xmin >= xmax:
            return
        self.ax.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def _apply_time_window(self):
        try:
            ms = float(self.time_win_entry.get())
        except ValueError:
            return
        ms = max(0.1, min(5000, ms))  # up to 5 seconds
        self._time_window_ms = ms
        self._time_samples = max(2, int(ms / 1000.0 * SAMPLE_RATE))
        self._time_x = np.arange(self._time_samples) / SAMPLE_RATE * 1000
        self.ax_time.set_xlim(0, ms)
        self.time_line.set_xdata(self._time_x)
        self.time_line.set_ydata(np.zeros(self._time_samples))
        self.canvas.draw_idle()

    def _apply_time_amp(self):
        val = self.time_amp_entry.get().strip().lower()
        if val == "auto":
            self._time_amp_fixed = None
        else:
            try:
                self._time_amp_fixed = float(val)
            except ValueError:
                return
        self.canvas.draw_idle()

    def _set_time_amp(self, val):
        self.time_amp_entry.delete(0, tk.END)
        self.time_amp_entry.insert(0, val)
        self._apply_time_amp()

    def _set_time_window(self, ms):
        self.time_win_entry.delete(0, tk.END)
        self.time_win_entry.insert(0, str(ms))
        self._apply_time_window()

    def _set_xrange(self, lo, hi):
        self.xmin_entry.delete(0, tk.END)
        self.xmin_entry.insert(0, str(lo))
        self.xmax_entry.delete(0, tk.END)
        self.xmax_entry.insert(0, str(hi))
        self._apply_xrange()

    _GAME_AXES = ("ax_pong", "ax_pacman", "ax_tron", "ax_maze", "ax_world")

    def _build_plot(self):
        # Initialize all game axes to None
        for attr in self._GAME_AXES:
            setattr(self, attr, None)

        if self._game in ("pong", "pacman", "tron", "maze", "world"):
            self.fig = plt.figure(figsize=(14, 6))
            self.fig.patch.set_facecolor("#1e1e2e")
            gs = gridspec.GridSpec(
                2, 2, figure=self.fig,
                width_ratios=[3, 2], height_ratios=[1, 2],
                hspace=0.35, wspace=0.3,
            )
            self.ax_time = self.fig.add_subplot(gs[0, 0])
            self.ax = self.fig.add_subplot(gs[1, 0])
            game_ax = self.fig.add_subplot(gs[:, 1])
            setattr(self, f"ax_{self._game}", game_ax)
        else:
            self.fig, (self.ax_time, self.ax) = plt.subplots(
                2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 2]},
            )
            self.fig.patch.set_facecolor("#1e1e2e")
            self.fig.subplots_adjust(hspace=0.35)

        # ── Time-domain waveform (top) ──
        self.ax_time.set_facecolor("#1e1e2e")
        self._time_window_ms = self.prefs["time_window_ms"]
        self._time_samples = max(2, int(self._time_window_ms / 1000.0 * SAMPLE_RATE))
        self._time_x = np.arange(self._time_samples) / SAMPLE_RATE * 1000  # ms
        self.ax_time.set_xlim(0, self._time_window_ms)
        self.ax_time.set_ylim(-1, 1)
        self.ax_time.set_xlabel("Time (ms)", color="white")
        self.ax_time.set_ylabel("Amplitude", color="white")
        self.ax_time.set_title("Microphone Waveform", color="white")
        self.ax_time.tick_params(colors="white")
        for spine in self.ax_time.spines.values():
            spine.set_color("#444466")
        self.ax_time.grid(True, alpha=0.25, color="#888888")
        self.time_line, = self.ax_time.plot(
            self._time_x, np.zeros(self._time_samples), color="#f9e2af", linewidth=1,
        )

        # ── FFT spectrum (bottom) ──
        self.ax.set_facecolor("#1e1e2e")
        self.ax.set_xlim(self.prefs["x_min"], self.prefs["x_max"])
        self.ax.set_ylim(0, 140)
        self.ax.set_xscale("linear")
        self.ax.set_xlabel("Frequency (Hz)", color="white")
        self.ax.set_ylabel("Magnitude (dB)", color="white")
        self.ax.set_title("Microphone FFT Spectrum", color="white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("#444466")
        self.ax.grid(True, alpha=0.25, color="#888888")

        # Pre-compute the frequency axis for FFT_SIZE
        self._fft_freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        magnitude = np.full_like(self._fft_freqs, 0.0)
        self.line, = self.ax.plot(self._fft_freqs, magnitude, color="#89b4fa", linewidth=1)

        # Marker for the generated tone frequency
        self.tone_marker = self.ax.axvline(
            x=self.prefs["frequency"], color="#f38ba8", linestyle="--", linewidth=1, alpha=0.7, label="Tone"
        )
        self.ax.legend(loc="upper right", facecolor="#1e1e2e", edgecolor="#444466",
                       labelcolor="white")

        # Fill under the curve for visibility
        self.fill = self.ax.fill_between(
            self._fft_freqs, 0, magnitude, color="#89b4fa", alpha=0.15,
        )

        # Peak frequency annotation
        self.peak_text = self.ax.text(
            0.5, 0.95, "", transform=self.ax.transAxes,
            ha="center", va="top", color="#a6e3a1", fontsize=11,
            fontweight="bold",
        )

        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

        # Smoothed magnitude for display (exponential moving average)
        self._smooth_db = np.full_like(self._fft_freqs, 0.0)

        # ── Pulse FFT state ──
        self._last_pulse_db = None          # cached FFT result to hold during gaps
        self._pulse_hold = 0                # frames to hold before next extraction
        self._pulse_corr = 0.0              # normalized correlation quality

        # ── FFT recording state ──
        self._fft_snapshots = []            # list of (timestamp_ms, magnitude_db)

        # ── Classifier state ──
        if self._clf_model is not None:
            lo, hi = self._clf_model["freq_range_hz"]
            self._clf_freq_mask = (
                (self._fft_freqs >= lo) & (self._fft_freqs <= hi))
        self._clf_history = collections.deque(maxlen=2)  # last N predictions
        self._clf_stable_label = None   # currently displayed (stable) label
        self._clf_stable_conf = 0.0
        self._clf_outlier_count = 0     # consecutive outlier frames
        self._clf_text = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            ha="left", va="top", color="#a6e3a1", fontsize=13,
            fontweight="bold", family="monospace",
        )

        # ── Pacman game setup ──
        if self.ax_pacman is not None:
            self._init_pacman()

        # ── Tron game setup ──
        if self.ax_tron is not None:
            self._init_tron()

        # ── Maze explorer setup ──
        if self.ax_maze is not None:
            self._init_maze()

        # ── World explorer setup ──
        if self.ax_world is not None:
            self._init_world()

        # ── Pong game setup ──
        if self.ax_pong is None:
            return
        self.ax_pong.set_facecolor("#1e1e2e")
        self.ax_pong.set_xlim(0, 100)
        self.ax_pong.set_ylim(0, 100)
        self.ax_pong.set_aspect("equal")
        self.ax_pong.set_xticks([])
        self.ax_pong.set_yticks([])
        for spine in self.ax_pong.spines.values():
            spine.set_color("#444466")
        self.ax_pong.set_title("PONG", color="white", fontsize=14, fontweight="bold")

        # Pong state
        self._pong_paddle_x = 50.0   # center of paddle (x)
        self._pong_paddle_w = 20.0   # paddle width
        self._pong_paddle_h = 2.5    # paddle height
        self._pong_paddle_speed = 3.0  # units per frame

        self._pong_ball_x = 50.0
        self._pong_ball_y = 50.0
        self._pong_ball_r = 1.5      # ball radius
        self._pong_ball_vx = 0.4
        self._pong_ball_vy = -0.5
        self._pong_ball_base_speed = 0.6
        self._pong_score = 0
        self._pong_hits = 0          # hits this rally, for speed increase

        # Pong artists
        self._pong_paddle = mpatches.FancyBboxPatch(
            (self._pong_paddle_x - self._pong_paddle_w / 2, 2),
            self._pong_paddle_w, self._pong_paddle_h,
            boxstyle="round,pad=0.5",
            facecolor="#89b4fa", edgecolor="#cdd6f4",
        )
        self.ax_pong.add_patch(self._pong_paddle)

        self._pong_ball_patch = mpatches.Circle(
            (self._pong_ball_x, self._pong_ball_y), self._pong_ball_r,
            facecolor="#f9e2af", edgecolor="#fab387",
        )
        self.ax_pong.add_patch(self._pong_ball_patch)

        # Top wall
        self._pong_top_wall = mpatches.Rectangle(
            (0, 97), 100, 3, facecolor="#585b70", edgecolor="#444466",
        )
        self.ax_pong.add_patch(self._pong_top_wall)

        # Score text
        self._pong_score_text = self.ax_pong.text(
            50, 90, "0", transform=self.ax_pong.transData,
            ha="center", va="top", color="#cdd6f4", fontsize=28,
            fontweight="bold", alpha=0.3,
        )

        # Label indicator
        self._pong_label_text = self.ax_pong.text(
            50, 1, "", transform=self.ax_pong.transData,
            ha="center", va="bottom", color="#585b70", fontsize=9,
        )

    # ── Pulse FFT helpers ────────────────────────────────────────
    def _get_reference_pulse(self):
        """Return a float32 numpy array of one clean pulse from current params."""
        with self.tone._lock:
            amp = self.tone.amplitude
            f_start = self.tone.pulse_f_start
            f_end = self.tone.pulse_f_end
            length_ms = self.tone.pulse_length_ms
        n = max(1, int(length_ms / 1000.0 * SAMPLE_RATE))
        idx = np.arange(n)
        t_norm = idx / n
        envelope = np.exp(-0.5 * ((t_norm - 0.5) / 0.15) ** 2)
        t_sec = idx / SAMPLE_RATE
        T = n / SAMPLE_RATE
        phase = 2 * np.pi * (f_start * t_sec + 0.5 * (f_end - f_start) * t_sec ** 2 / T)
        return (amp * envelope * np.sin(phase)).astype(np.float32)

# ── Callbacks ─────────────────────────────────────────────────
    def _on_freq_change(self, _val):
        freq = round(self.freq_var.get())
        self.freq_label.config(text=f"{freq:.0f} Hz")
        self.freq_entry.delete(0, tk.END)
        self.freq_entry.insert(0, f"{freq:.0f}")
        self.tone.set_frequency(freq)
        self.tone_marker.set_xdata([freq])

    def _on_freq_entry(self, _event):
        try:
            freq = float(self.freq_entry.get())
            freq = max(20, min(20000, freq))
        except ValueError:
            return
        self.freq_var.set(freq)
        self.freq_label.config(text=f"{freq:.0f} Hz")
        self.tone.set_frequency(freq)
        self.tone_marker.set_xdata([freq])

    def _on_vol_change(self, _val):
        amp = self.vol_var.get()
        self.vol_label.config(text=f"{amp * 100:.0f}%")
        self.tone.set_amplitude(amp)

    def _apply_pulse_params(self):
        try:
            f_start = int(self.pulse_f_start_entry.get())
            f_end = int(self.pulse_f_end_entry.get())
            length_ms = int(self.pulse_len_entry.get())
            gap_ms = int(self.pulse_gap_entry.get())
        except ValueError:
            return
        f_start = max(20, min(20000, f_start))
        f_end = max(20, min(20000, f_end))
        length_ms = max(1, length_ms)
        gap_ms = max(0, gap_ms)
        self.tone.set_pulse_params(f_start, f_end, length_ms, gap_ms)
        self._last_pulse_db = None
        self._pulse_hold = 0

    def _on_wave_change(self):
        # Swap the callback to produce different waveforms
        kind = self.wave_var.get()

        # Show/hide gaussian pulse controls
        if kind == "gaussian":
            self.gauss_frame.grid()
        else:
            self.gauss_frame.grid_remove()

        if kind == "gaussian":
            # Reset pulse position on waveform change
            self.tone._pulse_pos = 0

            def _callback(in_data, frame_count, time_info, status):
                with self.tone._lock:
                    amp = self.tone.amplitude
                    f_start = self.tone.pulse_f_start
                    f_end = self.tone.pulse_f_end
                    length_ms = self.tone.pulse_length_ms
                    gap_ms = self.tone.pulse_gap_ms

                pulse_samples = max(1, int(length_ms / 1000.0 * SAMPLE_RATE))
                gap_samples = int(gap_ms / 1000.0 * SAMPLE_RATE)
                cycle = max(1, pulse_samples + gap_samples)

                pos = self.tone._pulse_pos
                indices = np.arange(frame_count) + pos
                idx_in_cycle = indices % cycle
                in_pulse = idx_in_cycle < pulse_samples

                # Normalized time within pulse (0..1)
                t_norm = idx_in_cycle / pulse_samples
                # Gaussian envelope
                envelope = np.exp(-0.5 * ((t_norm - 0.5) / 0.15) ** 2)
                # Chirp phase: integral of linearly swept frequency
                t_sec = idx_in_cycle / SAMPLE_RATE
                T = pulse_samples / SAMPLE_RATE
                phase = 2 * np.pi * (f_start * t_sec + 0.5 * (f_end - f_start) * t_sec ** 2 / T)

                samples = np.where(in_pulse, amp * envelope * np.sin(phase), 0.0).astype(np.float32)

                self.tone._pulse_pos = pos + frame_count
                return (samples.tobytes(), pyaudio.paContinue)
        else:
            def _callback(in_data, frame_count, time_info, status):
                with self.tone._lock:
                    freq = self.tone.frequency
                    amp = self.tone.amplitude
                t = np.arange(frame_count) / SAMPLE_RATE
                phase = self.tone.phase
                angle = 2 * np.pi * freq * t + phase

                if kind == "sine":
                    samples = np.sin(angle)
                elif kind == "square":
                    samples = np.sign(np.sin(angle))
                elif kind == "sawtooth":
                    samples = 2 * (angle / (2 * np.pi) % 1) - 1
                elif kind == "triangle":
                    samples = 2 * np.abs(2 * (angle / (2 * np.pi) % 1) - 1) - 1
                else:
                    samples = np.sin(angle)

                samples = (amp * samples).astype(np.float32)
                self.tone.phase += 2 * np.pi * freq * frame_count / SAMPLE_RATE
                self.tone.phase %= 2 * np.pi
                return (samples.tobytes(), pyaudio.paContinue)

        self.tone._callback = _callback

    def _toggle_tone(self):
        if self.tone.playing:
            self.tone.stop()
            self.play_btn.config(text="▶  Play Tone")
        else:
            self._on_wave_change()  # ensure correct waveform callback
            self.tone.start()
            self.play_btn.config(text="⏹  Stop Tone")

    def _save_single_pulse(self, filepath):
        """Generate and save one clean pulse/cycle from current tone params."""
        kind = self.wave_var.get()
        with self.tone._lock:
            amp = self.tone.amplitude
        if kind == "gaussian":
            with self.tone._lock:
                f_start = self.tone.pulse_f_start
                f_end = self.tone.pulse_f_end
                length_ms = self.tone.pulse_length_ms
            n = max(1, int(length_ms / 1000.0 * SAMPLE_RATE))
            idx = np.arange(n)
            t_norm = idx / n
            envelope = np.exp(-0.5 * ((t_norm - 0.5) / 0.15) ** 2)
            t_sec = idx / SAMPLE_RATE
            T = n / SAMPLE_RATE
            phase = 2 * np.pi * (f_start * t_sec + 0.5 * (f_end - f_start) * t_sec ** 2 / T)
            samples = (amp * envelope * np.sin(phase)).astype(np.float32)
        else:
            with self.tone._lock:
                freq = self.tone.frequency
            n = int(SAMPLE_RATE / freq)
            t = np.arange(n) / SAMPLE_RATE
            angle = 2 * np.pi * freq * t
            if kind == "square":
                samples = np.sign(np.sin(angle))
            elif kind == "sawtooth":
                samples = 2 * (angle / (2 * np.pi) % 1) - 1
            elif kind == "triangle":
                samples = 2 * np.abs(2 * (angle / (2 * np.pi) % 1) - 1) - 1
            else:
                samples = np.sin(angle)
            samples = (amp * samples).astype(np.float32)
        self._save_wav(filepath, [samples.tobytes()])

    def _save_fft_snapshots(self, filepath):
        """Save accumulated FFT snapshots to CSV.

        Header row: timestamp_ms, then each frequency bin value.
        Each data row: elapsed ms since recording start, then dB values.
        """
        if not self._fft_snapshots:
            return
        freqs = self._fft_freqs
        with open(filepath, "w") as f:
            # Header: timestamp_ms, freq_0, freq_1, ...
            header = "timestamp_ms," + ",".join(f"{v:.1f}" for v in freqs)
            f.write(header + "\n")
            for ts_ms, db_arr in self._fft_snapshots:
                row = f"{ts_ms:.1f}," + ",".join(
                    f"{v:.2f}" for v in db_arr)
                f.write(row + "\n")

    def _save_wav(self, filepath, frames):
        """Write float32 frames to a 16-bit WAV file."""
        raw = b"".join(frames) if frames else b""
        float_samples = np.frombuffer(raw, dtype=np.float32) if raw else np.array([], dtype=np.float32)
        int_samples = np.clip(float_samples * 32767, -32768, 32767).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(int_samples.tobytes())

    def _toggle_record(self):
        if self.mic._recording:
            self._rec_timer_id = None
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
            os.makedirs(rec_dir, exist_ok=True)

            # Save mic recording
            mic_path = os.path.join(rec_dir, f"mic_{stamp}.wav")
            self.mic.stop_recording(mic_path)

            # Save a single transmitted pulse from current params
            tx_path = os.path.join(rec_dir, f"tx_{stamp}.wav")
            self._save_single_pulse(tx_path)

            # Save accumulated FFT snapshots
            fft_path = os.path.join(rec_dir, f"fft_{stamp}.csv")
            self._save_fft_snapshots(fft_path)

            self.rec_btn.config(text="●  Record")
            # Copy FFT filename to clipboard
            fft_basename = os.path.basename(fft_path)
            self.clipboard_clear()
            self.clipboard_append(fft_basename)
            self.rec_btn.config(text=f"Copied: {fft_basename}")
            for label, path in [("Mic", mic_path), ("Tx", tx_path),
                                ("FFT", fft_path)]:
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    if label == "FFT":
                        n = len(self._fft_snapshots)
                        print(f"{label}: {path} ({size} bytes, {n} snapshots)")
                    else:
                        duration = (size - 44) / (SAMPLE_RATE * 2)
                        print(f"{label}: {path} ({size} bytes, {duration:.1f}s)")
        else:
            self._rec_start = datetime.datetime.now()
            self._fft_snapshots = []
            self.mic.start_recording()
            self._update_rec_timer()

    def _update_rec_timer(self):
        if not self.mic._recording:
            return
        elapsed = datetime.datetime.now() - self._rec_start
        total_secs = int(elapsed.total_seconds())
        mins, secs = divmod(total_secs, 60)
        self.rec_btn.config(text=f"⏹  {mins:02d}:{secs:02d}")
        self._rec_timer_id = self.after(200, self._update_rec_timer)

    # ── Pong game logic ──────────────────────────────────────────
    def _reset_pong_ball(self):
        """Respawn ball at center with a random downward direction."""
        self._pong_ball_x = 50.0
        self._pong_ball_y = 60.0
        self._pong_hits = 0
        speed = self._pong_ball_base_speed
        angle = np.random.uniform(np.pi * 1.1, np.pi * 1.9)  # downward
        self._pong_ball_vx = speed * np.cos(angle)
        self._pong_ball_vy = speed * np.sin(angle)

    def _update_pong(self):
        """Step the Pong simulation once per frame."""
        # -- Move paddle based on classifier output --
        label = self._clf_stable_label
        if label == "left":
            self._pong_paddle_x -= self._pong_paddle_speed
        elif label == "right":
            self._pong_paddle_x += self._pong_paddle_speed
        # Clamp paddle to bounds
        half_w = self._pong_paddle_w / 2
        self._pong_paddle_x = max(half_w, min(100 - half_w, self._pong_paddle_x))

        # -- Move ball --
        self._pong_ball_x += self._pong_ball_vx
        self._pong_ball_y += self._pong_ball_vy
        r = self._pong_ball_r

        # Left/right wall bounce
        if self._pong_ball_x - r <= 0:
            self._pong_ball_x = r
            self._pong_ball_vx = abs(self._pong_ball_vx)
        elif self._pong_ball_x + r >= 100:
            self._pong_ball_x = 100 - r
            self._pong_ball_vx = -abs(self._pong_ball_vx)

        # Top wall bounce (wall at y=97)
        if self._pong_ball_y + r >= 97:
            self._pong_ball_y = 97 - r
            self._pong_ball_vy = -abs(self._pong_ball_vy)

        # Paddle collision
        paddle_top = 2 + self._pong_paddle_h
        paddle_left = self._pong_paddle_x - self._pong_paddle_w / 2
        paddle_right = self._pong_paddle_x + self._pong_paddle_w / 2
        if (self._pong_ball_vy < 0
                and self._pong_ball_y - r <= paddle_top
                and self._pong_ball_y + r >= 2
                and paddle_left - r <= self._pong_ball_x <= paddle_right + r):
            self._pong_ball_y = paddle_top + r
            self._pong_hits += 1
            self._pong_score += 1
            # Increase speed slightly with each hit
            speed = self._pong_ball_base_speed + self._pong_hits * 0.03
            # Reflect with angle based on where ball hit paddle
            offset = (self._pong_ball_x - self._pong_paddle_x) / (self._pong_paddle_w / 2)
            offset = max(-0.9, min(0.9, offset))
            angle = np.pi / 2 + offset * np.pi / 3  # spread the angle
            self._pong_ball_vx = speed * np.cos(angle)
            self._pong_ball_vy = abs(speed * np.sin(angle))

        # Ball missed (below paddle) -> reset
        if self._pong_ball_y - r < 0:
            self._reset_pong_ball()

        # -- Update artists --
        self._pong_paddle.set_x(self._pong_paddle_x - self._pong_paddle_w / 2)
        self._pong_paddle.set_y(2)
        self._pong_ball_patch.set_center(
            (self._pong_ball_x, self._pong_ball_y))
        self._pong_score_text.set_text(str(self._pong_score))
        if label in ("left", "right"):
            self._pong_label_text.set_text(label)
            self._pong_label_text.set_color("#a6e3a1")
        else:
            self._pong_label_text.set_text(label or "")
            self._pong_label_text.set_color("#585b70")

    # ── Pacman game logic ────────────────────────────────────────
    _PAC_DIR_VECTORS = {
        "right": (1, 0), "left": (-1, 0),
        "up": (0, -1), "down": (0, 1),
    }
    _PAC_DIR_ANGLES = {
        "right": (30, 330), "left": (210, 150),
        "up": (120, 60), "down": (300, 240),
    }

    def _init_pacman(self):
        """Set up the Pacman maze, dots, and Pacman patch."""
        MS = 15
        self._pac_maze_size = MS

        self.ax_pacman.set_facecolor("#1e1e2e")
        self.ax_pacman.set_xlim(0, MS)
        self.ax_pacman.set_ylim(0, MS)
        self.ax_pacman.set_aspect("equal")
        self.ax_pacman.set_xticks([])
        self.ax_pacman.set_yticks([])
        for spine in self.ax_pacman.spines.values():
            spine.set_color("#444466")
        self.ax_pacman.set_title(
            "PACMAN", color="white", fontsize=14, fontweight="bold")

        # Initialize maze: place dots on all open cells
        self._pac_maze = _PACMAN_MAZE_TEMPLATE.copy()
        self._pac_maze[self._pac_maze == 0] = 2

        # Pacman starts at bottom-center corridor
        self._pac_row = 13
        self._pac_col = 7
        self._pac_maze[self._pac_row, self._pac_col] = 0

        # Draw wall rectangles
        for r in range(MS):
            for c in range(MS):
                if _PACMAN_MAZE_TEMPLATE[r, c] == 1:
                    dy = MS - 1 - r
                    rect = mpatches.Rectangle(
                        (c, dy), 1, 1,
                        facecolor="#313244", edgecolor="#45475a",
                        linewidth=0.5,
                    )
                    self.ax_pacman.add_patch(rect)

        # Draw dot circles
        self._pac_dot_patches = {}
        for r in range(MS):
            for c in range(MS):
                if self._pac_maze[r, c] == 2:
                    dy = MS - 1 - r
                    dot = mpatches.Circle(
                        (c + 0.5, dy + 0.5), 0.1,
                        facecolor="#cdd6f4", edgecolor="none",
                    )
                    self.ax_pacman.add_patch(dot)
                    self._pac_dot_patches[(r, c)] = dot

        # Create Pacman wedge (facing right initially)
        px = self._pac_col + 0.5
        py = MS - 0.5 - self._pac_row
        self._pac_patch = mpatches.Wedge(
            (px, py), 0.45, 30, 330,
            facecolor="#f9e2af", edgecolor="none",
        )
        self.ax_pacman.add_patch(self._pac_patch)

        # Movement state
        self._pac_dir = "right"
        self._pac_desired_dir = "right"
        self._pac_move_timer = 0
        self._pac_move_interval = 6  # frames between grid moves
        self._pac_score = 0

        # Score display (overlaid on top wall)
        self._pac_score_text = self.ax_pacman.text(
            MS / 2, MS - 0.5, "Score: 0",
            ha="center", va="center", color="#cdd6f4", fontsize=14,
            fontweight="bold", alpha=0.3,
        )

        # Direction label
        self._pac_label_text = self.ax_pacman.text(
            0.5, 0.02, "", transform=self.ax_pacman.transAxes,
            ha="center", va="bottom", color="#585b70", fontsize=9,
        )

        # Arrow key bindings
        self.bind("<Up>", lambda e: setattr(self, '_pac_desired_dir', 'up'))
        self.bind("<Down>", lambda e: setattr(self, '_pac_desired_dir', 'down'))
        self.bind("<Left>", lambda e: setattr(self, '_pac_desired_dir', 'left'))
        self.bind("<Right>", lambda e: setattr(self, '_pac_desired_dir', 'right'))

    def _reset_pacman_maze(self):
        """Repopulate dots after all are eaten."""
        MS = self._pac_maze_size
        self._pac_maze = _PACMAN_MAZE_TEMPLATE.copy()
        self._pac_maze[self._pac_maze == 0] = 2

        # Reset Pacman position
        self._pac_row = 13
        self._pac_col = 7
        self._pac_maze[self._pac_row, self._pac_col] = 0

        # Recreate dot patches
        for r in range(MS):
            for c in range(MS):
                if self._pac_maze[r, c] == 2:
                    dy = MS - 1 - r
                    dot = mpatches.Circle(
                        (c + 0.5, dy + 0.5), 0.1,
                        facecolor="#cdd6f4", edgecolor="none",
                    )
                    self.ax_pacman.add_patch(dot)
                    self._pac_dot_patches[(r, c)] = dot

    def _update_pacman(self):
        """Step the Pacman simulation once per frame."""
        # Map classifier labels to directions
        label = self._clf_stable_label
        label_map = {
            "left": "left", "right": "right",
            "top": "up", "bottom": "down",
        }
        if label in label_map:
            self._pac_desired_dir = label_map[label]

        # Movement tick
        self._pac_move_timer += 1
        if self._pac_move_timer >= self._pac_move_interval:
            self._pac_move_timer = 0
            MS = self._pac_maze_size

            # Try switching to desired direction
            dcol, drow = self._PAC_DIR_VECTORS[self._pac_desired_dir]
            nr, nc = self._pac_row + drow, self._pac_col + dcol
            if (0 <= nr < MS and 0 <= nc < MS
                    and self._pac_maze[nr, nc] != 1):
                self._pac_dir = self._pac_desired_dir

            # Move in current direction
            dcol, drow = self._PAC_DIR_VECTORS[self._pac_dir]
            nr, nc = self._pac_row + drow, self._pac_col + dcol
            if (0 <= nr < MS and 0 <= nc < MS
                    and self._pac_maze[nr, nc] != 1):
                self._pac_row = nr
                self._pac_col = nc
                # Eat dot
                if self._pac_maze[nr, nc] == 2:
                    self._pac_maze[nr, nc] = 0
                    patch = self._pac_dot_patches.pop((nr, nc), None)
                    if patch is not None:
                        patch.remove()
                    self._pac_score += 1

        # Update Pacman position and mouth direction
        MS = self._pac_maze_size
        px = self._pac_col + 0.5
        py = MS - 0.5 - self._pac_row
        self._pac_patch.set_center((px, py))
        t1, t2 = self._PAC_DIR_ANGLES[self._pac_dir]
        self._pac_patch.set_theta1(t1)
        self._pac_patch.set_theta2(t2)

        # Update score
        self._pac_score_text.set_text(f"Score: {self._pac_score}")

        # Update label indicator
        if label in label_map:
            self._pac_label_text.set_text(label_map[label])
            self._pac_label_text.set_color("#a6e3a1")
        else:
            self._pac_label_text.set_text(label or "")
            self._pac_label_text.set_color("#585b70")

        # Win condition: all dots eaten -> reset maze
        if not self._pac_dot_patches:
            self._reset_pacman_maze()

    # ── Tron game logic ──────────────────────────────────────────
    _TRON_SIZE = 40
    _TRON_DIR_VECTORS = {
        "right": (1, 0), "left": (-1, 0),
        "up": (0, 1), "down": (0, -1),
    }
    _TRON_OPPOSITES = {
        "right": "left", "left": "right",
        "up": "down", "down": "up",
    }

    def _init_tron(self):
        """Set up the Tron light-cycle arena."""
        S = self._TRON_SIZE
        self.ax_tron.set_facecolor("#0a0a1a")
        self.ax_tron.set_xlim(0, S)
        self.ax_tron.set_ylim(0, S)
        self.ax_tron.set_aspect("equal")
        self.ax_tron.set_xticks([])
        self.ax_tron.set_yticks([])
        for spine in self.ax_tron.spines.values():
            spine.set_color("#333366")
        self.ax_tron.set_title(
            "TRON", color="#66ffff", fontsize=14, fontweight="bold")

        # Grid overlay for visual style
        for i in range(S + 1):
            self.ax_tron.axhline(i, color="#111133", linewidth=0.3)
            self.ax_tron.axvline(i, color="#111133", linewidth=0.3)

        # Trail stored as a 2D array: 0=empty, 1=trail
        self._tron_grid = np.zeros((S, S), dtype=np.int8)

        # Trail patches dict: (row, col) -> Rectangle
        self._tron_trail_patches = {}

        # Player starts left-center, moving right
        self._tron_row = S // 2
        self._tron_col = S // 4
        self._tron_dir = "right"
        self._tron_desired_dir = "right"
        self._tron_alive = True
        self._tron_score = 0  # number of cells survived
        self._tron_best = 0

        # Mark starting cell
        self._tron_grid[self._tron_row, self._tron_col] = 1
        self._tron_add_trail(self._tron_row, self._tron_col)

        # Head marker
        hx = self._tron_col + 0.5
        hy = self._tron_row + 0.5
        self._tron_head = mpatches.Rectangle(
            (hx - 0.5, hy - 0.5), 1, 1,
            facecolor="#66ffff", edgecolor="#aaffff", linewidth=1.5,
        )
        self.ax_tron.add_patch(self._tron_head)

        # Movement timing
        self._tron_move_timer = 0
        self._tron_move_interval = 8  # frames between moves

        # Score / status text
        self._tron_score_text = self.ax_tron.text(
            0.5, 0.97, "0", transform=self.ax_tron.transAxes,
            ha="center", va="top", color="#66ffff", fontsize=20,
            fontweight="bold", alpha=0.3,
        )
        self._tron_status_text = self.ax_tron.text(
            0.5, 0.5, "", transform=self.ax_tron.transAxes,
            ha="center", va="center", color="#ff4444", fontsize=16,
            fontweight="bold", alpha=0.0,
        )
        self._tron_label_text = self.ax_tron.text(
            0.5, 0.02, "", transform=self.ax_tron.transAxes,
            ha="center", va="bottom", color="#585b70", fontsize=9,
        )

        # Death animation counter (-1 = alive, 0+ = frames since death)
        self._tron_death_timer = -1
        self._tron_death_delay = 60  # frames before auto-restart

        # Arrow key bindings
        self.bind("<Up>", lambda e: setattr(self, '_tron_desired_dir', 'up'))
        self.bind("<Down>", lambda e: setattr(self, '_tron_desired_dir', 'down'))
        self.bind("<Left>", lambda e: setattr(self, '_tron_desired_dir', 'left'))
        self.bind("<Right>", lambda e: setattr(self, '_tron_desired_dir', 'right'))

    def _tron_add_trail(self, row, col):
        """Add a trail rectangle at the given grid cell."""
        rect = mpatches.Rectangle(
            (col, row), 1, 1,
            facecolor="#1a3a5c", edgecolor="#2255aa", linewidth=0.5,
        )
        self.ax_tron.add_patch(rect)
        self._tron_trail_patches[(row, col)] = rect

    def _reset_tron(self):
        """Clear the arena and restart."""
        S = self._TRON_SIZE
        self._tron_grid[:] = 0

        # Remove all trail patches
        for patch in self._tron_trail_patches.values():
            patch.remove()
        self._tron_trail_patches.clear()

        # Reset player position
        self._tron_row = S // 2
        self._tron_col = S // 4
        self._tron_dir = "right"
        self._tron_desired_dir = "right"
        self._tron_alive = True
        self._tron_score = 0
        self._tron_death_timer = -1

        # Mark starting cell
        self._tron_grid[self._tron_row, self._tron_col] = 1
        self._tron_add_trail(self._tron_row, self._tron_col)

        # Reset head
        hx = self._tron_col + 0.5
        hy = self._tron_row + 0.5
        self._tron_head.set_xy((hx - 0.5, hy - 0.5))
        self._tron_head.set_facecolor("#66ffff")

        # Clear status
        self._tron_status_text.set_alpha(0.0)

    def _update_tron(self):
        """Step the Tron simulation once per frame."""
        S = self._TRON_SIZE

        # Handle death state
        if self._tron_death_timer >= 0:
            self._tron_death_timer += 1
            # Blink the status text
            alpha = 0.6 + 0.4 * np.sin(self._tron_death_timer * 0.3)
            self._tron_status_text.set_alpha(alpha)
            if self._tron_death_timer >= self._tron_death_delay:
                self._reset_tron()
            return

        # Map classifier labels to directions
        label = self._clf_stable_label
        label_map = {
            "left": "left", "right": "right",
            "top": "up", "bottom": "down",
        }
        if label in label_map:
            self._tron_desired_dir = label_map[label]

        # Movement tick
        self._tron_move_timer += 1
        if self._tron_move_timer >= self._tron_move_interval:
            self._tron_move_timer = 0

            # Accept desired direction if it's not a 180-degree reversal
            desired = self._tron_desired_dir
            if desired != self._TRON_OPPOSITES.get(self._tron_dir):
                self._tron_dir = desired

            # Compute next cell
            dcol, drow = self._TRON_DIR_VECTORS[self._tron_dir]
            nr = self._tron_row + drow
            nc = self._tron_col + dcol

            # Check collision: wall or own trail
            if (nr < 0 or nr >= S or nc < 0 or nc >= S
                    or self._tron_grid[nr, nc] == 1):
                # Death
                self._tron_alive = False
                self._tron_death_timer = 0
                self._tron_head.set_facecolor("#ff4444")
                if self._tron_score > self._tron_best:
                    self._tron_best = self._tron_score
                self._tron_status_text.set_text(
                    f"CRASH\nbest: {self._tron_best}")
                self._tron_status_text.set_alpha(0.8)
                return

            # Move
            self._tron_row = nr
            self._tron_col = nc
            self._tron_grid[nr, nc] = 1
            self._tron_add_trail(nr, nc)
            self._tron_score += 1

        # Update head position
        hx = self._tron_col + 0.5
        hy = self._tron_row + 0.5
        self._tron_head.set_xy((hx - 0.5, hy - 0.5))

        # Update score
        self._tron_score_text.set_text(str(self._tron_score))

        # Update label indicator
        if label in label_map:
            self._tron_label_text.set_text(label_map[label])
            self._tron_label_text.set_color("#66ffff")
        else:
            self._tron_label_text.set_text(label or "")
            self._tron_label_text.set_color("#585b70")

    # ── Maze explorer logic ────────────────────────────────────────
    _MAZE_CELL = 2  # each maze cell = 2 grid units (wall between cells)

    @staticmethod
    def _generate_maze(rows, cols):
        """Generate a maze using recursive backtracking (DFS).

        Returns a 2D numpy array where the grid is (2*rows+1) x (2*cols+1).
        0 = passage, 1 = wall.  Cells are at odd coordinates, walls at even.
        """
        H = 2 * rows + 1
        W = 2 * cols + 1
        grid = np.ones((H, W), dtype=np.int8)

        # Carve cells and passages via DFS
        stack = [(0, 0)]
        visited = {(0, 0)}
        grid[1, 1] = 0  # start cell

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if neighbors:
                nr, nc = neighbors[np.random.randint(len(neighbors))]
                visited.add((nr, nc))
                # Knock down wall between current and neighbor
                wr = 1 + r + nr  # wall row in grid coords
                wc = 1 + c + nc  # wall col in grid coords
                grid[wr, wc] = 0
                grid[1 + 2 * nr, 1 + 2 * nc] = 0  # neighbor cell
                stack.append((nr, nc))
            else:
                stack.pop()

        return grid

    def _init_maze(self):
        """Set up the maze explorer."""
        self._maze_rows = 10
        self._maze_cols = 10
        self._maze_grid = self._generate_maze(self._maze_rows, self._maze_cols)
        H, W = self._maze_grid.shape  # 21 x 21

        self.ax_maze.set_facecolor("#1e1e2e")
        self.ax_maze.set_xlim(0, W)
        self.ax_maze.set_ylim(0, H)
        self.ax_maze.set_aspect("equal")
        self.ax_maze.set_xticks([])
        self.ax_maze.set_yticks([])
        for spine in self.ax_maze.spines.values():
            spine.set_color("#444466")
        self.ax_maze.set_title(
            "MAZE", color="white", fontsize=14, fontweight="bold")

        # Player position in grid coords (start at top-left cell)
        self._maze_r = 1
        self._maze_c = 1

        # Exit at bottom-right cell
        self._maze_exit_r = H - 2
        self._maze_exit_c = W - 2

        # Visited cells tracking
        self._maze_visited = set()
        self._maze_visited.add((self._maze_r, self._maze_c))
        self._maze_visited_patches = {}

        # Draw everything
        self._maze_wall_patches = []
        self._maze_draw_walls()
        self._maze_draw_exit()

        # Player marker
        py = H - 1 - self._maze_r
        self._maze_player = mpatches.Circle(
            (self._maze_c + 0.5, py + 0.5), 0.35,
            facecolor="#89b4fa", edgecolor="#b4befe", linewidth=1.5,
        )
        self.ax_maze.add_patch(self._maze_player)

        # Movement state
        self._maze_dir = "right"
        self._maze_desired_dir = "right"
        self._maze_move_timer = 0
        self._maze_move_interval = 6
        self._maze_count = 0  # mazes completed

        # Status text
        self._maze_status_text = self.ax_maze.text(
            0.5, 0.97, "", transform=self.ax_maze.transAxes,
            ha="center", va="top", color="#a6e3a1", fontsize=12,
            fontweight="bold", alpha=0.4,
        )
        self._maze_label_text = self.ax_maze.text(
            0.5, 0.02, "", transform=self.ax_maze.transAxes,
            ha="center", va="bottom", color="#585b70", fontsize=9,
        )

        # Arrow key bindings
        self.bind("<Up>", lambda e: setattr(self, '_maze_desired_dir', 'up'))
        self.bind("<Down>", lambda e: setattr(self, '_maze_desired_dir', 'down'))
        self.bind("<Left>", lambda e: setattr(self, '_maze_desired_dir', 'left'))
        self.bind("<Right>", lambda e: setattr(self, '_maze_desired_dir', 'right'))

    def _maze_draw_walls(self):
        """Draw wall rectangles for the current maze."""
        H, W = self._maze_grid.shape
        for r in range(H):
            for c in range(W):
                if self._maze_grid[r, c] == 1:
                    dy = H - 1 - r
                    rect = mpatches.Rectangle(
                        (c, dy), 1, 1,
                        facecolor="#313244", edgecolor="#3b3b54",
                        linewidth=0.3,
                    )
                    self.ax_maze.add_patch(rect)
                    self._maze_wall_patches.append(rect)

    def _maze_draw_exit(self):
        """Draw the exit marker."""
        H = self._maze_grid.shape[0]
        ey = H - 1 - self._maze_exit_r
        self._maze_exit_patch = mpatches.Rectangle(
            (self._maze_exit_c, ey), 1, 1,
            facecolor="#a6e3a1", edgecolor="none", alpha=0.4,
        )
        self.ax_maze.add_patch(self._maze_exit_patch)
        # Small star/marker in the exit cell
        self._maze_exit_marker = mpatches.RegularPolygon(
            (self._maze_exit_c + 0.5, ey + 0.5), numVertices=5, radius=0.3,
            facecolor="#a6e3a1", edgecolor="none", alpha=0.8,
        )
        self.ax_maze.add_patch(self._maze_exit_marker)

    def _maze_draw_visited(self):
        """Clear any existing visited-cell overlays."""
        for p in self._maze_visited_patches.values():
            p.remove()
        self._maze_visited_patches.clear()

    def _maze_new(self):
        """Generate a new maze and redraw."""
        H_old = self._maze_grid.shape[0]

        # Remove old patches
        for p in self._maze_wall_patches:
            p.remove()
        self._maze_wall_patches.clear()
        self._maze_exit_patch.remove()
        self._maze_exit_marker.remove()
        self._maze_draw_visited()

        # Generate new maze
        self._maze_grid = self._generate_maze(self._maze_rows, self._maze_cols)
        H, W = self._maze_grid.shape

        # Reset player to top-left
        self._maze_r = 1
        self._maze_c = 1
        self._maze_visited = {(1, 1)}

        # Redraw
        self._maze_draw_walls()
        self._maze_draw_exit()

        # Update player position
        py = H - 1 - self._maze_r
        self._maze_player.set_center((self._maze_c + 0.5, py + 0.5))

        self._maze_count += 1

    def _update_maze(self):
        """Step the maze explorer once per frame."""
        H, W = self._maze_grid.shape

        # Map classifier labels to directions
        label = self._clf_stable_label
        label_map = {
            "left": "left", "right": "right",
            "top": "up", "bottom": "down",
        }
        if label in label_map:
            self._maze_desired_dir = label_map[label]

        # Movement tick
        self._maze_move_timer += 1
        if self._maze_move_timer >= self._maze_move_interval:
            self._maze_move_timer = 0

            # Direction vectors (y-flipped for display)
            dvecs = {
                "right": (0, 1), "left": (0, -1),
                "up": (-1, 0), "down": (1, 0),
            }

            # Try desired direction
            dr, dc = dvecs[self._maze_desired_dir]
            nr, nc = self._maze_r + dr, self._maze_c + dc
            if (0 <= nr < H and 0 <= nc < W
                    and self._maze_grid[nr, nc] == 0):
                self._maze_dir = self._maze_desired_dir

            # Move in current direction
            dr, dc = dvecs[self._maze_dir]
            nr, nc = self._maze_r + dr, self._maze_c + dc
            if (0 <= nr < H and 0 <= nc < W
                    and self._maze_grid[nr, nc] == 0):
                self._maze_r = nr
                self._maze_c = nc

                # Mark visited with a subtle trail
                if (nr, nc) not in self._maze_visited:
                    self._maze_visited.add((nr, nc))
                    dy = H - 1 - nr
                    trail = mpatches.Rectangle(
                        (nc, dy), 1, 1,
                        facecolor="#89b4fa", edgecolor="none", alpha=0.1,
                    )
                    self.ax_maze.add_patch(trail)
                    self._maze_visited_patches[(nr, nc)] = trail

        # Update player position
        py = H - 1 - self._maze_r
        self._maze_player.set_center((self._maze_c + 0.5, py + 0.5))

        # Check if reached exit
        if (self._maze_r == self._maze_exit_r
                and self._maze_c == self._maze_exit_c):
            self._maze_new()

        # Update status
        if self._maze_count > 0:
            self._maze_status_text.set_text(
                f"solved: {self._maze_count}")

        # Update label indicator
        if label in label_map:
            self._maze_label_text.set_text(label_map[label])
            self._maze_label_text.set_color("#a6e3a1")
        else:
            self._maze_label_text.set_text(label or "")
            self._maze_label_text.set_color("#585b70")

    # ── World explorer logic ────────────────────────────────────────
    _WORLD_SIZE = 200
    _WORLD_VIEW = 25
    _WORLD_HALF = 12
    _WORLD_TERRAIN_COLORS = np.array([
        [0.05, 0.12, 0.35],  # 0 deep water
        [0.10, 0.25, 0.55],  # 1 shallow water
        [0.60, 0.55, 0.35],  # 2 sand
        [0.30, 0.50, 0.15],  # 3 light grass
        [0.15, 0.35, 0.10],  # 4 grass
        [0.08, 0.22, 0.06],  # 5 forest
        [0.05, 0.15, 0.04],  # 6 dense forest
        [0.45, 0.40, 0.35],  # 7 rock
        [0.60, 0.58, 0.55],  # 8 mountain
        [0.85, 0.85, 0.90],  # 9 snow
    ])
    _WORLD_WALKABLE = np.array(
        [False, False, True, True, True, True, True, True, True, True])

    def _init_world(self):
        """Generate terrain and set up the world explorer."""
        S = self._WORLD_SIZE
        V = self._WORLD_VIEW

        rng = np.random.RandomState(42)

        # Large-scale elevation (continents)
        raw1 = rng.rand(S, S)
        k1 = np.ones(12) / 12
        e1 = np.apply_along_axis(
            lambda r: np.convolve(r, k1, mode='same'), 1, raw1)
        e1 = np.apply_along_axis(
            lambda c: np.convolve(c, k1, mode='same'), 0, e1)

        # Medium-scale detail (hills)
        raw2 = rng.rand(S, S)
        k2 = np.ones(5) / 5
        e2 = np.apply_along_axis(
            lambda r: np.convolve(r, k2, mode='same'), 1, raw2)
        e2 = np.apply_along_axis(
            lambda c: np.convolve(c, k2, mode='same'), 0, e2)

        elev = 0.65 * e1 + 0.35 * e2
        # Normalize to full 0-1 range
        elev = (elev - elev.min()) / (elev.max() - elev.min())

        # Threshold into 10 terrain types
        terrain = np.full((S, S), 4, dtype=np.int8)
        terrain[elev < 0.15] = 0   # deep water
        terrain[(elev >= 0.15) & (elev < 0.30)] = 1  # shallow water
        terrain[(elev >= 0.30) & (elev < 0.35)] = 2  # sand/beach
        terrain[(elev >= 0.35) & (elev < 0.45)] = 3  # light grass
        terrain[(elev >= 0.45) & (elev < 0.55)] = 4  # grass
        terrain[(elev >= 0.55) & (elev < 0.65)] = 5  # forest
        terrain[(elev >= 0.65) & (elev < 0.75)] = 6  # dense forest
        terrain[(elev >= 0.75) & (elev < 0.85)] = 7  # rock
        terrain[(elev >= 0.85) & (elev < 0.93)] = 8  # mountain
        terrain[elev >= 0.93] = 9  # snow
        self._world_terrain = terrain

        # Axes setup
        self.ax_world.set_facecolor("#1e1e2e")
        self.ax_world.set_xticks([])
        self.ax_world.set_yticks([])
        for spine in self.ax_world.spines.values():
            spine.set_color("#444466")
        self.ax_world.set_title(
            "WORLD", color="white", fontsize=14, fontweight="bold")

        # Player starts at center on walkable ground
        self._world_r = S // 2
        self._world_c = S // 2
        self._world_terrain[self._world_r, self._world_c] = 4  # grass

        # Initial viewport
        rgb = self._world_get_viewport()
        self._world_img = self.ax_world.imshow(
            rgb, interpolation='nearest', aspect='equal')

        # Player marker at viewport center
        H = self._WORLD_HALF
        self._world_player = mpatches.Circle(
            (H, H), 0.4,
            facecolor="#f9e2af", edgecolor="#fab387",
            linewidth=1.5, zorder=5,
        )
        self.ax_world.add_patch(self._world_player)

        # Coordinate text
        self._world_coord_text = self.ax_world.text(
            0.5, 0.97, f"({self._world_c}, {self._world_r})",
            transform=self.ax_world.transAxes,
            ha="center", va="top", color="white", fontsize=9, alpha=0.4,
        )
        self._world_label_text = self.ax_world.text(
            0.5, 0.02, "", transform=self.ax_world.transAxes,
            ha="center", va="bottom", color="#585b70", fontsize=9,
        )

        # Movement state
        self._world_dir = "right"
        self._world_desired_dir = "right"
        self._world_move_timer = 0
        self._world_move_interval = 4

        # Arrow key bindings
        self.bind("<Up>",
                  lambda e: setattr(self, '_world_desired_dir', 'up'))
        self.bind("<Down>",
                  lambda e: setattr(self, '_world_desired_dir', 'down'))
        self.bind("<Left>",
                  lambda e: setattr(self, '_world_desired_dir', 'left'))
        self.bind("<Right>",
                  lambda e: setattr(self, '_world_desired_dir', 'right'))

    def _world_get_viewport(self):
        """Extract the visible viewport as an RGB array."""
        V = self._WORLD_VIEW
        H = self._WORLD_HALF
        S = self._WORLD_SIZE
        rows = (np.arange(V) + self._world_r - H) % S
        cols = (np.arange(V) + self._world_c - H) % S
        terrain_view = self._world_terrain[np.ix_(rows, cols)]
        return self._WORLD_TERRAIN_COLORS[terrain_view]

    def _update_world(self):
        """Step the world explorer once per frame."""
        S = self._WORLD_SIZE
        dvecs = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }

        label = self._clf_stable_label
        label_map = {
            "left": "left", "right": "right",
            "top": "up", "bottom": "down",
        }
        if label in label_map:
            self._world_desired_dir = label_map[label]

        moved = False
        self._world_move_timer += 1
        if self._world_move_timer >= self._world_move_interval:
            self._world_move_timer = 0

            # Try desired direction
            dr, dc = dvecs[self._world_desired_dir]
            nr = (self._world_r + dr) % S
            nc = (self._world_c + dc) % S
            if self._WORLD_WALKABLE[self._world_terrain[nr, nc]]:
                self._world_dir = self._world_desired_dir

            # Move in current direction
            dr, dc = dvecs[self._world_dir]
            nr = (self._world_r + dr) % S
            nc = (self._world_c + dc) % S
            if self._world_terrain[nr, nc] != 0:
                self._world_r = nr
                self._world_c = nc
                moved = True

        if moved:
            self._world_img.set_data(self._world_get_viewport())
            self._world_coord_text.set_text(
                f"({self._world_c}, {self._world_r})")

        # Label indicator
        if label in label_map:
            self._world_label_text.set_text(label_map[label])
            self._world_label_text.set_color("#a6e3a1")
        else:
            self._world_label_text.set_text(label or "")
            self._world_label_text.set_color("#585b70")

    # ── FFT update loop ────────────────────────────────────────────
    _hann = np.hanning(FFT_SIZE).astype(np.float32)
    SMOOTH = 0.3  # 0 = no smoothing, 1 = frozen

    def _update_plot(self):
        samples = self.mic.get_snapshot()

        if samples is not None:
            # Update time-domain waveform from dedicated buffer
            tail = self.mic.get_time_samples(self._time_samples)
            self.time_line.set_xdata(self._time_x)
            self.time_line.set_ydata(tail)
            if self._time_amp_fixed is not None:
                self.ax_time.set_ylim(-self._time_amp_fixed, self._time_amp_fixed)
            else:
                peak_amp = np.max(np.abs(tail))
                if peak_amp > 0:
                    margin = peak_amp * 1.2
                    self.ax_time.set_ylim(-margin, margin)
                else:
                    self.ax_time.set_ylim(-0.01, 0.01)

            # ── Pulse FFT mode ──
            use_pulse_fft = (self.pulse_fft_var.get()
                             and self.wave_var.get() == "gaussian")
            magnitude_db = None

            if use_pulse_fft:
                try:
                    # Get pulse timing params
                    with self.tone._lock:
                        length_ms = self.tone.pulse_length_ms
                        gap_ms = self.tone.pulse_gap_ms
                    pulse_n = max(1, int(length_ms / 1000.0 * SAMPLE_RATE))
                    gap_n = int(gap_ms / 1000.0 * SAMPLE_RATE)
                    cycle = max(1, pulse_n + gap_n)

                    # Hold between pulses - skip correlation during cooldown
                    if self._pulse_hold > 0:
                        self._pulse_hold -= 1
                        if self._last_pulse_db is not None:
                            magnitude_db = self._last_pulse_db
                    else:
                        # Grab ~2 cycles of mic data
                        cal_samples = cycle * 2
                        mic_buf = self.mic.get_time_samples(cal_samples)
                        ref = self._get_reference_pulse()
                        n = len(mic_buf)
                        m = len(ref)
                        if n >= m:
                            # FFT cross-correlation
                            fft_n = 1
                            while fft_n < n + m - 1:
                                fft_n <<= 1
                            fft_mic = np.fft.rfft(mic_buf, fft_n)
                            fft_ref = np.fft.rfft(ref, fft_n)
                            corr = np.fft.irfft(
                                fft_mic * np.conj(fft_ref), fft_n)
                            valid = corr[:n - m + 1]

                            # Find the latest correlation peak
                            peak_idx = np.argmax(np.abs(valid))
                            peak_corr = np.abs(valid[peak_idx])
                            # Normalized correlation: 0 = no match, 1 = perfect
                            ref_energy = np.sqrt(np.sum(ref ** 2))
                            mic_seg = mic_buf[peak_idx:peak_idx + m]
                            mic_energy = np.sqrt(np.sum(mic_seg ** 2))
                            norm_corr = (peak_corr / max(ref_energy * mic_energy, 1e-12))
                            self._pulse_corr = norm_corr
                            candidate = peak_idx
                            if candidate + pulse_n <= n:
                                pulse_data = mic_buf[
                                    candidate:candidate + pulse_n]
                                win = np.hanning(pulse_n).astype(np.float32)
                                windowed_pulse = pulse_data * win
                                fft_len = max(FFT_SIZE, pulse_n)
                                padded = np.zeros(fft_len, dtype=np.float32)
                                padded[:pulse_n] = windowed_pulse
                                spectrum = np.fft.rfft(padded)
                                mag = np.abs(spectrum) / (pulse_n / 2)
                                magnitude_db_full = 20 * np.log10(
                                    np.maximum(mag, 1e-10)) + 140
                                if fft_len != FFT_SIZE:
                                    freqs_full = np.fft.rfftfreq(
                                        fft_len, 1.0 / SAMPLE_RATE)
                                    magnitude_db = np.interp(
                                        self._fft_freqs, freqs_full,
                                        magnitude_db_full)
                                else:
                                    magnitude_db = magnitude_db_full
                                self._last_pulse_db = magnitude_db
                                # Classify the new pulse
                                if (self.classify_var.get()
                                        and self._clf_model is not None):
                                    try:
                                        # Skip if correlation with reference
                                        # pulse is too weak (no real pulse)
                                        if False and self._pulse_corr < 0.3:
                                            self._clf_history.clear()
                                            self._clf_stable_label = None
                                            self._clf_stable_conf = 0.0
                                            self._clf_outlier_count = 0
                                            self._clf_text.set_text("(noise)")
                                            self._clf_text.set_color("#585b70")
                                            raise ValueError("no pulse")
                                        band_db = magnitude_db[
                                            self._clf_freq_mask]
                                        # Outlier check: is this spectrum
                                        # close enough to any known class?
                                        centroids = self._clf_model.get(
                                            "class_centroids")
                                        cutoffs = self._clf_model.get(
                                            "class_outlier_cutoffs")
                                        is_outlier = False
                                        if False and centroids and cutoffs:
                                            for lbl in centroids:
                                                d = np.linalg.norm(
                                                    band_db - centroids[lbl])
                                                if d <= cutoffs[lbl]:
                                                    is_outlier = False
                                                    break
                                        else:
                                            is_outlier = False
                                        if is_outlier:
                                            self._clf_outlier_count += 1
                                            if self._clf_outlier_count >= 3:
                                                self._clf_history.clear()
                                                self._clf_stable_label = None
                                                self._clf_stable_conf = 0.0
                                                self._clf_text.set_text(
                                                    "(outlier)")
                                                self._clf_text.set_color(
                                                    "#585b70")
                                            raise ValueError("outlier")
                                        self._clf_outlier_count = 0
                                        feat = band_db.reshape(1, -1)
                                        feat_sc = self._clf_model[
                                            "scaler"].transform(feat)
                                        clf = self._clf_model["classifier"]
                                        proba = clf.predict_proba(feat_sc)[0]
                                        best_idx = np.argmax(proba)
                                        label = clf.classes_[best_idx]
                                        conf = proba[best_idx]
                                        self._clf_history.append((label, conf))
                                        # Require all samples in buffer to agree
                                        hist = self._clf_history
                                        if (len(hist) == hist.maxlen
                                                and all(h[0] == hist[0][0]
                                                        for h in hist)):
                                            avg_conf = np.mean(
                                                [h[1] for h in hist])
                                            thresh = self._clf_model[
                                                "confidence_thresholds"]
                                            min_conf = thresh[label]["min"]
                                            color = ("#a6e3a1"
                                                     if avg_conf >= min_conf
                                                     else "#f9e2af")
                                            self._clf_stable_label = label
                                            self._clf_stable_conf = avg_conf
                                            self._clf_text.set_text(
                                                f"{label}  {avg_conf:.0%}")
                                            self._clf_text.set_color(color)
                                        elif self._clf_stable_label is not None:
                                            # Disagreement -- keep showing last
                                            # stable result but dim it
                                            self._clf_text.set_text(
                                                f"{self._clf_stable_label}  "
                                                f"{self._clf_stable_conf:.0%}")
                                            self._clf_text.set_color("#585b70")
                                    except Exception:
                                        pass
                                # Record snapshot on new extraction
                                if self.mic._recording:
                                    elapsed = (
                                        datetime.datetime.now()
                                        - self._rec_start
                                    ).total_seconds() * 1000
                                    self._fft_snapshots.append(
                                        (elapsed, magnitude_db.copy()))
                                # Hold for one cycle before next extraction
                                hold_frames = max(
                                    1, int(cycle / SAMPLE_RATE * 30) - 2)
                                self._pulse_hold = hold_frames

                        # Use cached if extraction didn't produce a result
                        if magnitude_db is None and \
                                self._last_pulse_db is not None:
                            magnitude_db = self._last_pulse_db
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    magnitude_db = None

            # Reset smoothing on mode transitions
            if use_pulse_fft != getattr(self, '_was_pulse_fft', False):
                self._smooth_db[:] = 0.0
                self._was_pulse_fft = use_pulse_fft

            # Clear classification text when not actively classifying
            if not (use_pulse_fft and self.classify_var.get()):
                self._clf_text.set_text("")
                if self._clf_history:
                    self._clf_history.clear()
                    self._clf_stable_label = None
                    self._clf_stable_conf = 0.0
                    self._clf_outlier_count = 0

            if magnitude_db is None:
                # Normal full-buffer FFT (also fallback on pulse FFT error)
                windowed = samples * self._hann
                spectrum = np.fft.rfft(windowed)
                magnitude = np.abs(spectrum) / (FFT_SIZE / 2)
                magnitude_db = 20 * np.log10(
                    np.maximum(magnitude, 1e-10)) + 140

            # Exponential moving average for smoother visuals
            self._smooth_db = (self.SMOOTH * self._smooth_db
                               + (1 - self.SMOOTH) * magnitude_db)

            # Update line
            self.line.set_ydata(self._smooth_db)

            # Update filled area
            self.fill.remove()
            self.fill = self.ax.fill_between(
                self._fft_freqs, 0, self._smooth_db,
                color="#89b4fa", alpha=0.15,
            )

            # Show peak frequency
            peak_idx = np.argmax(self._smooth_db)
            peak_freq = self._fft_freqs[peak_idx]
            peak_db = self._smooth_db[peak_idx]
            self.peak_text.set_text(
                f"Peak: {peak_freq:.1f} Hz  ({peak_db:.1f} dB)"
            )

        # Update Pong every frame (even when no new audio samples)
        if self.ax_pong is not None:
            self._update_pong()
        if self.ax_pacman is not None:
            self._update_pacman()
        if self.ax_tron is not None:
            self._update_tron()
        if self.ax_maze is not None:
            self._update_maze()
        if self.ax_world is not None:
            self._update_world()

        self.canvas.draw_idle()

        if not self._closing:
            self._after_id = self.after(33, self._update_plot)  # ~30 fps

    def _on_close(self):
        self._closing = True
        # Cancel any pending after callback
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        # Save preferences
        try:
            xmin = int(self.xmin_entry.get())
            xmax = int(self.xmax_entry.get())
        except ValueError:
            xmin, xmax = 20, 20000
        # Read gaussian pulse params from entries
        try:
            pulse_f_start = int(self.pulse_f_start_entry.get())
        except ValueError:
            pulse_f_start = self.prefs["pulse_f_start"]
        try:
            pulse_f_end = int(self.pulse_f_end_entry.get())
        except ValueError:
            pulse_f_end = self.prefs["pulse_f_end"]
        try:
            pulse_length_ms = int(self.pulse_len_entry.get())
        except ValueError:
            pulse_length_ms = self.prefs["pulse_length_ms"]
        try:
            pulse_gap_ms = int(self.pulse_gap_entry.get())
        except ValueError:
            pulse_gap_ms = self.prefs["pulse_gap_ms"]

        try:
            time_window_ms = float(self.time_win_entry.get())
        except ValueError:
            time_window_ms = self.prefs["time_window_ms"]

        self.prefs.update({
            "x_min": xmin,
            "x_max": xmax,
            "frequency": self.freq_var.get(),
            "volume": self.vol_var.get(),
            "waveform": self.wave_var.get(),
            "pulse_f_start": pulse_f_start,
            "pulse_f_end": pulse_f_end,
            "pulse_length_ms": pulse_length_ms,
            "pulse_gap_ms": pulse_gap_ms,
            "time_window_ms": time_window_ms,
            "time_amplitude": self.time_amp_entry.get().strip(),
            "pulse_fft": self.pulse_fft_var.get(),
            "classify": self.classify_var.get(),
        })
        save_prefs(self.prefs)
        # Stop audio streams
        self.tone.stop()
        self.mic.stop()
        pa.terminate()
        # Close matplotlib figure
        plt.close(self.fig)
        self.destroy()
        self.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tone Generator & FFT Spectrum Analyzer")
    parser.add_argument("--game", choices=["pong", "pacman", "tron", "maze", "world"], default=None,
                        help="Enable a game panel (e.g. --game pong)")
    args = parser.parse_args()
    app = App(game=args.game)
    app.mainloop()
