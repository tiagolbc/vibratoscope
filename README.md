# 🎵 VibratoScope

**VibratoScope** is a Python toolkit for high-resolution analysis of vibrato in the singing voice.

It extracts vibrato rate, extent (in cents), jitter, shimmer, sample entropy, and other regularity metrics from sustained vowels or melodic phrases. A user-friendly GUI is included for region selection and visual feedback, and batch processing is supported for multiple recordings.

![VibratoScope GUI](figures/gui.png)

---

## 🧠 Features

- GUI for region selection and interactive spectrogram navigation.
- Batch processing of `.wav` files with automatic export of results.
- Multiple pitch extraction methods:
  - YIN (`librosa.pyin`)
  - Praat autocorrelation
  - Harmonic Product Spectrum (HPS)
  - REAPER (Robust Epoch and Pitch Estimator)
  - **SFEEDS** (Spectral F0 Estimation using Energy Distribution Smoothing) – adapted from the original Praat implementation
- Bandpass filtering (default 3–9 Hz) for vibrato isolation
- Extraction of:
  - Vibrato rate (Hz)
  - Vibrato extent (cents)
  - Jitter (cycle-to-cycle frequency variability)
  - Shimmer (amplitude variability)
  - Sample Entropy
  - Coefficient of Variation
- Automatic visualization:
  - Pitch traces
  - Vibrato cycles
  - Entropy and extent barplots
- CSV export for region-based and full-file summaries
- Cross-platform (Windows, macOS, Linux)

---

## 🛠️ Installation

Requires **Python 3.9+**

```bash
git clone https://github.com/tiagolbc/vibratoscope.git
cd vibratoscope
pip install -r requirements.txt
```

---

## 🚀 Running VibratoScope

To launch the GUI:

```bash
python run.py
```

All functional modules are located under the `src/` directory.

---

## 📂 Example Dataset

The `examples/` folder includes synthetic vowel samples with known vibrato parameters (e.g., 6.0 Hz rate, 0.5 semitone extent).

Each test case includes:

- `.wav` file
- `.csv` results
- Pitch and vibrato analysis figures

These examples are used in validation and reproducibility. See `docs/paper.md` for citation.

---

## 📖 Citation

If you use this toolkit in research, please cite:

**Cruz, T. L. B. (2025). VibratoScope: A Python Toolkit for High-Resolution Vibrato Analysis in Singing Voice.**  
*Zenodo.* https://doi.org/10.5281/zenodo.15262079

Or use the “Cite this repository” button on GitHub for BibTeX.

---

## 📃 License

MIT License — see [`LICENSE`](LICENSE) for terms.
