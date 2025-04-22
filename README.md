# VibratoScope

**VibratoScope** is a Python toolkit for the analysis of vibrato in the singing voice. It provides automatic extraction of vibrato rate, extent (in cents), jitter, sample entropy, and generates summary plots. A user-friendly GUI is included for selection of analysis regions and batch processing.

![screenshot](docs/screenshot.png) <!-- optional -->

---

## 🧠 Features

- GUI for region selection and interactive spectrograms.
- Batch processing of multiple WAV files.
- Extraction of pitch using YIN algorithm (`librosa.pyin`)
- Bandpass filtering (3–9 Hz) for vibrato isolation.
- Calculation of:
  - Vibrato rate (Hz)
  - Extent (cents)
  - Jitter metrics
  - Sample Entropy
  - Coefficient of variability
- Visualizations saved automatically as PNGs.
- CSV export of region-based and global results.

---

## 🛠️ Installation

Requires Python 3.9 or higher.

1. Clone the repository:
```bash
git clone https://github.com/tiagolbc/vibratoscope.git
cd vibratoscope

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

python VibratoScope.py

## 📄 Citation

If you use this software, please cite:

Tiago Lima Bicalho Cruz. (2025). *VibratoScope: A Python Toolkit for High-Resolution Vibrato Analysis in Singing Voice* (v1.0.1). Journal of Open Source Software. https://doi.org/10.5281/zenodo.15262079
