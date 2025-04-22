---
title: 'VibratoScope: A Python Toolkit for High-Resolution Vibrato Analysis in Singing Voice'
tags:
  - vibrato
  - singing voice
  - acoustic analysis
  - pitch
  - voice science
authors:
  - name: Tiago Lima Bicalho Cruz
    orcid: 0000-0002-8355-5436
    affiliation: 1
affiliations:
  - name: Federal University of Minas Gerais (UFMG)
    index: 1
date: 2025-04-22
---

# Summary

VibratoScope is an open-source Python toolkit designed for high-resolution analysis of vibrato in singing voice recordings. It extracts key acoustic features such as vibrato rate, extent (in cents), jitter, and sample entropy. VibratoScope includes a graphical user interface (GUI) for region-based inspection and a batch-processing mode for unattended analysis. Outputs are saved in CSV and PNG formats and include raw and smoothed vibrato traces, coefficient of variation time series, and entropy bar charts.

The software is platform-independent and free from proprietary dependencies, supporting reproducible voice analysis in research and pedagogy. It has been validated with both synthetic and real singing voice materials and shows strong agreement with previously established vibrato metrics.

# Statement of Need

Despite the centrality of vibrato in vocal performance, current tools often present barriers due to licensing, platform limitations, or lack of batch functionality. Proprietary software such as VoceVista Pro and BioVoice provide visual vibrato overlays but restrict access to processing pipelines and output formats. Open-source scripts (e.g., EML Praat) offer transparent algorithms but lack a unified user experience or scalability.

VibratoScope bridges these gaps by combining transparency, usability, and extensibility. It supports both interactive and automated workflows, aligning with the needs of voice researchers, teachers, and developers aiming to study vibrato quantitatively.

# Installation

Install Python 3.9 or higher. Then run:

```bash
git clone https://github.com/tiagolbc/vibratoscope.git
cd vibratools
pip install -r requirements.txt
python VibratoScope.py
```

# Example Use

To analyze a .wav file manually:
1. Open the GUI.
2. Load the file and select a region of interest.
3. Click “Run Analysis” to extract vibrato metrics.

To use batch mode:
1. Select multiple .wav files.
2. Run the analysis without region selection.
3. Results are saved in structured folders with CSV and PNG output.

# Acknowledgements

This software was developed as part of doctoral research at the Federal University of Minas Gerais (UFMG) and refined with feedback from collaborators in voice science and music technology. Special thanks to reviewers of the Pan-European Voice Conference and The Voice Foundation for helpful validation comments.

# References

References are included in the separate `paper.bib` file.