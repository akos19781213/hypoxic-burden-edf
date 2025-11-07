# Hypoxic Burden Calculator

**Open-source tool to compute hypoxic burden from PSG EDF files**  
Based on:  
> Azarbarzin A, et al. *European Heart Journal* (2019) – DOI: [10.1093/eurheartj/ehy624](https://doi.org/10.1093/eurheartj/ehy624)

Computes:
- **AHI** (from airflow)
- **ODI** (3% or 4% desaturation)
- **Hypoxic Burden** (%min/h) — **per sleep stage + overall**
- Deep learning sleep staging (YASA) or rule-based fallback
- Interactive Azarbarzin-style desaturation curves

---

## Live Demo
[Try it online](https://hypoxic-burden-edf.streamlit.app)

---

## Installation

```bash
git clone https://github.com/Apolloplectic/hypoxic-burden-edf.git
cd hypoxic-burden-edf
pip install -r requirements.txt
streamlit run app.py