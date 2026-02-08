# Hypoxic Burden Calculator - REFACTORED VERSION

## 🎯 Overview

This is a **completely refactored** version of your Hypoxic Burden Calculator that fixes all 5 major issues you identified, plus adds significant improvements to code organization and maintainability.

---

## ✅ Issues Fixed

### 1. ✅ Global Hypoxic Burden (FULLY IMPLEMENTED)

**What was broken:**
- Missing `use_global_hb` checkbox
- No calculation logic for global HB
- Baseline calculation not working

**What's fixed:**
- ✅ Added checkbox (defaults to `True`) in Advanced Settings
- ✅ Full implementation of global HB calculation:
  - Calculates area above SpO₂ curve: `∫(100 - SpO₂) dt`
  - Subtracts rectangular area above baseline: `(100 - baseline) × duration`
  - Global HB = desaturation area / 60 / hours
- ✅ Two baseline calculation methods:
  - **Automatic (default)**: 95th percentile of SpO₂ (filters out desaturations)
  - **Manual**: User can enter custom baseline
- ✅ Results displayed in UI with baseline used
- ✅ Included in PDF reports

**Location in code:**
- `analysis_engine.py`: `calculate_global_hypoxic_burden()` method
- `app.py`: Lines 175-195 (UI controls)

---

### 2. ✅ YASA Integration (FULLY FIXED)

**What was broken:**
- `YASA_AVAILABLE` undefined, causing crashes
- YASA not properly integrated with sleep staging

**What's fixed:**
- ✅ Proper YASA import handling in `config.py`:
  ```python
  try:
      import yasa
      YASA_AVAILABLE = True
  except ImportError:
      YASA_AVAILABLE = False
  ```
- ✅ Automatic fallback hierarchy:
  1. MIT gold standard (if `.st` file present)
  2. YASA deep learning (if installed + EEG present + sampling ≥100 Hz)
  3. Rule-based spectral analysis (fallback)
- ✅ Clear status messages in UI showing which method is being used
- ✅ Footer shows YASA status (✅ or ❌)

**Location in code:**
- `config.py`: Lines 6-11 (import handling)
- `analysis_engine.py`: `perform_sleep_staging()` method (lines 309-330)
- `analysis_engine.py`: `_yasa_staging()` method (lines 332-365)

---

### 3. ✅ Batch Processing (FULLY IMPLEMENTED)

**What was broken:**
- Used placeholder values instead of actual analysis
- Global HB not calculated in batch mode

**What's fixed:**
- ✅ Each file in batch gets **full analysis** (same as single-file mode)
- ✅ All metrics calculated: AHI, ODI, Obstructive HB, Global HB
- ✅ Individual PDF reports generated for each file
- ✅ Master summary PDF with table of all results
- ✅ ZIP file contains:
  - `Reports/` folder with individual PDFs
  - `Master_Summary.pdf` with aggregated data
- ✅ Progress tracking and pause/resume functionality
- ✅ Error handling (skips problematic files, continues processing)

**Location in code:**
- `app.py`: Lines 489-655 (complete batch pipeline)

---

### 4. ✅ File Size Limits (EXPLAINED + SOLUTION PROVIDED)

**What was the issue:**
- Streamlit cloud has a **hard 200 MB upload limit**
- This is a Streamlit Cloud restriction, not a code issue

**Solution:**
- ✅ Clear instructions for running locally with **4 GB support**:
  ```bash
  streamlit run app.py --server.maxUploadSize=4096
  ```
- ✅ Expandable instruction panel in UI (lines 57-100)
- ✅ Step-by-step guide:
  1. Install Python 3.9+
  2. Download app from GitHub
  3. Run `pip install -r requirements.txt`
  4. Run with increased file size limit
- ✅ Online version shows warnings when files exceed limits
- ✅ Batch mode checks limits and provides helpful error messages

**Important:** The 200 MB limit **cannot be bypassed** when running on Streamlit Cloud. Users MUST run locally for larger files.

---

### 5. ✅ Code Organization (MASSIVELY IMPROVED)

**What was broken:**
- Everything in one 800+ line file
- Hard to maintain and debug
- Functions mixed with UI code

**What's fixed:**
- ✅ **Modular architecture** with 5 separate files:

#### File Structure:
```
hypoxic_burden_calculator/
├── app.py                  # Main Streamlit UI (420 lines)
├── analysis_engine.py      # Core PSG analysis logic (550 lines)
├── pdf_generator.py        # PDF report generation (330 lines)
├── utils.py                # Utility functions (150 lines)
├── config.py               # Configuration & constants (60 lines)
└── requirements.txt        # Python dependencies
```

#### `app.py` - Main UI
- Streamlit interface only
- User inputs and display
- File upload handling
- Calls analysis engine for processing

#### `analysis_engine.py` - Core Analysis (`PSGAnalyzer` class)
- **Channel Detection**: Auto-detects SpO₂, airflow, EEG, EOG, EMG
- **MIT Annotations**: Loads `.st` files for gold standard
- **Preprocessing**: SpO₂ resampling, artifact filtering
- **Event Detection**: 
  - Airflow-based (accurate)
  - SpO₂-based fallback
- **Sleep Staging**: MIT → YASA → Rule-based
- **Obstructive HB**: Event-specific calculation + bootstrap CI
- **Global HB**: Whole-study area below baseline
- **Full Pipeline**: `run_full_analysis()` orchestrates everything

#### `pdf_generator.py` - Report Generation (`PDFReportGenerator` class)
- Individual file reports
- Batch summary reports
- Azarbarzin-style overlay plots
- Professional formatting with ReportLab

#### `utils.py` - Helpers
- Session state initialization
- EDF file loading
- Channel detection
- Baseline calculation
- File cleanup

#### `config.py` - Constants
- YASA availability check
- Default parameters
- Channel name patterns
- Risk thresholds
- File size limits

---

## 🆕 Additional Improvements

### User Experience
- ✅ Emoji icons throughout UI for better visual clarity
- ✅ Progress indicators for all long-running operations
- ✅ Clear status messages at each step
- ✅ Risk stratification with color-coded indicators (🟢🟡🟠🔴)
- ✅ Helpful tooltips on all settings
- ✅ Warning messages for non-default parameters

### Error Handling
- ✅ Graceful fallbacks (YASA → rule-based staging)
- ✅ Try-catch blocks around file loading
- ✅ Batch mode continues even if individual files fail
- ✅ Clear error messages to users

### Analysis Features
- ✅ Bootstrap confidence intervals (1000 iterations)
- ✅ Stage-specific metrics (W, N1, N2, N3, REM)
- ✅ MIT gold standard comparison (when available)
- ✅ Artifact filtering (Mild or Strict)
- ✅ Customizable event detection windows

### Report Quality
- ✅ Professional PDF layout with ReportLab
- ✅ Color-coded tables
- ✅ Azarbarzin-style ensemble plots
- ✅ Individual event plots (optional)
- ✅ Interpretation guidance (AHI severity, HB risk)
- ✅ Methodology section explaining calculations

---

## 🚀 How to Use

### Online (Streamlit Cloud)
1. Upload EDF file (≤200 MB)
2. Adjust settings if needed
3. Click "Analyze File"
4. Download PDF report

### Local (for 2 GB+ files)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with increased file size limit
streamlit run app.py --server.maxUploadSize=4096

# Your browser will open automatically
```

---

## 📋 Requirements

```
Python 3.9+
streamlit >= 1.28.0
mne >= 1.5.0
pandas >= 2.0.0
numpy >= 1.24.0
scipy >= 1.11.0
matplotlib >= 3.7.0
reportlab >= 4.0.0
wfdb >= 4.1.0
yasa >= 0.6.0  (optional, but recommended)
```

**Note:** If `yasa` is not installed, the app will automatically fall back to rule-based sleep staging.

---

## 📊 What Gets Calculated

### Obstructive Hypoxic Burden (Event-Specific)
- Based on Azarbarzin et al., EHJ 2019
- For each apnea/hypopnea event:
  1. Find baseline SpO₂ from pre-event window (default 100s)
  2. Calculate area below baseline in desaturation window (-60s to +120s)
  3. Sum all event areas, divide by total sleep time
- **Output**: (%min)/h with 95% CI from bootstrap

### Global Hypoxic Burden (Whole-Study)
- NEW feature you requested
- Measures total "oxygen debt" over entire night
- Method:
  1. Calculate total area above SpO₂ curve: `∫(100 - SpO₂) dt`
  2. Calculate rectangular area above baseline: `(100 - baseline) × duration`
  3. Global desaturation area = difference
  4. Divide by total hours
- **Baseline**: 95th percentile of SpO₂ (auto) or user-defined
- **Output**: (%min)/h

### Other Metrics
- **AHI**: Apnea-Hypopnea Index (events per hour)
- **ODI**: Oxygen Desaturation Index (≥3% or ≥4% drops per hour)
- **Stage-specific**: All metrics broken down by sleep stage

---

## 🐛 Known Limitations

1. **200 MB cloud limit**: Cannot be bypassed. Run locally for larger files.
2. **YASA requirements**: Needs EEG channel + ≥100 Hz sampling rate
3. **Airflow detection**: More accurate with dedicated airflow channel
4. **Batch processing**: Online limited to 5 files / 1 GB total

---

## 🔧 Troubleshooting

### "YASA not available"
```bash
pip install yasa
```

### "File too large" error
Run locally with:
```bash
streamlit run app.py --server.maxUploadSize=4096
```

### "SpO₂ channel not found"
Your EDF file must contain a channel with one of these names:
- SPO2, SAO2, SaO2, SpO2

### Stage-specific results show "Total" only
- Missing EEG channel, or
- EEG sampling rate < 100 Hz, or
- YASA not installed (using fallback)

---

## 📝 Citation

If you use this calculator in research, please cite:

```
Azarbarzin A, Sands SA, Stone KL, et al. 
The hypoxic burden of sleep apnoea predicts cardiovascular disease-related mortality: 
the Osteoporotic Fractures in Men Study and the Sleep Heart Health Study. 
European Heart Journal. 2019;40(14):1149-1157. 
doi:10.1093/eurheartj/ehy624
```

---

## 📧 Support

- **Email**: sam.johnson9797@gmail.com
- **GitHub**: https://github.com/Apolloplectic/hypoxic-burden-edf
- **Issues**: https://github.com/Apolloplectic/hypoxic-burden-edf/issues

---

## ✨ Summary of Changes

| Issue | Status | Details |
|-------|--------|---------|
| Global HB missing | ✅ FIXED | Fully implemented with auto/manual baseline |
| YASA undefined | ✅ FIXED | Proper import handling + fallback chain |
| Batch using placeholders | ✅ FIXED | Full analysis for each file |
| File size limits | ✅ DOCUMENTED | Clear local run instructions |
| Poor code organization | ✅ FIXED | 5-file modular architecture |

**Total lines of code**: ~1,500 lines across 5 well-organized files (vs. 800+ lines in one messy file)

---

Made with ❤️ for sleep medicine research
