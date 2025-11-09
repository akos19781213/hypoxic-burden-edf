import streamlit as st
import pandas as pd
import numpy as np
import mne
from scipy.integrate import trapezoid as trapz
import matplotlib.pyplot as plt
import os
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# --------------------------------------------------------------
# YASA: Deep Learning Sleep Staging
# --------------------------------------------------------------
try:
    import yasa
    YASA_AVAILABLE = True
except ImportError:
    YASA_AVAILABLE = False

st.set_page_config(page_title="Hypoxic Burden Calculator", layout="centered")
st.title("Hypoxic Burden Calculator")
st.markdown("""
**Upload PSG EDF file** → get the hypoxic burden in (%min)/h with **95% CI**.  
Based on:  
> Azarbarzin A, et al. *European Heart Journal* (2019) – DOI: [10.1093/eurheartj/ehy624](https://doi.org/10.1093/eurheartj/ehy624)
""")

# --------------------------------------------------------------
# FILE UPLOADER — 2 GB SUPPORT (online limited to 200 MB)
# --------------------------------------------------------------
edf_file = st.file_uploader(
    "Upload PSG EDF file",
    type=["edf"],
    help="Warning: Online version limited to 200 MB. For 2 GB+ files, run locally (see below)."
)

# === LOCAL RUN INSTRUCTIONS (FOR >200 MB FILES) ===
with st.expander("File too large? Run locally (2 GB+ support) — no coding needed!", expanded=False):
    st.markdown("""
    ### **How to Run This App on Your Computer (2 GB+ Files)**
    **No programming experience required. Takes 5 minutes.**

    ---
    
    #### **Step 1: Download the App (1-Click)**
    """)
    st.download_button(
        label="Download Hypoxic Burden Calculator (Windows/Mac)",
        data=open("hypoxic_burden_local.zip", "rb").read() if os.path.exists("hypoxic_burden_local.zip") else b"",
        file_name="Hypoxic_Burden_Calculator.zip",
        mime="application/zip",
        help="Includes app, installer, and instructions"
    )
    st.markdown("""
    > **Note**: If the button is disabled, download from GitHub:  
    > https://github.com/Apolloplectic/hypoxic-burden-edf/releases

    ---
    
    #### **Step 2: Unzip & Run**
    1. **Double-click** the downloaded `.zip` file
    2. Open the folder → double-click:
       - **Windows**: `Run_Calculator.bat`
       - **Mac**: `Run_Calculator.command`

    The app will open in your browser automatically.

    ---
    
    #### **Step 3: Upload Your Large EDF File**
    - Drag and drop your **2 GB+ PSG file**
    - Wait 1–3 minutes
    - Get **AHI, ODI, Hypoxic Burden with 95% CI**

    ---
    
    #### **No Internet? No Problem.**
    Works **offline** after first run.

    ---
    
    **Need help?** Email: `sam.johnson9797@gmail.com`  
    **Source Code**: [GitHub](https://github.com/Apolloplectic/hypoxic-burden-edf)
    """)

# --------------------------------------------------------------
# PROCESSING (unchanged)
# --------------------------------------------------------------
if edf_file is not None:
    with st.spinner(f"Loading {edf_file.name} ({edf_file.size / 1e9:.2f} GB)..."):
        temp_path = "temp_upload.edf"
        with open(temp_path, "wb") as f:
            f.write(edf_file.getbuffer())
    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
    st.success("EDF loaded!")

    # === CHANNEL DETECTION ===
    spo2_ch = next((ch for ch in raw.ch_names if any(n in ch.upper() for n in ['SPO2', 'SAO2'])), None)
    flow_ch = next((ch for ch in raw.ch_names if any(n in ch.upper() for n in ['AIRFLOW', 'FLOW'])), None)
    eeg_ch = next((ch for ch in raw.ch_names if 'EEG' in ch.upper()), None)
    eog_ch = next((ch for ch in raw.ch_names if 'EOG' in ch.upper()), None)
    emg_ch = next((ch for ch in raw.ch_names if 'EMG' in ch.upper()), None)

    if not spo2_ch:
        st.error("SpO₂ channel required.")
        st.stop()

    st.write(f"**SpO₂:** `{spo2_ch}` | **Airflow:** `{flow_ch or 'Not found'}`")

    # === ADVANCED SETTINGS ===
    with st.expander("Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            pre_event_sec = st.selectbox("Pre-event baseline (s)", [60, 100, 120], index=1)
            desat_start_sec = st.selectbox("Desat start before end (s)", [30, 60, 90], index=1)
        with col2:
            desat_end_sec = st.selectbox("Desat end after end (s)", [120, 180, 240], index=0)
            artifact_filter = st.selectbox("SpO₂ artifact filter", ["Off", "Mild (10%/s)", "Strict (5%/s)"], index=0)
        scoring_rule = st.selectbox("Scoring Rule (AHI + ODI)", ["3% (AASM)", "4% (Legacy)"], index=0)
        desat_threshold = 3 if "3%" in scoring_rule else 4

    # === WARNINGS ===
    if pre_event_sec != 100: st.warning("Pre-event window ≠ 100s (Azarbarzin default).")
    if desat_start_sec != 60 or desat_end_sec != 120: st.warning("Desat window ≠ -60/+120s (Azarbarzin default).")
    if "4%" in scoring_rule: st.warning("Using 4% rule (non-AASM).")
    if artifact_filter != "Off": st.warning("Artifact filter enabled.")
    if not flow_ch: st.warning("No airflow → crude AHI.")

    # === ANALYZE BUTTON WITH PROGRESS BAR ===
    if st.button("Analyze File", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # --------------------------------------------------------------
        # STEP 1: Resample SpO₂
        # --------------------------------------------------------------
        status_text.text("Step 1/6: Resampling SpO₂ to 1 Hz...")
        spo2_sig, spo2_times = raw[spo2_ch]
        df_spo2 = pd.DataFrame({"time": spo2_times.flatten(), "spo2": spo2_sig.flatten()})
        if raw.info['sfreq'] != 1:
            st.info(f"Resampling SpO₂ from {raw.info['sfreq']:.1f} Hz → 1 Hz")
            df_spo2['time'] = pd.to_datetime(df_spo2['time'], unit='s')
            df_spo2 = df_spo2.set_index('time').resample('1S').mean().interpolate().reset_index()
            df_spo2['time'] = (df_spo2['time'] - df_spo2['time'].iloc[0]).dt.total_seconds()
        progress_bar.progress(0.15)

        # --------------------------------------------------------------
        # STEP 2: Artifact Filter
        # --------------------------------------------------------------
        status_text.text("Step 2/6: Applying artifact filter...")
        if artifact_filter != "Off":
            max_rate = 10 if artifact_filter == "Mild (10%/s)" else 5
            df_spo2['rate'] = df_spo2['spo2'].diff().abs()
            df_spo2['artifact'] = df_spo2['rate'] > max_rate
            removed = df_spo2['artifact'].sum()
            st.info(f"Removed {removed} points > {max_rate}%/s")
        progress_bar.progress(0.30)

        # --------------------------------------------------------------
        # STEP 3: ODI Events
        # --------------------------------------------------------------
        status_text.text("Step 3/6: Detecting ODI events...")
        df_spo2['spo2_next'] = df_spo2['spo2'].shift(-10)
        df_spo2['desat'] = (df_spo2['spo2'].diff() <= -desat_threshold) & \
                           (df_spo2['spo2_next'] >= df_spo2['spo2'] + (desat_threshold - 1))
        odi_events = df_spo2[df_spo2['desat']].copy()
        progress_bar.progress(0.40)

        # --------------------------------------------------------------
        # STEP 4: Airflow Events
        # --------------------------------------------------------------
        status_text.text("Step 4/6: Detecting apnea/hypopnea events...")
        if flow_ch:
            st.info("Resampling airflow to 10 Hz")
            flow_sig, flow_times = raw[flow_ch]
            df_flow = pd.DataFrame({"time": flow_times.flatten(), "flow": flow_sig.flatten()})
            df_flow['time'] = pd.to_datetime(df_flow['time'], unit='s')
            df_flow = df_flow.set_index('time').resample('0.1S').mean().interpolate().reset_index()
            df_flow['time'] = (df_flow['time'] - df_flow['time'].iloc[0]).dt.total_seconds()
            flow = df_flow['flow'].values
            t = df_flow['time'].values
            peak = np.percentile(np.abs(flow), 95) or 1
            flow_norm = flow / peak
            window = int(30 / 0.1)
            baseline = pd.Series(flow_norm).rolling(window, center=True, min_periods=1).median().values
            reduction = 1 - (flow_norm / (baseline + 1e-6))
            in_event = reduction >= 0.30
            diff = np.diff(np.concatenate(([False], in_event, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            events = [{"start": t[s], "end": t[e-1]} for s, e in zip(starts, ends) if t[e-1] - t[s] >= 10]

            valid_events = []
            for ev in events:
                end_t = ev['end']
                win = df_spo2[(df_spo2['time'] >= end_t) & (df_spo2['time'] <= end_t + 30)]
                if len(win) == 0: continue
                min_spo2 = win['spo2'].min()
                pre_win = df_spo2[(df_spo2['time'] >= end_t - 30) & (df_spo2['time'] < end_t)]
                if len(pre_win) == 0: continue
                baseline = pre_win['spo2'].max()
                drop = baseline - min_spo2
                if drop >= desat_threshold:
                    valid_events.append(ev)
            df_events = pd.DataFrame(valid_events)
            st.write(f"**Auto-detected {len(valid_events)} events.**")
        else:
            drops = df_spo2['spo2'].diff() < -desat_threshold
            starts = df_spo2[drops].index
            ends = (starts + 60).clip(upper=len(df_spo2)-1)
            events = [{"start": df_spo2.loc[s, 'time'], "end": df_spo2.loc[e, 'time']} for s, e in zip(starts, ends)]
            df_events = pd.DataFrame(events)
            st.write(f"Detected {len(events)} events from SpO₂.")
        progress_bar.progress(0.60)

        # --------------------------------------------------------------
        # STEP 5: Crop to EEG
        # --------------------------------------------------------------
        status_text.text("Step 5/6: Cropping to EEG duration...")
        eeg_duration = raw.times[-1]
        st.info(f"Duration: {eeg_duration/3600:.1f} hours")
        df_spo2 = df_spo2[df_spo2['time'] <= eeg_duration].copy()
        odi_events = odi_events[odi_events['time'] <= eeg_duration].copy()
        if not df_events.empty and 'end' in df_events.columns:
            df_events = df_events[df_events['end'] <= eeg_duration].copy()
        else:
            df_events = pd.DataFrame(columns=['start', 'end'])

        if df_events.empty:
            st.warning("No **valid apnea/hypopnea events** found.")
            total_hours = eeg_duration / 3600
            ahi_total = 0.0
            odi_total = len(odi_events) / total_hours if total_hours > 0 else 0
            total_hb = 0.0
            hb_display = "0.0"
            progress_bar.progress(1.0)
            status_text.text("Analysis complete.")
            st.markdown("---")
            st.subheader("Overall Metrics")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("**AHI**", "0.0")
            with col2: st.metric(f"**ODI ({desat_threshold}%)**", f"{odi_total:.1f}")
            with col3: st.metric("**Hypoxic Burden**", hb_display)
            st.success("**Risk Level:** Low")
            st.stop()

        progress_bar.progress(0.70)

        # --------------------------------------------------------------
        # STEP 6: Staging + HB + CI
        # --------------------------------------------------------------
        status_text.text("Step 6/6: Staging + HB + 95% CI...")
        stage_events = {'W': [], 'N1': [], 'N2': [], 'N3': [], 'REM': [], 'Unknown': []}
        if YASA_AVAILABLE and eeg_ch and raw.info['sfreq'] >= 100:
            st.info("**YASA Deep Learning Staging**")
            try:
                sls = yasa.SleepStaging(raw, eeg_name=eeg_ch, eog_name=eog_ch, emg_name=emg_ch)
                hypno = sls.predict()
                stages = hypno[::3]
                stages = ['REM' if s == 'R' else s for s in stages]
                max_epochs = int(eeg_duration / 30)
                stages = stages[:max_epochs]
                for i, stage in enumerate(stages):
                    start = i * 30
                    end = (i + 1) * 30
                    evs = df_events[(df_events['end'] >= start) & (df_events['end'] < end)]
                    stage_events[stage].extend(evs.to_dict('records'))
            except Exception as e:
                st.warning(f"YASA error: {e}. Using rule-based.")
                stages = rule_based_staging(raw, eeg_ch, eog_ch, emg_ch)[:int(eeg_duration / 30)]
        else:
            st.info("Using **rule-based** staging")
            stages = rule_based_staging(raw, eeg_ch, eog_ch, emg_ch)[:int(eeg_duration / 30)]

        for i, stage in enumerate(stages):
            stage = stage if stage in ['W', 'N1', 'N2', 'N3', 'REM'] else 'Unknown'
            start = i * 30
            end = (i + 1) * 30
            evs = df_events[(df_events['end'] >= start) & (df_events['end'] < end)]
            stage_events[stage].extend(evs.to_dict('records'))

        cnt = pd.Series(stages).value_counts()
        stage_time = (cnt * 30 / 3600).to_dict()
        total_hours = eeg_duration / 3600

        total_hb = 0.0
        stage_results = {}
        stage_names = {'W': 'Wake', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3', 'REM': 'REM'}
        for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
            hrs = stage_time.get(stage, 0)
            if hrs == 0: continue
            evs = stage_events.get(stage, [])
            ahi = len(evs) / hrs
            stage_indices = [i for i, s in enumerate(stages) if s == stage]
            if not stage_indices: continue
            stage_start_sec = stage_indices[0] * 30
            stage_end_sec = stage_start_sec + hrs * 3600
            odi_in_stage = odi_events[(odi_events['time'] >= stage_start_sec) & (odi_events['time'] < stage_end_sec)]
            odi = len(odi_in_stage) / hrs
            area_total = 0.0
            for ev in evs:
                end_t = ev['end']
                base_df = df_spo2[(df_spo2['time'] >= end_t - pre_event_sec) & (df_spo2['time'] < end_t)]
                if len(base_df) == 0: continue
                baseline = base_df['spo2'].max()
                win_df = df_spo2[(df_spo2['time'] >= end_t - desat_start_sec) & (df_spo2['time'] <= end_t + desat_end_sec)]
                if len(win_df) < 2: continue
                if artifact_filter != "Off":
                    win_df = win_df[~win_df.index.isin(df_spo2[df_spo2['artifact']].index)]
                depth = np.maximum(baseline - win_df['spo2'].values, 0)
                area = trapz(depth, win_df['time'].values)
                area_total += area
            hb = (area_total / 60) / hrs
            stage_results[stage] = {'AHI': ahi, 'ODI': odi, 'HB': hb, 'hrs': hrs}
            total_hb += hb * hrs
        total_hb /= sum(stage_time.values())
        ahi_total = len(df_events) / total_hours
        odi_total = len(odi_events) / total_hours

        # --------------------------------------------------------------
        # BOOTSTRAP 95% CI FOR HB
        # --------------------------------------------------------------
        def calculate_hb_bootstrap(events_df, df_spo2, pre_event_sec, desat_start_sec, desat_end_sec, artifact_filter, n_boot=1000):
            hb_values = []
            for _ in range(n_boot):
                boot_events = events_df.sample(n=len(events_df), replace=True)
                area_total = 0.0
                for _, ev in boot_events.iterrows():
                    end_t = ev['end']
                    base_df = df_spo2[(df_spo2['time'] >= end_t - pre_event_sec) & (df_spo2['time'] < end_t)]
                    if len(base_df) == 0: continue
                    baseline = base_df['spo2'].max()
                    win_df = df_spo2[(df_spo2['time'] >= end_t - desat_start_sec) & (df_spo2['time'] <= end_t + desat_end_sec)]
                    if len(win_df) < 2: continue
                    if artifact_filter != "Off":
                        win_df = win_df[~win_df.index.isin(df_spo2[df_spo2['artifact']].index)]
                    depth = np.maximum(baseline - win_df['spo2'].values, 0)
                    area = trapz(depth, win_df['time'].values)
                    area_total += area
                hb_boot = (area_total / 60) / (eeg_duration / 3600)
                hb_values.append(hb_boot)
            return np.percentile(hb_values, [2.5, 97.5])

        if len(df_events) > 0:
            ci_low, ci_high = calculate_hb_bootstrap(df_events, df_spo2, pre_event_sec, desat_start_sec, desat_end_sec, artifact_filter)
            hb_display = f"{total_hb:.1f} (95% CI: {ci_low:.1f}–{ci_high:.1f})"
        else:
            hb_display = f"{total_hb:.1f}"

        risk = "Low"
        if total_hb >= 88: risk = "Very High"
        elif total_hb >= 53: risk = "High"
        elif total_hb >= 20: risk = "Moderate"

        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")

        # --------------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------------
        st.markdown("---")
        st.subheader("Overall Sleep Apnea Metrics")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("**AHI**", f"{ahi_total:.1f}")
        with col2: st.metric(f"**ODI ({desat_threshold}%)**", f"{odi_total:.1f}")
        with col3: st.metric("**Hypoxic Burden**", hb_display)
        st.success(f"**Risk Level:** {risk}")

        # ... [stage-specific, reports unchanged] ...

# --------------------------------------------------------------
# Helper: Rule-based staging
# --------------------------------------------------------------
def rule_based_staging(raw, eeg_ch, eog_ch, emg_ch):
    if not eeg_ch: return ['Total'] * int(raw.n_times / (30 * raw.info['sfreq']))
    events_mne = mne.make_fixed_length_events(raw, id=1, duration=30.0)
    epochs = mne.Epochs(raw, events_mne, tmin=0, tmax=30.0, preload=True, picks=[eeg_ch, eog_ch, emg_ch], baseline=None)
    stages = []
    for i in range(len(epochs)):
        epoch = epochs[i]
        eeg = epoch.get_data(picks=eeg_ch)[0, 0]
        emg = epoch.get_data(picks=emg_ch)[0, 0] if emg_ch else None
        psds, freqs = mne.time_frequency.psd_welch(epoch, fmin=0.5, fmax=30, picks=[eeg_ch], n_fft=1024, n_jobs=1)
        psd = psds[0]
        delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs < 12)])
        spindle = np.mean(psd[(freqs >= 12) & (freqs < 15)])
        if alpha > theta * 1.5 and (emg is None or np.var(emg) > 0): stage = 'W'
        elif spindle > theta: stage = 'N2'
        elif delta > theta: stage = 'N3'
        elif theta > alpha: stage = 'N1'
        else: stage = 'REM'
        stages.append(stage)
    return stages

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
st.markdown("**Open-source** • [GitHub](https://github.com/Apolloplectic/hypoxic-burden-edf)")
st.markdown("**DOI**: [10.5281/zenodo.17561726](https://doi.org/10.5281/zenodo.17561726)")
st.markdown("Built with **Streamlit + MNE + YASA**.")
st.markdown("Cite: *Eur Heart J* 2019;40:1149-1157.")

