import streamlit as st
import pandas as pd
import numpy as np
import mne
from scipy.integrate import trapezoid as trapz
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
**Upload PSG EDF file** → get the hypoxic burden in (%min)/h.

Based on:  
> Azarbarzin A, et al. *European Heart Journal* (2019) – DOI: [10.1093/eurheartj/ehy624](https://doi.org/10.1093/eurheartj/ehy624)
""")

edf_file = st.file_uploader("Upload PSG EDF file", type=["edf"])

# --------------------------------------------------------------
# Helper: Rule-based staging
# --------------------------------------------------------------
def rule_based_staging(raw, eeg_ch, eog_ch, emg_ch):
    if not eeg_ch:
        return ['Total'] * int(raw.n_times / (30 * raw.info['sfreq']))
    events_mne = mne.make_fixed_length_events(raw, id=1, duration=30.0)
    epochs = mne.Epochs(raw, events_mne, tmin=0, tmax=30.0, preload=True,
                        picks=[eeg_ch, eog_ch, emg_ch], baseline=None)

    stages = []
    for i in range(len(epochs)):
        epoch = epochs[i]
        eeg = epoch.get_data(picks=eeg_ch)[0, 0]
        emg = epoch.get_data(picks=emg_ch)[0, 0] if emg_ch else None

        psds, freqs = mne.time_frequency.psd_welch(
            epoch, fmin=0.5, fmax=30, picks=[eeg_ch], n_fft=1024, n_jobs=1)
        psd = psds[0]
        delta   = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        theta   = np.mean(psd[(freqs >= 4)   & (freqs < 8)])
        alpha   = np.mean(psd[(freqs >= 8)   & (freqs < 12)])
        spindle = np.mean(psd[(freqs >= 12)  & (freqs < 15)])

        if alpha > theta * 1.5 and (emg is None or np.var(emg) > 0):
            stage = 'W'
        elif spindle > theta:
            stage = 'N2'
        elif delta > theta:
            stage = 'N3'
        elif theta > alpha:
            stage = 'N1'
        else:
            stage = 'REM'
        stages.append(stage)
    return stages

# --------------------------------------------------------------
# Plot Function: Azarbarzin Style
# --------------------------------------------------------------
def plot_desats_az_style(events_list, stage_name, df_spo2):
    if not events_list:
        st.warning(f"No events in {stage_name} sleep.")
        return
    pre_sec = 100
    rel_times = np.arange(-pre_sec, 180, 1.0)
    curves = []
    for ev in events_list:
        end_t = ev['end']
        win = df_spo2[(df_spo2['time'] >= end_t - pre_sec) & (df_spo2['time'] <= end_t + 180)]
        if len(win) < 10: continue
        interp = np.interp(rel_times, win['time'] - end_t, win['spo2'], left=np.nan, right=np.nan)
        curves.append(interp)
    if not curves:
        st.warning(f"No valid desaturation curves in {stage_name}.")
        return
    avg = np.nanmean(curves, axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in curves:
        ax.plot(rel_times, c, color='lightgray', alpha=0.15)
    ax.plot(rel_times, avg, color='#1f77b4', linewidth=3, label='Average')
    ax.set_xlabel('Time from Event End (seconds)')
    ax.set_ylabel('SpO₂ (%)')
    ax.set_title(f'{stage_name} Sleep: Average Desaturation (n={len(curves)} events)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# --------------------------------------------------------------
# Main App
# --------------------------------------------------------------
if edf_file is not None:
    temp_path = "temp_upload.edf"
    with open(temp_path, "wb") as f:
        f.write(edf_file.getbuffer())
    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
    st.success("EDF loaded!")

    # Channels
    spo2_ch = next((ch for ch in raw.ch_names if any(n in ch.upper() for n in ['SPO2', 'SAO2'])), None)
    flow_ch = next((ch for ch in raw.ch_names if any(n in ch.upper() for n in ['AIRFLOW', 'FLOW'])), None)
    eeg_ch  = next((ch for ch in raw.ch_names if 'EEG' in ch.upper()), None)
    eog_ch  = next((ch for ch in raw.ch_names if 'EOG' in ch.upper()), None)
    emg_ch  = next((ch for ch in raw.ch_names if 'EMG' in ch.upper()), None)

    if not spo2_ch:
        st.error("SpO₂ channel required.")
        st.stop()

    # --------------------------------------------------------------
    # STEP 1: ODI Threshold Selector
    # --------------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        odi_threshold = st.selectbox(
            "ODI Desaturation Threshold",
            options=[3, 4],
            index=0,
	    format_func=lambda x: f"{x}%",
            help="3% = AASM standard | 4% = Medicare/older studies"
        )
    with col2:
        st.info(f"Using **{odi_threshold}%** desaturation for ODI")

    # --------------------------------------------------------------
    # Resample SpO₂ to 1 Hz
    # --------------------------------------------------------------
    spo2_sig, spo2_times = raw[spo2_ch]
    df_spo2 = pd.DataFrame({"time": spo2_times.flatten(), "spo2": spo2_sig.flatten()})
    if raw.info['sfreq'] != 1:
        st.info(f"Resampling SpO₂ from {raw.info['sfreq']:.1f} Hz → 1 Hz")
        df_spo2['time'] = pd.to_datetime(df_spo2['time'], unit='s')
        df_spo2 = df_spo2.set_index('time').resample('1S').mean().interpolate().reset_index()
        df_spo2['time'] = (df_spo2['time'] - df_spo2['time'].iloc[0]).dt.total_seconds()

    # --------------------------------------------------------------
    # STEP 2: Detect ODI Events (3% or 4%)
    # --------------------------------------------------------------
    df_spo2['spo2_next'] = df_spo2['spo2'].shift(-10)
    df_spo2['desat'] = (df_spo2['spo2'].diff() <= -odi_threshold) & \
                       (df_spo2['spo2_next'] >= df_spo2['spo2'] + (odi_threshold - 1))
    odi_events = df_spo2[df_spo2['desat']].copy()
    odi_events = odi_events[['time']].reset_index(drop=True)

    # --------------------------------------------------------------
    # Event detection from airflow
    # --------------------------------------------------------------
    if flow_ch:
        st.write(f"**Airflow channel:** `{flow_ch}`")
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
        df_events = pd.DataFrame(events)
        st.write(f"**Auto-detected {len(events)} events from airflow.**")
    else:
        st.warning("No airflow → using SpO₂ drops.")
        drops = df_spo2['spo2'].diff() < -3
        starts = df_spo2[drops].index
        ends = (starts + 60).clip(upper=len(df_spo2)-1)
        events = [{"start": df_spo2.loc[s, 'time'], "end": df_spo2.loc[e, 'time']} for s, e in zip(starts, ends)]
        df_events = pd.DataFrame(events)
        st.write(f"Detected {len(events)} events from SpO₂.")

    # --------------------------------------------------------------
    # CRITICAL: Crop to EEG duration
    # --------------------------------------------------------------
    eeg_duration = raw.times[-1]
    st.info(f"EEG duration: {eeg_duration/3600:.2f} hours → cropping SpO₂ and events")

    df_spo2 = df_spo2[df_spo2['time'] <= eeg_duration].copy()
    df_events = df_events[df_events['end'] <= eeg_duration].copy()
    odi_events = odi_events[odi_events['time'] <= eeg_duration].copy()

    if len(df_events) == 0:
        st.error("No events within EEG recording window.")
        st.markdown("""
        **Why?**  
        - EEG/EOG/EMG are shorter than SpO₂/airflow  
        - YASA can only stage where EEG exists  
        - Events after EEG end are **ignored**
        """)
        st.stop()

    st.success(f"Using {len(df_events)} events within EEG window")

    # --------------------------------------------------------------
    # Sleep Staging + Event Assignment
    # --------------------------------------------------------------
    stage_events = {'W': [], 'N1': [], 'N2': [], 'N3': [], 'REM': [], 'Unknown': []}

    if YASA_AVAILABLE and eeg_ch and raw.info['sfreq'] >= 100:
        st.info("**Deep Learning Staging Active** (YASA – 87% accuracy on 3000+ nights)")
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
    st.write(f"**Staged duration:** {total_hours:.2f} hours")

    # --------------------------------------------------------------
    # STEP 3–7: AHI, ODI, HB — PER STAGE & OVERALL
    # --------------------------------------------------------------
    total_hb = 0.0
    stage_results = {}
    stage_names = {'W': 'Wake', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3', 'REM': 'REM'}

    for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
        hrs = stage_time.get(stage, 0)
        if hrs == 0: continue

        # AHI
        evs = stage_events.get(stage, [])
        ahi = len(evs) / hrs

        # ODI
        stage_start_sec = [i*30 for i, s in enumerate(stages) if s == stage][0]
        stage_end_sec = stage_start_sec + hrs * 3600
        odi_in_stage = odi_events[
            (odi_events['time'] >= stage_start_sec) & 
            (odi_events['time'] < stage_end_sec)
        ]
        odi = len(odi_in_stage) / hrs

        # HB
        area_total = 0.0
        for ev in evs:
            end_t = ev['end']
            base_df = df_spo2[(df_spo2['time'] >= end_t - 100) & (df_spo2['time'] < end_t)]
            if len(base_df) == 0: continue
            baseline = base_df['spo2'].max()
            win_df = df_spo2[(df_spo2['time'] >= end_t - 60) & (df_spo2['time'] <= end_t + 120)]
            if len(win_df) < 2: continue
            depth = np.maximum(baseline - win_df['spo2'].values, 0)
            area = trapz(depth, win_df['time'].values)
            area_total += area
        hb = (area_total / 60) / hrs

        stage_results[stage] = {'AHI': ahi, 'ODI': odi, 'HB': hb, 'hrs': hrs}
        total_hb += hb * hrs

    total_hb /= sum(stage_time.values())
    ahi_total = len(df_events) / total_hours
    odi_total = len(odi_events) / total_hours

    # Risk
    risk = "Low"
    if total_hb >= 88: risk = "Very High (2.7x CVD mortality)"
    elif total_hb >= 53: risk = "High"
    elif total_hb >= 20: risk = "Moderate"

    # --------------------------------------------------------------
    # STEP 5: Display Overall Metrics
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("Overall Sleep Apnea Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**AHI**", f"{ahi_total:.1f}", "events/h")
    with col2:
        st.metric(f"**ODI ({odi_threshold}%)**", f"{odi_total:.1f}", "events/h")
    with col3:
        st.metric("**Hypoxic Burden**", f"{total_hb:.1f}", "(%min)/h")
    st.success(f"**Risk Level:** {risk}")

    # --------------------------------------------------------------
    # STEP 6: Stage-Specific Expanders
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("Stage-Specific Metrics")
    for stage, r in stage_results.items():
        with st.expander(f"**{stage_names[stage]}** – {r['hrs']:.1f}h", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("AHI", f"{r['AHI']:.1f}")
            c2.metric(f"ODI ({odi_threshold}%)", f"{r['ODI']:.1f}")
            c3.metric("HB", f"{r['HB']:.1f}")

    # --------------------------------------------------------------
    # Average Desaturation (All)
    # --------------------------------------------------------------
    if events:
        pre_sec = 100
        rel_times = np.arange(-pre_sec, 180, 1.0)
        curves = []
        for ev in events:
            end_t = ev['end']
            win = df_spo2[(df_spo2['time'] >= end_t - pre_sec) & (df_spo2['time'] <= end_t + 180)]
            if len(win) < 10: continue
            interp = np.interp(rel_times, win['time'] - end_t, win['spo2'], left=np.nan, right=np.nan)
            curves.append(interp)
        if curves:
            avg = np.nanmean(curves, axis=0)
            peaks, _ = find_peaks(-avg, prominence=0.3, distance=30)
            w_start = rel_times[peaks[0]] if len(peaks) >= 2 else -60
            w_end = rel_times[peaks[-1]] if len(peaks) >= 2 else 120

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(rel_times, avg, label='Avg Desat', color='blue', linewidth=2)
            ax.axvline(w_start, color='red', linestyle='--', label=f'Start ({w_start:.0f}s)')
            ax.axvline(w_end, color='green', linestyle='--', label=f'End ({w_end:.0f}s)')
            ax.set_xlabel('Time from event end (s)')
            ax.set_ylabel('SpO₂ (%)')
            ax.set_title('Average Event-Associated Desaturation')
            ax.legend()
            st.pyplot(fig)

    # --------------------------------------------------------------
    # CLICKABLE STAGE PLOTS
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("Show Desaturation Curves by Sleep Stage")
    stage_map = {'Wake': 'W', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3', 'REM': 'REM'}
    stages_to_show = st.multiselect(
        "Select stages to visualize:",
        options=list(stage_map.keys()),
        default=[]
    )
    submit = st.button("Generate Plots")

    if submit and stages_to_show:
        for ui_stage in stages_to_show:
            stage = stage_map[ui_stage]
            events_in_stage = stage_events.get(stage, [])
            with st.expander(f"{ui_stage} Sleep (n={len(events_in_stage)} events)", expanded=True):
                plot_desats_az_style(events_in_stage, ui_stage, df_spo2)

    # --------------------------------------------------------------
    # Updated Report (TXT)
    # --------------------------------------------------------------
    def txt_report():
        lines = [
            f"Hypoxic Burden Report - {datetime.now():%Y-%m-%d %H:%M}",
            f"File: {edf_file.name}",
            f"Staged duration: {total_hours:.2f} hours",
            f"ODI Threshold: {odi_threshold}%",
            "",
            f"OVERALL: AHI={ahi_total:.1f}, ODI={odi_total:.1f}, HB={total_hb:.1f} → {risk}",
            "",
            "STAGE-SPECIFIC:"
        ]
        for stage, r in stage_results.items():
            lines.append(f"  {stage_names[stage]} ({r['hrs']:.1f}h): AHI={r['AHI']:.1f}, ODI={r['ODI']:.1f}, HB={r['HB']:.1f}")
        return "\n".join(lines)

    st.download_button("Download Full Report (TXT)", txt_report(), "hypoxic_burden_report.txt", "text/plain")

    # --------------------------------------------------------------
    # PDF Report Export (MUST BE OUTSIDE FUNCTION!)
    # --------------------------------------------------------------
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    def pdf_report():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Hypoxic Burden Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"File: {edf_file.name}", styles['Normal']))
        story.append(Paragraph(f"Date: {datetime.now():%Y-%m-%d %H:%M}", styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("OVERALL METRICS", styles['Heading2']))
        data = [
            ["Metric", "Value"],
            ["AHI", f"{ahi_total:.1f} events/h"],
            [f"ODI ({odi_threshold}%)", f"{odi_total:.1f} events/h"],
            ["Hypoxic Burden", f"{total_hb:.1f} (%min)/h"],
            ["Risk Level", risk]
        ]
        table = Table(data, colWidths=[200, 200])
        table.setStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)])
        story.append(table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("STAGE-SPECIFIC METRICS", styles['Heading2']))
        data = [["Stage", "Hours", "AHI", f"ODI({odi_threshold}%)", "HB"]]
        for stage, r in stage_results.items():
            data.append([
                stage_names[stage],
                f"{r['hrs']:.1f}",
                f"{r['AHI']:.1f}",
                f"{r['ODI']:.1f}",
                f"{r['HB']:.1f}"
            ])
        table = Table(data, colWidths=[100, 70, 70, 80, 70])
        table.setStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)])
        story.append(table)

        doc.build(story)
        buffer.seek(0)
        return buffer

    pdf_buffer = pdf_report()
    st.download_button(
        "Download Full Report (PDF)",
        pdf_buffer,
        "hypoxic_burden_report.pdf",
        "application/pdf"
    )
# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
st.markdown("**Open-source** • [GitHub](https://github.com/Apolloplectic/hypoxic-burden-edf)")
st.markdown("Built with **Streamlit + MNE-Python + YASA**.")
st.markdown("Cite: *Eur Heart J* 2019;40:1149-1157.")