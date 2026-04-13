"""
ECG Heartbeat Classifier — Streamlit Dashboard
Drop-in app.py for github.com/rade-ark/ecg-classifier

Reuses the exact same preprocessing and feature extraction
logic from preprocess.py and extract_features.py.

Run:
    pip install streamlit joblib scipy numpy pandas matplotlib pywt
    streamlit run app.py

Place best_model.pkl, scaler.pkl, feature_names.csv in the same folder.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats
from scipy.fft import fft
import joblib
import os

warnings.filterwarnings("ignore")

# ── Constants (must match training) ─────────────────────────────────────────
FS = 125          # MIT-BIH sampling rate
LOWCUT = 1.0      # Bandpass low cutoff (Hz)
HIGHCUT = 40.0    # Bandpass high cutoff (Hz)
ORDER = 4         # Butterworth order
SEGMENT_LEN = 187 # Samples per beat window (MIT-BIH standard)

CLASS_NAMES = ["Normal", "Supraventricular", "Premature Ventricular",
               "Fusion", "Unclassifiable"]
CLASS_COLORS = ["#1D9E75", "#185FA5", "#E24B4A", "#BA7517", "#888780"]
CLASS_SHORT  = ["N", "S", "V", "F", "U"]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { font-size: 0.72rem; color: #6c757d; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-bottom: 4px; }
    .metric-value { font-size: 1.6rem; font-weight: 600; line-height: 1; }
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .stProgress > div > div { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Preprocessing (from preprocess.py) ───────────────────────────────────────
def design_filter(lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a

def apply_filter(signal, b, a):
    return filtfilt(b, a, signal)


# ── Feature extraction (from extract_features.py) ────────────────────────────
def extract_time_features(signal):
    f = {}
    f["mean"] = np.mean(signal)
    f["std"]  = np.std(signal)
    f["min"]  = np.min(signal)
    f["max"]  = np.max(signal)
    f["range"] = f["max"] - f["min"]
    f["rms"]  = np.sqrt(np.mean(signal ** 2))
    peaks, props = find_peaks(signal, height=0, distance=10)
    if len(peaks) > 0:
        f["num_peaks"]       = len(peaks)
        f["mean_peak_height"] = np.mean(props["peak_heights"])
        f["max_peak_height"]  = np.max(props["peak_heights"])
    else:
        f["num_peaks"] = f["mean_peak_height"] = f["max_peak_height"] = 0
    return f

def extract_frequency_features(signal, fs=FS):
    f = {}
    N = len(signal)
    yf    = fft(signal)
    freqs = np.fft.fftfreq(N, 1 / fs)
    mask  = freqs > 0
    freqs, mag = freqs[mask], np.abs(yf[mask])
    f["spectral_energy"]   = np.sum(mag ** 2)
    f["spectral_centroid"] = (np.sum(freqs * mag) / np.sum(mag)
                               if f["spectral_energy"] > 0 else 0)
    lo = (freqs >= 1)  & (freqs < 10)
    mi = (freqs >= 10) & (freqs < 25)
    hi = (freqs >= 25) & (freqs <= 40)
    f["energy_low"]  = np.sum(mag[lo] ** 2)
    f["energy_mid"]  = np.sum(mag[mi] ** 2)
    f["energy_high"] = np.sum(mag[hi] ** 2)
    tot = f["energy_low"] + f["energy_mid"] + f["energy_high"]
    f["ratio_low"]  = f["energy_low"]  / tot if tot > 0 else 0
    f["ratio_mid"]  = f["energy_mid"]  / tot if tot > 0 else 0
    f["ratio_high"] = f["energy_high"] / tot if tot > 0 else 0
    return f

def extract_statistical_features(signal):
    f = {}
    f["skewness"]           = float(stats.skew(signal))
    f["kurtosis"]           = float(stats.kurtosis(signal))
    zc = np.where(np.diff(np.sign(signal)))[0]
    f["zero_crossing_rate"] = len(zc) / len(signal)
    return f

def extract_all_features(signal):
    feat = {}
    feat.update(extract_time_features(signal))
    feat.update(extract_frequency_features(signal))
    feat.update(extract_statistical_features(signal))
    return feat


# ── Wavelet features (from improve_model.py) ─────────────────────────────────
def extract_wavelet_features(signal, wavelet="db4", level=4):
    try:
        import pywt
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        f = {}
        for i, c in enumerate(coeffs):
            f[f"wavelet_energy_level_{i}"] = float(np.sum(c ** 2))
            f[f"wavelet_std_level_{i}"]    = float(np.std(c))
            f[f"wavelet_mean_level_{i}"]   = float(np.mean(np.abs(c)))
        return f
    except ImportError:
        return {}


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    missing = [f for f in ("best_model.pkl", "scaler.pkl", "feature_names.csv")
               if not os.path.exists(f)]
    if missing:
        return None, None, None
    model        = joblib.load("best_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    feature_names = pd.read_csv("feature_names.csv", header=None)[0].tolist()
    return model, scaler, feature_names


# ── Classify one beat segment ─────────────────────────────────────────────────
def classify_beat(segment, model, scaler, feature_names):
    feat = extract_all_features(segment)
    feat.update(extract_wavelet_features(segment))
    vec = np.array([feat.get(n, 0.0) for n in feature_names]).reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    pred  = int(model.predict(vec_scaled)[0])
    probs = model.predict_proba(vec_scaled)[0]
    return pred, probs


# ── Detect R-peaks and slice beats ───────────────────────────────────────────
def segment_beats(signal, fs=FS, seg_len=SEGMENT_LEN):
    """Simple R-peak detector → fixed-length windows."""
    threshold = np.mean(signal) + 0.6 * np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold, distance=int(fs * 0.4))
    half = seg_len // 2
    beats, positions = [], []
    for p in peaks:
        start, end = p - half, p + half + 1
        if start >= 0 and end <= len(signal):
            beat = signal[start:end]
            if len(beat) == seg_len:
                beats.append(beat)
                positions.append(p)
    return np.array(beats), np.array(positions)


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_signal(signal, fs=FS, peaks=None, title="ECG signal", color="#1D9E75"):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(t, signal, color=color, linewidth=0.9, alpha=0.9)
    if peaks is not None and len(peaks):
        ax.scatter(peaks / fs, signal[peaks], color="#E24B4A",
                   s=30, zorder=5, label="R-peaks")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    fig.tight_layout()
    return fig

def plot_beat(segment, pred_class, probs, fs=FS):
    t = np.linspace(0, len(segment) / fs, len(segment))
    color = CLASS_COLORS[pred_class]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8),
                             gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(t * 1000, segment, color=color, linewidth=1.4)
    axes[0].set_xlabel("Time (ms)", fontsize=9)
    axes[0].set_ylabel("Amplitude", fontsize=9)
    axes[0].set_title(f"Beat — {CLASS_NAMES[pred_class]}", fontsize=10)
    axes[0].grid(True, alpha=0.2, linewidth=0.4)
    bars = axes[1].barh(CLASS_NAMES, probs * 100,
                         color=[CLASS_COLORS[i] if i == pred_class
                                else "#e9ecef" for i in range(5)],
                         height=0.55)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Confidence (%)", fontsize=9)
    axes[1].set_title("Class probabilities", fontsize=10)
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{p*100:.1f}%", va="center", fontsize=8)
    axes[1].grid(True, axis="x", alpha=0.2, linewidth=0.4)
    fig.tight_layout()
    return fig

def plot_summary(results_df):
    counts = results_df["class"].value_counts().reindex(range(5), fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(CLASS_NAMES, counts.values,
                  color=CLASS_COLORS, width=0.55, alpha=0.85)
    for bar, v in zip(bars, counts.values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Beat count", fontsize=9)
    ax.set_title("Classification summary", fontsize=10)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)
    plt.xticks(rotation=15, fontsize=8)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 ECG Classifier")
    st.caption("github.com/rade-ark/ecg-classifier")
    st.divider()

    st.markdown("### Input source")
    input_mode = st.radio("", ["Upload CSV file", "Use built-in demo signal"],
                          label_visibility="collapsed")

    st.divider()
    st.markdown("### Replay settings")
    replay_speed = st.slider("Replay speed", 1, 10, 3,
                              help="Higher = faster beat-by-beat replay")
    show_filter  = st.checkbox("Show filter comparison", value=False)

    st.divider()
    st.markdown("### Model")
    model, scaler, feature_names = load_model()
    if model is None:
        st.error("No model found.\n\nPlace **best_model.pkl**, **scaler.pkl**, "
                 "**feature_names.csv** in the same folder and restart.")
    else:
        st.success(f"Model loaded · {len(feature_names)} features")

    st.divider()
    st.caption("Preprocessing: Butterworth bandpass 1–40 Hz, order 4, filtfilt")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# ECG Heartbeat Classification")

if model is None:
    st.warning("Train your model first (`python train_model.py` or "
               "`python improve_model.py`), then place the output files here.")
    st.stop()

# ── Load signal ───────────────────────────────────────────────────────────────
raw_signal = None

if input_mode == "Upload CSV file":
    uploaded = st.file_uploader(
        "Upload a CSV — either the raw MIT-BIH dataset or a single-column "
        "signal file (no header needed)",
        type=["csv"],
    )
    if uploaded:
        df_up = pd.read_csv(uploaded, header=None)
        # If it looks like the mitbih format (187 cols + label) take first row as demo
        if df_up.shape[1] >= SEGMENT_LEN:
            # Multi-row dataset: use values as pre-segmented beats
            st.info(f"Detected dataset format — {len(df_up)} rows × "
                    f"{df_up.shape[1]} columns. "
                    "Using signal values directly (no re-segmentation).")
            raw_signal = df_up.iloc[:, :SEGMENT_LEN].values.flatten()
        else:
            # Single-column continuous signal
            raw_signal = df_up.iloc[:, 0].values.astype(float)
else:
    # Demo: generate a synthetic but realistic ECG-like signal
    np.random.seed(42)
    t = np.linspace(0, 10, 10 * FS)

    def ecg_template(t_beat):
        x = t_beat % 1
        if x < 0.05:   return np.sin(x / 0.05 * np.pi) * 0.15
        if x < 0.12:   return -0.05
        if x < 0.14:   return -0.25
        if x < 0.16:   return 1.0
        if x < 0.18:   return -0.35
        if x < 0.30:   return np.sin((x - 0.18) / 0.12 * np.pi) * 0.25
        if x < 0.40:   return np.sin((x - 0.30) / 0.10 * np.pi) * 0.10
        return 0.0

    raw_signal = np.array([ecg_template(ti) for ti in t])
    raw_signal += np.random.normal(0, 0.04, len(raw_signal))
    st.info("Using built-in demo signal (10 s synthetic ECG). "
            "Upload your own CSV for real results.")

# ── Process ───────────────────────────────────────────────────────────────────
if raw_signal is not None and len(raw_signal) >= SEGMENT_LEN:

    b, a = design_filter()

    # Pad short signals so filtfilt doesn't crash
    pad = max(3 * max(len(a), len(b)), 15)
    if len(raw_signal) <= pad * 2:
        raw_signal = np.pad(raw_signal, pad, mode="edge")

    filtered = apply_filter(raw_signal, b, a)

    # ── Waveform display ──────────────────────────────────────────────────────
    st.markdown("### Signal")

    col_sig1, col_sig2 = st.columns([3, 1])
    with col_sig2:
        view_len_s = st.slider("View window (s)", 2, min(30, len(raw_signal) // FS), 5)

    display_raw  = raw_signal[:view_len_s * FS]
    display_filt = filtered[:view_len_s * FS]
    _, r_peaks   = segment_beats(display_filt)

    if show_filter:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_signal(display_raw,  title="Raw signal",      color="#888780"))
        with c2:
            st.pyplot(plot_signal(display_filt, title="Filtered signal", peaks=r_peaks))
    else:
        st.pyplot(plot_signal(display_filt, title="Filtered ECG", peaks=r_peaks))

    plt.close("all")

    # ── Segment beats ─────────────────────────────────────────────────────────
    beats, positions = segment_beats(filtered)

    if len(beats) == 0:
        st.warning("No beats detected. Try a longer or louder signal.")
        st.stop()

    # ── Classify all beats ────────────────────────────────────────────────────
    results = []
    with st.spinner(f"Classifying {len(beats)} beats…"):
        for beat in beats:
            pred, probs = classify_beat(beat, model, scaler, feature_names)
            results.append({"class": pred, "confidence": float(probs[pred]),
                             "probs": probs.tolist()})
    results_df = pd.DataFrame(results)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown("### Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    total   = len(results_df)
    n_anom  = (results_df["class"] != 0).sum()
    anom_pct = round(n_anom / total * 100, 1) if total else 0
    top_cls  = int(results_df["class"].mode()[0])
    avg_conf = round(results_df["confidence"].mean() * 100, 1)

    for col, label, value, color in [
        (m1, "Total beats",    str(total),                 "#1D9E75"),
        (m2, "Normal",         str((results_df["class"]==0).sum()), "#1D9E75"),
        (m3, "Anomalies",      str(n_anom),                "#E24B4A"),
        (m4, "Anomaly rate",   f"{anom_pct}%",             "#BA7517" if anom_pct > 10 else "#1D9E75"),
        (m5, "Avg confidence", f"{avg_conf}%",             "#185FA5"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Chart + table ─────────────────────────────────────────────────────────
    c_chart, c_table = st.columns([1, 1])
    with c_chart:
        st.pyplot(plot_summary(results_df))
        plt.close("all")
    with c_table:
        st.markdown("**Beat log**")
        log_rows = []
        for i, r in results_df.iterrows():
            log_rows.append({
                "Beat #":      i + 1,
                "Class":       CLASS_NAMES[r["class"]],
                "Confidence":  f"{r['confidence']*100:.1f}%",
                "Flag":        "⚠" if r["class"] != 0 else "✓",
            })
        st.dataframe(pd.DataFrame(log_rows), height=260, use_container_width=True)

    # ── Beat-by-beat replay ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### Beat-by-beat replay")
    st.caption("Iterates through every detected beat and shows the live classification.")

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    run_replay  = col_btn1.button("▶  Start replay", type="primary")
    stop_replay = col_btn2.button("⏹  Stop")

    if run_replay:
        beat_placeholder  = st.empty()
        prog_placeholder  = st.empty()
        label_placeholder = st.empty()

        for i, (beat, pos) in enumerate(zip(beats, positions)):
            if stop_replay:
                break

            pred, probs = results[i]["class"], np.array(results[i]["probs"])
            conf = probs[pred]

            # Progress
            prog_placeholder.progress((i + 1) / len(beats),
                                       text=f"Beat {i+1} / {len(beats)}")

            # Badge
            badge_color = CLASS_COLORS[pred]
            label_placeholder.markdown(
                f'<span class="badge" style="background:{badge_color}20;'
                f'color:{badge_color};border:1px solid {badge_color}60">'
                f'  {CLASS_NAMES[pred]}  {conf*100:.1f}% confidence</span>',
                unsafe_allow_html=True,
            )

            # Beat plot
            with beat_placeholder.container():
                st.pyplot(plot_beat(beat, pred, probs))
                plt.close("all")

            time.sleep(1.0 / replay_speed)

        prog_placeholder.empty()
        label_placeholder.markdown("✅ Replay complete.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Export results")
    export_df = results_df.copy()
    export_df["class_name"] = export_df["class"].map(lambda c: CLASS_NAMES[c])
    export_df["confidence_pct"] = (export_df["confidence"] * 100).round(2)
    export_df = export_df.drop(columns=["probs"])

    st.download_button(
        "⬇  Download results CSV",
        data=export_df.to_csv(index=False).encode(),
        file_name="ecg_classification_results.csv",
        mime="text/csv",
    )

else:
    st.info("Upload a CSV file or select the demo signal to begin.")
