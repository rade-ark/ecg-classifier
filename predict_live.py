import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy import stats as scipy_stats

# LOAD MODEL + METADATA

print("Loading model...")
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = (
    pd.read_csv('feature_names.csv', header=None)
    .iloc[:, 0]
    .astype(str)
    .tolist()
)

# Load test data
test_df = pd.read_csv('mitbih_train_filtered.csv', header=None)
test_signals = test_df.iloc[:, :-1].values
test_labels = test_df.iloc[:, -1].values

FS = 125

# FEATURE EXTRACTION

def extract_all_features(signal):
    features = {}

    # ---- Time domain ----
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['range'] = features['max'] - features['min']
    features['rms'] = np.sqrt(np.mean(signal ** 2))

    peaks, props = find_peaks(signal, height=0, distance=10)
    features['num_peaks'] = len(peaks)
    features['mean_peak_height'] = np.mean(props['peak_heights']) if len(peaks) else 0
    features['max_peak_height'] = np.max(props['peak_heights']) if len(peaks) else 0

    # ---- Frequency domain ----
    N = len(signal)
    yf = fft(signal)
    freqs = np.fft.fftfreq(N, 1 / FS)
    pos = freqs > 0

    freqs = freqs[pos]
    mag = np.abs(yf[pos])

    features['spectral_energy'] = np.sum(mag ** 2)
    features['spectral_centroid'] = (
        np.sum(freqs * mag) / np.sum(mag) if np.sum(mag) else 0
    )

    bands = {
        'low': (1, 10),
        'mid': (10, 25),
        'high': (25, 40)
    }

    energies = {}
    for name, (f1, f2) in bands.items():
        mask = (freqs >= f1) & (freqs < f2)
        energies[name] = np.sum(mag[mask] ** 2)
        features[f'energy_{name}'] = energies[name]

    total_energy = sum(energies.values())
    for name in bands:
        features[f'ratio_{name}'] = energies[name] / total_energy if total_energy else 0

    # ---- Statistical ----
    features['skewness'] = scipy_stats.skew(signal)
    features['kurtosis'] = scipy_stats.kurtosis(signal)
    features['zero_crossing_rate'] = np.mean(np.diff(np.sign(signal)) != 0)

    # ---- Wavelet (conditional) ----
    if any('wavelet' in f for f in feature_names):
        import pywt
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        for i, c in enumerate(coeffs):
            features[f'wavelet_energy_level_{i}'] = np.sum(c ** 2)
            features[f'wavelet_std_level_{i}'] = np.std(c)
            features[f'wavelet_mean_level_{i}'] = np.mean(np.abs(c))

    return features

# PREDICTION (FIXED)

def predict_single(signal):
    features = extract_all_features(signal)
    feature_df = pd.DataFrame([features])

    for col in feature_names:
        if col not in feature_df.columns:
            feature_df[col] = 0

    feature_df = feature_df[feature_names]
    feature_df = feature_df.fillna(0)

    X_scaled = scaler.transform(feature_df)

    pred = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0]
    confidence = probs[pred]

    return pred, confidence, probs

# REAL-TIME VISUALIZATION

class_names = [
    'Normal',
    'Supraventricular',
    'Premature Ventricular',
    'Fusion',
    'Unclassifiable'
]

class_colors = ['green', 'orange', 'red', 'purple', 'gray']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Real-Time ECG Classification', fontsize=16, fontweight='bold')

time_axis = np.arange(186) / FS
current_idx = 0

line, = ax1.plot([], [], lw=1.5)
ax1.set_xlim(0, time_axis[-1])
ax1.set_ylim(-2, 2)
ax1.grid(True)

prediction_text = ax1.text(
    0.02, 0.95, '',
    transform=ax1.transAxes,
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    va='top'
)

bars = ax2.bar(range(5), [0] * 5, color=class_colors)
ax2.set_ylim(0, 1)
ax2.set_xticks(range(5))
ax2.set_xticklabels(class_names, rotation=45)


def update(_):
    global current_idx

    signal = test_signals[current_idx]
    true_label = int(test_labels[current_idx])

    pred, conf, probs = predict_single(signal)

    line.set_data(time_axis, signal)

    correct = pred == true_label
    prediction_text.set_text(
        f"{'✓ CORRECT' if correct else '✗ INCORRECT'}\n"
        f"Predicted: {class_names[pred]} ({conf * 100:.1f}%)\n"
        f"True: {class_names[true_label]}"
    )
    prediction_text.set_color('green' if correct else 'red')

    for bar, p in zip(bars, probs):
        bar.set_height(p)

    current_idx = (current_idx + 1) % len(test_signals)
    return []

print("\nStarting real-time demo...")
ani = FuncAnimation(fig, update, interval=1500, blit=False)
plt.tight_layout()
plt.show()