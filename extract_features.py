import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks
import os

FS = 125  # Sampling rate (Hz)

# TIME DOMAIN FEATURES

def extract_time_features(signal):
    """    
    WHY THESE FEATURES?
    - Mean/Std: Captures baseline and variability
    - Min/Max: Captures peak amplitudes (QRS height)
    - Range: Total amplitude variation
    - RMS: Energy in signal
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['range'] = features['max'] - features['min']
    
    # Root mean square (energy measure)
    features['rms'] = np.sqrt(np.mean(signal**2))
    
    # Peak-related features
    # Find peaks (QRS complexes, T waves, etc.)
    peaks, properties = find_peaks(signal, height=0, distance=10)
    
    if len(peaks) > 0:
        features['num_peaks'] = len(peaks)
        features['mean_peak_height'] = np.mean(properties['peak_heights'])
        features['max_peak_height'] = np.max(properties['peak_heights'])
    else:
        features['num_peaks'] = 0
        features['mean_peak_height'] = 0
        features['max_peak_height'] = 0
    
    return features

# FREQUENCY DOMAIN FEATURES

def extract_frequency_features(signal, fs=FS):
    """
    Extract frequency-domain features using FFT.
    
    WHY FREQUENCY FEATURES?
    Different heartbeat types have different frequency distributions.
    - Normal beats: Energy concentrated 5-20 Hz
    - Abnormal beats: May have different spectral characteristics
    """
        
    features = {}
    
    # Compute FFT
    N = len(signal)
    yf = fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Take positive frequencies only
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    magnitude = np.abs(yf[pos_mask])
    
    # Total spectral energy
    features['spectral_energy'] = np.sum(magnitude**2)
    
    # Spectral centroid (center of mass of spectrum)
    # Tells you "where" the energy is concentrated
    if features['spectral_energy'] > 0:
        features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
    else:
        features['spectral_centroid'] = 0
    
    # Energy in specific frequency bands
    # Based on your 1-40 Hz passband, let's divide it up:
    low_band = (freqs >= 1) & (freqs < 10)    # Low frequencies
    mid_band = (freqs >= 10) & (freqs < 25)   # Mid frequencies (main QRS)
    high_band = (freqs >= 25) & (freqs <= 40) # High frequencies
    
    features['energy_low'] = np.sum(magnitude[low_band]**2)
    features['energy_mid'] = np.sum(magnitude[mid_band]**2)
    features['energy_high'] = np.sum(magnitude[high_band]**2)
    
    # Ratio features (relative energy distribution)
    total_energy = features['energy_low'] + features['energy_mid'] + features['energy_high']
    if total_energy > 0:
        features['ratio_low'] = features['energy_low'] / total_energy
        features['ratio_mid'] = features['energy_mid'] / total_energy
        features['ratio_high'] = features['energy_high'] / total_energy
    else:
        features['ratio_low'] = 0
        features['ratio_mid'] = 0
        features['ratio_high'] = 0
    
    return features

# STATISTICAL FEATURES

def extract_statistical_features(signal):
    """
    Extract statistical features.
    
    WHY THESE?
    - Skewness: Asymmetry of signal distribution
    - Kurtosis: "Peakedness" - how sharp are the peaks?
    - These capture SHAPE characteristics beyond simple amplitude
    """
    features = {}
    
    # Higher-order statistics
    features['skewness'] = stats.skew(signal)
    features['kurtosis'] = stats.kurtosis(signal)
    
    # Zero crossing rate (how often signal crosses zero)
    # Different heartbeat types may have different crossing patterns
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
    
    return features

# COMBINED FEATURE EXTRACTION

def extract_all_features(signal):
    features = {}
    
    # Extract each feature type
    features.update(extract_time_features(signal))
    features.update(extract_frequency_features(signal))
    features.update(extract_statistical_features(signal))
    
    return features

# BATCH PROCESSING

def process_dataset(filepath):
    """
    Extract features from entire dataset.
    
    INPUT: CSV with filtered signals
    OUTPUT: Feature matrix (samples × features) ready for ML
    """
    print("=" * 60)
    print("FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Load filtered data
    print("\n[1/3] Loading filtered signals...")
    df = pd.read_csv(filepath, header=None)
    
    signals = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    print(f"✓ Loaded {len(signals)} signals")
    print(f"✓ Signal length: {signals.shape[1]} samples")
    
    # Extract features for all signals
    print(f"\n[2/3] Extracting features from {len(signals)} signals...")
    print("This may take 1-2 minutes...")
    
    feature_list = []
    
    for i, signal in enumerate(signals):
        features = extract_all_features(signal)
        feature_list.append(features)
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(signals)} signals...")
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_list)
    feature_df['label'] = labels
    
    print(f"✓ Extracted {len(feature_df.columns) - 1} features per signal")
    
    # Save features
    print(f"\n[3/3] Saving feature matrix...")
    output_file = 'ecg_features.csv'
    feature_df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    # Display feature summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    
    feature_names = [col for col in feature_df.columns if col != 'label']
    
    print(f"\nTotal features extracted: {len(feature_names)}")
    print(f"\nFeature categories:")
    
    time_features = [f for f in feature_names if any(x in f for x in 
                    ['mean', 'std', 'min', 'max', 'range', 'rms', 'peak'])]
    freq_features = [f for f in feature_names if any(x in f for x in 
                    ['spectral', 'energy', 'ratio'])]
    stat_features = [f for f in feature_names if any(x in f for x in 
                    ['skewness', 'kurtosis', 'zero'])]
    
    print(f"  Time domain: {len(time_features)} features")
    for f in time_features:
        print(f"    - {f}")
    
    print(f"\n  Frequency domain: {len(freq_features)} features")
    for f in freq_features:
        print(f"    - {f}")
    
    print(f"\n  Statistical: {len(stat_features)} features")
    for f in stat_features:
        print(f"    - {f}")
    
    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Output: {output_file}")
    print("Ready for ML training!")
    print("=" * 60)
    
    return feature_df

# MAIN

if __name__ == "__main__":
    # Process the filtered dataset
    filtered_file = 'mitbih_train_filtered.csv'
    
    if not os.path.exists(filtered_file):
        print(f"❌ ERROR: {filtered_file} not found!")
        print("Please run preprocess_signals.py first.")
        exit(1)
    
    feature_df = process_dataset(filtered_file)
    
    print("\n" + "!" * 60)
    print("!" * 60)