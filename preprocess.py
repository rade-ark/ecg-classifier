"""
FILTER DESIGN:
- Bandpass: 1-40 Hz
- Type: Butterworth (maximally flat, no ripple)
- Order: 4 (effective order 8 after filtfilt)

- 1 Hz high-pass: Removes baseline wander from breathing/movement
- 40 Hz low-pass: Keeps QRS energy, removes EMG noise
- Butterworth: No ripple = no artificial peaks that could confuse diagnosis
- filtfilt: Zero-phase = preserves exact timing of heartbeat events
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz
import os

# CONFIGURATION

DATA_PATH = r"/home/codespace/.cache/kagglehub/datasets/shayanfazeli/heartbeat/versions/1"  # TODO: Update this!

# Filter specifications (YOUR DESIGN!)
LOWCUT = 1.0    # Hz - High-pass cutoff
HIGHCUT = 40.0  # Hz - Low-pass cutoff
ORDER = 4       # Filter order
FS = 125        # Sampling rate (Hz) - from MIT-BIH database

# FILTER DESIGN FUNCTIONS

def design_butterworth_bandpass(lowcut, highcut, fs, order=4):
    """
    Parameters:
    -----------
    lowcut : float
        Lower cutoff frequency (Hz)
    highcut : float
        Upper cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int
        Filter order (effective order will be 2x with filtfilt)
    
    Returns:
    --------
    b, a : ndarray
        Filter coefficients (numerator and denominator)

    - Nyquist frequency = fs/2 = 62.5 Hz (max frequency we can represent)
    - We normalize cutoffs by Nyquist (required by scipy)
    - butter() returns transfer function coefficients H(z) = B(z)/A(z)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design the filter
    b, a = butter(order, [low, high], btype='band')
    
    return b, a

def apply_filter(signal, b, a):
    """
    - Regular lfilter() introduces phase delay (shifts signal in time)
    - filtfilt() applies filter forward AND backward
    - Phase delays cancel out → output aligned with input
    - Critical for medical signals where TIMING matters!
    
    TRADE-OFF:
    - Effective order doubles (order 4 → effective 8)
    - But we get perfect time alignment!
    """
    filtered = filtfilt(b, a, signal)
    return filtered

def visualize_filter_response(b, a, fs):
    """
    Plot frequency response of the filter.
    
    - Passband: frequencies we keep (flat region)
    - Stopband: frequencies we remove (steep dropoff)
    - Transition: how sharp the cutoff is
    """
    # Compute frequency response
    w, h = freqz(b, a, worN=8000)
    freq = (fs * 0.5 / np.pi) * w
    
    # Plot magnitude response
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Magnitude in dB
    plt.subplot(1, 2, 1)
    plt.plot(freq, 20 * np.log10(abs(h)), 'b', linewidth=2)
    plt.axvline(LOWCUT, color='r', linestyle='--', label=f'Low cutoff: {LOWCUT} Hz')
    plt.axvline(HIGHCUT, color='r', linestyle='--', label=f'High cutoff: {HIGHCUT} Hz')
    plt.axhline(-3, color='g', linestyle=':', label='-3 dB (half power)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Butterworth Bandpass Filter Response (Order {ORDER})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 60])
    
    # Subplot 2: Zoom into passband
    plt.subplot(1, 2, 2)
    plt.plot(freq, 20 * np.log10(abs(h)), 'b', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Passband Detail (1-40 Hz)')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.5, 45])
    plt.ylim([-5, 1])
    
    plt.tight_layout()
    plt.savefig('3_filter_frequency_response.png', dpi=150)
    print("✓ Saved: 3_filter_frequency_response.png")
    plt.show()


def compare_signals(original, filtered, title, sample_idx=0):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f'{title} - Sample {sample_idx}', fontsize=14, fontweight='bold')
    
    time = np.arange(len(original)) / FS
    
    # Time domain comparison
    axes[0, 0].plot(time, original, 'b-', linewidth=1.5, label='Original', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Original Signal (Raw)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(time, filtered, 'r-', linewidth=1.5, label='Filtered', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Filtered Signal (Cleaned)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Frequency domain comparison
    from scipy.fft import fft, fftfreq
    
    # Original FFT
    yf_orig = fft(original)
    xf = fftfreq(len(original), 1/FS)
    pos_mask = xf >= 0
    
    axes[1, 0].plot(xf[pos_mask], np.abs(yf_orig[pos_mask]), 'b-', linewidth=1.5)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Original Frequency Spectrum')
    axes[1, 0].set_xlim([0, 60])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Filtered FFT
    yf_filt = fft(filtered)
    axes[1, 1].plot(xf[pos_mask], np.abs(yf_filt[pos_mask]), 'r-', linewidth=1.5)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Filtered Frequency Spectrum')
    axes[1, 1].set_xlim([0, 60])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(LOWCUT, color='orange', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(HIGHCUT, color='orange', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

# MAIN PREPROCESSING PIPELINE

def main():
    print("=" * 60)
    print("ECG SIGNAL PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_file = os.path.join(DATA_PATH, "mitbih_train.csv")
    train_df = pd.read_csv(train_file, header=None)
    
    signals = train_df.iloc[:, :-1].values
    labels = train_df.iloc[:, -1].values
    
    print(f"✓ Loaded {len(signals)} samples")
    
    # Design filter
    print(f"\n[2/5] Designing Butterworth bandpass filter...")
    print(f"  Specifications:")
    print(f"    - Passband: {LOWCUT}-{HIGHCUT} Hz")
    print(f"    - Order: {ORDER} (effective: {ORDER*2} with filtfilt)")
    print(f"    - Type: Butterworth (maximally flat)")
    
    b, a = design_butterworth_bandpass(LOWCUT, HIGHCUT, FS, ORDER)
    print(f"✓ Filter designed")
    
    # Visualize filter response
    print(f"\n[3/5] Visualizing filter frequency response...")
    visualize_filter_response(b, a, FS)
    
    # Apply filter to all signals
    print(f"\n[4/5] Applying filter to all {len(signals)} signals...")
    filtered_signals = np.zeros_like(signals)
    
    for i in range(len(signals)):
        filtered_signals[i] = apply_filter(signals[i], b, a)
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(signals)} signals...")
    
    print(f"✓ All signals filtered")
    
    # Visualize results for each class
    print(f"\n[5/5] Creating before/after comparisons...")
    
    class_names = ['Normal', 'Supraventricular', 'Premature Ventricular', 
                   'Fusion', 'Unclassifiable']
    
    # Show one example from each class
    for class_id in range(5):
        class_indices = np.where(labels == class_id)[0]
        sample_idx = class_indices[0]  # Take first sample
        
        fig = compare_signals(
            signals[sample_idx],
            filtered_signals[sample_idx],
            f'Class {class_id}: {class_names[class_id]}',
            sample_idx
        )
        
        filename = f'4_comparison_class_{class_id}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close(fig)
    
    # Save filtered data
    print(f"\n[6/6] Saving filtered dataset...")
    filtered_df = pd.DataFrame(filtered_signals)
    filtered_df['label'] = labels
    
    output_file = 'mitbih_train_filtered.csv'
    filtered_df.to_csv(output_file, index=False, header=False)
    print(f"✓ Saved: {output_file}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"""
Summary:
- Original signals: {len(signals)}
- Filter applied: Butterworth bandpass ({LOWCUT}-{HIGHCUT} Hz, order {ORDER})
- Output saved: {output_file}
"""
    )

if __name__ == "__main__":
    main()