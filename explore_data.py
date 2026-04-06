import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = r"C:\Users\ishan\.cache\kagglehub\datasets\shayanfazeli\heartbeat\versions\1"

if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: Path not found: {DATA_PATH}")
    print(f"Please update DATA_PATH with the output from download_data.py")
    exit(1)

train_file = os.path.join(DATA_PATH, "mitbih_train.csv")
print(f"Loading: {train_file}")
train_df = pd.read_csv(train_file, header=None)

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)
print(f"Total samples: {len(train_df)}")
print(f"Columns: {train_df.shape[1]}")
print(f"  - Signal points: {train_df.shape[1] - 1}")
print(f"  - Label column: 1")

# Last column is the label
labels = train_df.iloc[:, -1]
print(f"\nClass distribution:")
for class_id in range(5):
    count = (labels == class_id).sum()
    percentage = 100 * count / len(labels)
    print(f"  Class {class_id}: {count:6d} samples ({percentage:5.2f}%)")

# Extract signals and labels
signals = train_df.iloc[:, :-1].values  # All columns except last
labels = train_df.iloc[:, -1].values    # Last column

print(f"\nSignal shape: {signals.shape}")
print(f"Labels shape: {labels.shape}")

input("\nPress Enter to continue to visualization...")

# VISUALIZATION: Normal vs Abnormal Heartbeats

fig, axes = plt.subplots(5, 3, figsize=(15, 12))
fig.suptitle('ECG Heartbeat Patterns by Class', fontsize=16, fontweight='bold')

class_names = [
    'Normal',
    'Supraventricular',
    'Premature Ventricular',
    'Fusion',
    'Unclassifiable'
]

for class_id in range(5):
    class_indices = np.where(labels == class_id)[0]
    random_samples = np.random.choice(class_indices, 3, replace=False)
    
    for i, sample_idx in enumerate(random_samples):
        ax = axes[class_id, i]
        signal = signals[sample_idx]
        
        sampling_rate = 125  # Hz
        time = np.arange(len(signal)) / sampling_rate
        
        ax.plot(time, signal, linewidth=1.5, color='steelblue')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (seconds)', fontsize=9)
        
        if i == 0:
            ax.set_ylabel(f'Class {class_id}\n{class_names[class_id]}\nAmplitude', 
                         fontweight='bold', fontsize=9)
        
        if class_id == 0:
            ax.set_title(f'Sample {i+1}', fontsize=10)

plt.tight_layout()
output_file = 'ecg_heartbeat_patterns.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved as '{output_file}'")
print(f"✓ Opening image...")
plt.show()

print("\n" + "=" * 60)
print("=" * 60)