import kagglehub
import os

print("Downloading ECG heartbeat dataset from Kaggle...")
path = kagglehub.dataset_download("shayanfazeli/heartbeat")

print(f"\n✓ Dataset downloaded successfully!")
print(f"Path to dataset files: {path}")
print(f"\nCopy this path for next step:")
print(f">>> {path}")
print(f"\nFiles in dataset:")
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  - {file} ({size_mb:.2f} MB)")