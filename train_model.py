import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

print("=" * 60)
print("RANDOM FOREST TRAINING PIPELINE")
print("=" * 60)

print("\n[1/6] Loading features...")
df = pd.read_csv('ecg_features.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

print(f"✓ Loaded {len(X)} samples with {len(X.columns)} features")
print(f"\nClass distribution:")
for i in range(5):
    count = (y == i).sum()
    print(f"  Class {i}: {count} samples ({100*count/len(y):.1f}%)")

print("\n" + "!" * 60)
print("YOUR FEATURE PREDICTIONS")
print("!" * 60)

your_picks = ['max_peak_height', 'ratio_mid', 'kurtosis']

print(f"\nYou predicted these 3 features would be most important:")
for pick in your_picks:
    print(f"  ✓ {pick}")

print("\nLet's see if you were right...")
input("Press Enter to continue...")

#train test split

print("\n[2/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# feature scaling

print("\n[3/6] Scaling features...")
print("WHY? Features have different ranges (e.g., max_peak_height vs ratio_mid)")
print("Scaling puts them on equal footing for the ML algorithm")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features normalized to zero mean, unit variance")

#train random forest

print("\n[4/6] Training Random Forest classifier...")
print("Parameters:")
print("  - n_estimators=100 (100 decision trees)")
print("  - max_depth=20 (tree depth limit)")
print("  - random_state=42 (reproducible results)")

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

clf.fit(X_train_scaled, y_train)
print("✓ Model trained!")

#feature importance analysis

print("\n[5/6] Analyzing feature importance...")

# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Create DataFrame for easy sorting
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 60)

for i, row in importance_df.head(10).iterrows():
    print(f"{row['feature']:25s} {row['importance']:.4f}")

# Check YOUR predictions
print("\n" + "!" * 60)
print("CHECKING YOUR PREDICTIONS")
print("!" * 60)

for pick in your_picks:
    if pick in importance_df['feature'].values:
        rank = importance_df[importance_df['feature'] == pick].index[0] + 1
        importance = importance_df[importance_df['feature'] == pick]['importance'].values[0]
        
        print(f"\n{pick}:")
        print(f"  Rank: #{rank} out of {len(feature_names)}")
        print(f"  Importance: {importance:.4f}")
        
        if rank <= 5:
            print(f"  ✓ GREAT PREDICTION! This is a top-5 feature!")
        elif rank <= 10:
            print(f"  ✓ Good prediction! In top 10.")
        else:
            print(f"  ⚠ This feature ranked lower than expected.")

# Visualize feature importance
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('5_feature_importance.png', dpi=150)
print(f"\n✓ Saved: 5_feature_importance.png")

# EVALUATE MODEL

print("\n[6/6] Evaluating model performance...")

# Predictions
y_pred = clf.predict(X_test_scaled)

# Overall accuracy
accuracy = (y_pred == y_test).mean()
print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*60}")

# Per-class performance
print("\nDetailed Classification Report:")
print("-" * 60)

class_names = ['Normal', 'Supraventricular', 'Premature Ventricular', 
               'Fusion', 'Unclassifiable']

report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('6_confusion_matrix.png', dpi=150)
print(f"\n✓ Saved: 6_confusion_matrix.png")

plt.show()

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"""
Summary:
- Features used: {len(X.columns)}
- Training samples: {len(X_train)}
- Test accuracy: {accuracy*100:.2f}%

Files generated:
- 5_feature_importance.png
- 6_confusion_matrix.png
""")