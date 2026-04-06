import numpy as np
import pandas as pd
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# WAVELET FEATURE EXTRACTION

def extract_wavelet_features(signal, wavelet='db4', level=4):
    """
    Extract features using Discrete Wavelet Transform.
    
    Wavelets capture both time AND frequency information - 
    perfect for ECG where timing of events matters!
    """
    # Decompose signal using DWT
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    features = {}
    
    # Extract statistics from each decomposition level
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
        features[f'wavelet_std_level_{i}'] = np.std(coeff)
        features[f'wavelet_mean_level_{i}'] = np.mean(np.abs(coeff))
    
    return features


def add_wavelet_features(filepath):
    """
    Load existing features and add wavelet features.
    """
    print("=" * 60)
    print("ADDING WAVELET FEATURES")
    print("=" * 60)
    
    # Load filtered signals
    print("\n[1/3] Loading filtered signals...")
    signals_df = pd.read_csv('mitbih_train_filtered.csv', header=None)
    signals = signals_df.iloc[:, :-1].values
    labels = signals_df.iloc[:, -1].values
    
    # Load existing features
    print("[2/3] Loading existing features...")
    existing_features = pd.read_csv(filepath)
    existing_features = existing_features.drop('label', axis=1)
    
    print(f"  Existing features: {len(existing_features.columns)}")
    
    # Extract wavelet features
    print("[3/3] Extracting wavelet features...")
    wavelet_features_list = []
    
    for i, signal in enumerate(signals):
        wf = extract_wavelet_features(signal)
        wavelet_features_list.append(wf)
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(signals)}...")
    
    wavelet_df = pd.DataFrame(wavelet_features_list)
    
    print(f"  Wavelet features: {len(wavelet_df.columns)}")
    
    # Combine features
    combined = pd.concat([existing_features.reset_index(drop=True), 
                          wavelet_df.reset_index(drop=True)], axis=1)
    combined['label'] = labels
    
    print(f"\n✓ Total features: {len(combined.columns) - 1}")
    
    # Save
    output_file = 'ecg_features_enhanced.csv'
    combined.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    return combined

# MODEL COMPARISON

def compare_models():
    """
    Train models with and without wavelet features - compare performance.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: Baseline vs Enhanced")
    print("=" * 60)
    
    # Load both feature sets
    baseline_df = pd.read_csv('ecg_features.csv')
    enhanced_df = pd.read_csv('ecg_features_enhanced.csv')
    
    results = {}
    
    for name, df in [('Baseline', baseline_df), ('Enhanced', enhanced_df)]:
        print(f"\n{'='*60}")
        print(f"Training {name} Model")
        print(f"{'='*60}")
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        print(f"Features: {len(X.columns)}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        clf = RandomForestClassifier(n_estimators=100, max_depth=20, 
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {acc*100:.2f}%")
        
        results[name] = {
            'accuracy': acc,
            'model': clf,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
        
        # Detailed report
        class_names = ['Normal', 'Supraventricular', 'Premature Ventricular', 
                       'Fusion', 'Unclassifiable']
        print("\n" + classification_report(y_test, y_pred, target_names=class_names))
    
    # Comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    baseline_acc = results['Baseline']['accuracy']
    enhanced_acc = results['Enhanced']['accuracy']
    improvement = (enhanced_acc - baseline_acc) * 100
    
    print(f"\nBaseline accuracy:  {baseline_acc*100:.2f}%")
    print(f"Enhanced accuracy:  {enhanced_acc*100:.2f}%")
    print(f"Improvement:        {improvement:+.2f}%")
    
    if improvement > 0:
        print("\n✓ Wavelet features improved performance!")
    elif improvement == 0:
        print("\n→ No change (wavelet features didn't help)")
    else:
        print("\n⚠ Performance decreased (overfitting possible)")
    
    # Save best model
    if enhanced_acc >= baseline_acc:
        print(f"\nSaving enhanced model...")
        import joblib
        joblib.dump(results['Enhanced']['model'], 'best_model.pkl')
        joblib.dump(results['Enhanced']['scaler'], 'scaler.pkl')
        pd.Series(results['Enhanced']['feature_names']).to_csv('feature_names.csv', index=False)
        print("✓ Saved: best_model.pkl, scaler.pkl, feature_names.csv")
    else:
        print(f"\nSaving baseline model (better performance)...")
        import joblib
        joblib.dump(results['Baseline']['model'], 'best_model.pkl')
        joblib.dump(results['Baseline']['scaler'], 'scaler.pkl')
        pd.Series(results['Baseline']['feature_names']).to_csv('feature_names.csv', index=False)
        print("✓ Saved: best_model.pkl, scaler.pkl, feature_names.csv")
    
    return results

# MAIN

if __name__ == "__main__":
    # Add wavelet features
    enhanced_df = add_wavelet_features('ecg_features.csv')
    
    # Compare models
    results = compare_models()
    
    print("\n" + "=" * 60)
    print("READY FOR REAL-TIME PREDICTION!")
    print("=" * 60)
    print("\nRun: python predict_live.py")