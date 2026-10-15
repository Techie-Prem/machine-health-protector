"""
Predictive Maintenance System - Model Training & Calibration Pipeline
Supports raw multi-channel vibration time-series with sliding-window feature extraction.
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime

# Windows Console UTF-8 safety
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# ==========================================
# SLIDING WINDOW & FEATURE EXTRACTION
# ==========================================
def create_windows(signal, window_size=256, step=128):
    """Slices a 1D continuous signal array into overlapping sliding windows."""
    n_samples = len(signal)
    if n_samples < window_size:
        return []
    return [
        signal[i:i + window_size]
        for i in range(0, n_samples - window_size + 1, step)
    ]


def extract_features(window):
    """
    Extracts 7 fundamental statistical time-domain features from a signal window:
    1. Mean
    2. Standard Deviation
    3. Root Mean Square (RMS)
    4. Minimum Value
    5. Maximum Value
    6. Skewness
    7. Kurtosis
    """
    feats = [
        np.mean(window),
        np.std(window),
        np.sqrt(np.mean(window ** 2)),  # RMS
        np.min(window),
        np.max(window),
        float(skew(window)),
        float(kurtosis(window))
    ]
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).tolist()


def process_dataframe(df, sensors, window_size=256, step=128):
    """
    Extracts statistical features across all specified sensor channels
    using synchronised sliding windows.
    """
    sensor_windows = []
    for col in sensors:
        if col not in df.columns:
            raise ValueError(f"Sensor column '{col}' not found in dataframe. Available columns: {list(df.columns)}")
        sensor_windows.append(create_windows(df[col].values, window_size=window_size, step=step))

    num_windows = min(len(w) for w in sensor_windows) if sensor_windows else 0
    if num_windows == 0:
        raise ValueError(f"Dataframe has {len(df)} samples, which is insufficient for window_size={window_size}.")

    X = []
    for i in range(num_windows):
        feats = []
        for sw in sensor_windows:
            feats.extend(extract_features(sw[i]))
        X.append(feats)

    return np.array(X)


# ==========================================
# SYNTHETIC BENCHMARK GENERATOR
# ==========================================
def generate_synthetic_dataset(output_dir="Data", n_samples=25000):
    """Generates a realistic 4-channel continuous vibration benchmark dataset."""
    os.makedirs(output_dir, exist_ok=True)
    t = np.linspace(0, 100, n_samples)
    np.random.seed(42)

    print(f"[DATA] Generating synthetic vibration benchmark in '{output_dir}/' ({n_samples} samples per state)...")

    # Healthy Condition (stable harmonics, low noise, balanced amplitudes)
    healthy_df = pd.DataFrame({
        "a1": 1.85 * np.sin(2 * np.pi * 24 * t) + 0.3 * np.sin(2 * np.pi * 48 * t) + np.random.normal(0, 0.08, n_samples),
        "a2": 2.10 * np.cos(2 * np.pi * 24 * t) + 0.25 * np.sin(2 * np.pi * 72 * t) + np.random.normal(0, 0.09, n_samples),
        "a3": 3.20 * np.sin(2 * np.pi * 32 * t) + 0.4 * np.sin(2 * np.pi * 64 * t) + np.random.normal(0, 0.12, n_samples),
        "a4": 2.65 * np.cos(2 * np.pi * 32 * t) + 0.35 * np.cos(2 * np.pi * 96 * t) + np.random.normal(0, 0.10, n_samples),
    })

    # Broken/Faulty Condition (severe harmonic distortion, impulsive impacts, bearing raceway noise)
    broken_df = pd.DataFrame({
        "a1": 0.05 * np.sin(2 * np.pi * 12 * t) + 0.9 * np.random.normal(0, 0.35, n_samples) * (np.random.rand(n_samples) > 0.94),
        "a2": 0.04 * np.cos(2 * np.pi * 12 * t) + 0.8 * np.random.normal(0, 0.30, n_samples) * (np.random.rand(n_samples) > 0.93),
        "a3": 0.03 * np.sin(2 * np.pi * 18 * t) + 1.2 * np.random.normal(0, 0.45, n_samples) * (np.random.rand(n_samples) > 0.91),
        "a4": 0.05 * np.cos(2 * np.pi * 18 * t) + 1.0 * np.random.normal(0, 0.40, n_samples) * (np.random.rand(n_samples) > 0.92),
    })

    healthy_path = os.path.join(output_dir, "fullHealthy.csv")
    broken_path = os.path.join(output_dir, "fullBroken.csv")
    healthy_df.to_csv(healthy_path, index=False)
    broken_df.to_csv(broken_path, index=False)

    print(f"[OK] Generated '{healthy_path}' ({len(healthy_df):,} rows)")
    print(f"[OK] Generated '{broken_path}' ({len(broken_df):,} rows)")
    return healthy_path, broken_path


# ==========================================
# DATA LOADING & RESOLUTION
# ==========================================
def find_dataset_files(data_dir="Data", healthy_arg=None, broken_arg=None, dataset_arg=None):
    """Resolves dataset paths automatically from Data directory or CLI arguments."""
    if dataset_arg and os.path.exists(dataset_arg):
        return None, None, dataset_arg

    # Check explicit arguments
    if healthy_arg and broken_arg:
        h_path = os.path.join(data_dir, healthy_arg) if not os.path.isabs(healthy_arg) else healthy_arg
        b_path = os.path.join(data_dir, broken_arg) if not os.path.isabs(broken_arg) else broken_arg
        if os.path.exists(h_path) and os.path.exists(b_path):
            return h_path, b_path, None

    # Search inside data_dir
    if not os.path.exists(data_dir):
        return None, None, None

    # Look for known healthy / broken file patterns
    h_candidates = glob.glob(os.path.join(data_dir, "*healthy*.csv")) + glob.glob(os.path.join(data_dir, "*normal*.csv"))
    b_candidates = glob.glob(os.path.join(data_dir, "*broken*.csv")) + glob.glob(os.path.join(data_dir, "*fault*.csv"))

    if h_candidates and b_candidates:
        return h_candidates[0], b_candidates[0], None

    # Check for single combined dataset
    all_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    if len(all_csvs) == 1:
        return None, None, all_csvs[0]

    return None, None, None


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def run_training_pipeline(
    data_dir="Data",
    healthy_file=None,
    broken_file=None,
    dataset_file=None,
    label_col="condition",
    sensors=None,
    window_size=256,
    step=128,
    model_choice="best",
    generate_synthetic_if_empty=False
):
    print("=" * 70)
    print("[START] PREDICTIVE MAINTENANCE ML MODEL TRAINING & CALIBRATION")
    print("=" * 70)

    # 1. Resolve Dataset Files
    h_path, b_path, d_path = find_dataset_files(data_dir, healthy_file, broken_file, dataset_file)

    if h_path is None and b_path is None and d_path is None:
        if generate_synthetic_if_empty:
            h_path, b_path = generate_synthetic_dataset(data_dir)
        else:
            print(f"[ERROR] No dataset files found in '{data_dir}/'.")
            print("  Please place your CSV files in the 'Data/' folder (e.g., fullHealthy.csv and fullBroken.csv),")
            print("  or run with '--generate-synthetic' to generate a benchmark dataset.")
            sys.exit(1)

    # 2. Ingest Data
    sensors_list: list[str] = list(sensors) if sensors is not None else ["a1", "a2", "a3", "a4"]

    if h_path and b_path:
        print(f"[LOAD] Healthy dataset : {h_path}")
        print(f"[LOAD] Broken dataset  : {b_path}")
        healthy_df = pd.read_csv(h_path)
        broken_df = pd.read_csv(b_path)

        # Detect or validate sensor channels
        if sensors is None:
            common_cols = [c for c in healthy_df.columns if c in broken_df.columns]
            numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(healthy_df[c])]
            detected_sensors = [c for c in numeric_cols if c.lower() not in ['time', 'timestamp', 'index', 'id', 'label', 'target', 'condition']][:4]
            if detected_sensors:
                sensors_list = detected_sensors

        print(f"[CONFIG] Monitored Sensor Channels: {sensors_list}")

        # Compute baseline amplitudes for app synchronization
        normal_vals = [float(np.mean(np.abs(healthy_df[s].values))) for s in sensors_list]
        fault_vals  = [float(np.mean(np.abs(broken_df[s].values))) for s in sensors_list]

        print(f"[BASELINE] Normal Amplitudes : {normal_vals}")
        print(f"[BASELINE] Fault Amplitudes  : {fault_vals}")

        # Extract features per condition
        print(f"[EXTRACT] Extracting features (window_size={window_size}, step={step})...")
        X_healthy = process_dataframe(healthy_df, sensors_list, window_size, step)
        X_broken  = process_dataframe(broken_df, sensors_list, window_size, step)

        y_healthy = np.zeros(len(X_healthy))
        y_broken  = np.ones(len(X_broken))

        X = np.asarray(np.vstack([X_healthy, X_broken]), dtype=np.float64)
        y = np.asarray(np.hstack([y_healthy, y_broken]), dtype=np.float64)

    elif d_path:
        print(f"[LOAD] Combined dataset: {d_path}")
        df = pd.read_csv(d_path)
        if label_col not in df.columns:
            potential_labels = [c for c in df.columns if c.lower() in ['label', 'target', 'condition', 'status', 'health', 'fault']]
            if potential_labels:
                label_col = potential_labels[0]
            else:
                raise ValueError(f"Label column '{label_col}' not found in {d_path}. Available: {list(df.columns)}")

        print(f"[CONFIG] Target Column: '{label_col}'")
        if sensors is None:
            detected_sensors = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])][:4]
            if detected_sensors:
                sensors_list = detected_sensors

        print(f"[CONFIG] Monitored Sensor Channels: {sensors_list}")

        healthy_df = df[df[label_col].isin([0, 'healthy', 'Healthy', 'normal', 'Normal'])]
        broken_df  = df[df[label_col].isin([1, 'broken', 'Broken', 'faulty', 'Faulty', 'failure', 'Failure'])]

        normal_vals = [float(np.mean(np.abs(healthy_df[s].values))) for s in sensors_list]
        fault_vals  = [float(np.mean(np.abs(broken_df[s].values))) for s in sensors_list]

        X_healthy = process_dataframe(healthy_df, sensors_list, window_size, step)
        X_broken  = process_dataframe(broken_df, sensors_list, window_size, step)

        y_healthy = np.zeros(len(X_healthy))
        y_broken  = np.ones(len(X_broken))

        X = np.asarray(np.vstack([X_healthy, X_broken]), dtype=np.float64)
        y = np.asarray(np.hstack([y_healthy, y_broken]), dtype=np.float64)

    else:
        raise RuntimeError(f"Unable to load training dataset from '{data_dir}'.")

    print(f"[OK] Total Extracted Windows: {X.shape[0]:,} (Healthy: {len(y_healthy):,}, Faulty: {len(y_broken):,})")
    print(f"[OK] Feature Vector Dimension: {X.shape[1]} ({len(sensors_list)} sensors x 7 statistical moments)")

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Standard Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 5. PCA Transformation (Retain 95% variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca  = pca.transform(X_test_s)

    n_pca_components = X_train_pca.shape[1]
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"[PCA] Dimensionality Reduction: {X.shape[1]} features -> {n_pca_components} components ({explained_var:.1%} variance preserved)")

    # 6. Candidate Models
    candidate_models = {
        "SVM (RBF)": SVC(C=10, gamma="scale", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_split=50, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42)
    }

    # 7. Model Evaluation & Benchmarking
    model_accuracies: dict[str, float] = {}
    model_f1_scores: dict[str, float] = {}

    print("\n" + "=" * 70)
    print("[EVAL] MODEL EVALUATION & BENCHMARKING:")
    print("=" * 70)

    for name, model in candidate_models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        model_accuracies[name] = acc
        model_f1_scores[name] = f1
        print(f"  {name:<22} | Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")

    # 8. Selection of Champion Model
    if model_choice.lower() == "best":
        best_model_name = max(model_accuracies, key=lambda k: (model_accuracies[k], model_f1_scores[k]))
    else:
        matched = [k for k in candidate_models.keys() if model_choice.lower() in k.lower()]
        best_model_name = matched[0] if matched else "SVM (RBF)"

    best_acc: float = model_accuracies[best_model_name]
    print("\n" + "=" * 70)
    print(f"[CHAMPION] CHAMPION MODEL SELECTED: {best_model_name} (Accuracy: {best_acc:.2%})")
    print("=" * 70)

    # 9. Retrain Champion Model on Full Dataset
    print(f"\n[RETRAIN] Retraining {best_model_name} on entire dataset (train + test)...")
    X_all_s = scaler.fit_transform(X)  # type: ignore[arg-type]
    X_all_pca = pca.fit_transform(X_all_s)  # type: ignore[arg-type]

    champion_model = candidate_models[best_model_name]
    champion_model.fit(X_all_pca, y)

    # 10. Compute Manifold Centroids & Cluster Reference Samples
    mask_h = (y == 0)
    mask_b = (y == 1)

    mean_h_pca = np.asarray(np.mean(X_all_pca[mask_h], axis=0).reshape(1, -1), dtype=np.float64)
    mean_b_pca = np.asarray(np.mean(X_all_pca[mask_b], axis=0).reshape(1, -1), dtype=np.float64)

    f_healthy = scaler.inverse_transform(pca.inverse_transform(mean_h_pca))[0].tolist()  # type: ignore[arg-type]
    f_broken  = scaler.inverse_transform(pca.inverse_transform(mean_b_pca))[0].tolist()  # type: ignore[arg-type]

    sample_size = min(250, int(np.sum(mask_h)), int(np.sum(mask_b)))
    sample_indices_h = np.random.choice(np.where(mask_h)[0], size=sample_size, replace=False)
    sample_indices_b = np.random.choice(np.where(mask_b)[0], size=sample_size, replace=False)

    sv_healthy_pca = X_all_pca[sample_indices_h, :2].tolist()
    sv_broken_pca  = X_all_pca[sample_indices_b, :2].tolist()

    # 11. Save Serialized Artifacts
    joblib.dump(champion_model, "trained_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")

    metadata = {
        "model_type": best_model_name,
        "accuracy": float(best_acc),
        "test_accuracy": float(best_acc),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features_per_sensor": 7,
        "sensors": sensors_list,
        "feature_names": sensors_list,
        "target_classes": ["healthy", "broken"],
        "normal_vals": normal_vals,
        "fault_vals": fault_vals,
        "f_healthy": f_healthy,
        "f_broken": f_broken,
        "sv_healthy_pca": sv_healthy_pca,
        "sv_broken_pca": sv_broken_pca,
        "window_size": int(window_size),
        "step": int(step),
        "trained_date": datetime.now().isoformat()
    }

    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n[SAVED] Model artifacts saved successfully:")
    print("   * trained_model.pkl")
    print("   * scaler.pkl")
    print("   * pca.pkl")
    print("   * model_metadata.json")
    print("\n[DONE] Training pipeline completed successfully!\n")


# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Maintenance ML Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to data directory (default: Data)")
    parser.add_argument("--healthy", type=str, default=None, help="Filename of healthy CSV inside data-dir")
    parser.add_argument("--broken", type=str, default=None, help="Filename of broken/faulty CSV inside data-dir")
    parser.add_argument("--dataset", type=str, default=None, help="Path to single combined CSV dataset with target column")
    parser.add_argument("--label-col", type=str, default="condition", help="Target column name if using single combined CSV")
    parser.add_argument("--sensors", type=str, default=None, help="Comma-separated sensor names (e.g. a1,a2,a3,a4)")
    parser.add_argument("--window-size", type=int, default=256, help="Sliding window size (default: 256)")
    parser.add_argument("--step", type=int, default=128, help="Sliding window step/stride (default: 128)")
    parser.add_argument("--model-choice", type=str, default="best", help="Model choice: 'best', 'svm', 'rf', 'gb', 'knn', 'dt', 'lr'")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic benchmark dataset if no files found")

    args = parser.parse_args()

    sensors_list = [s.strip() for s in args.sensors.split(",")] if args.sensors else None

    run_training_pipeline(
        data_dir=args.data_dir,
        healthy_file=args.healthy,
        broken_file=args.broken,
        dataset_file=args.dataset,
        label_col=args.label_col,
        sensors=sensors_list,
        window_size=args.window_size,
        step=args.step,
        model_choice=args.model_choice,
        generate_synthetic_if_empty=args.generate_synthetic
    )
