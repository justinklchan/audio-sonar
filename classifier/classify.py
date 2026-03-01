#!/usr/bin/env python3
"""Train and evaluate a classifier on pulse FFT snapshots.

Reads meta.txt for label assignments, loads FFT CSVs from ../recordings/,
removes outlier spectra per class, trains a classifier, and reports accuracy.
"""

import sys
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: classify.py [options]

Train and evaluate a classifier on pulse FFT snapshots.

Reads meta.txt in the same directory for file-label mappings.
Format: one line per file, "filename-label"

Outlier detection is applied per class before training. Spectra whose
median distance to the class centroid exceeds 2x the MAD (median absolute
deviation) are removed. This eliminates wildly varying or anomalous spectra.

Options:
  --meta PATH       Path to meta.txt (default: meta.txt in script dir)
  --recordings DIR  Path to recordings directory (default: ../recordings)
  --save MODEL      Save trained model to file (pickle)
  --no-outlier      Disable outlier removal
  --threshold N     MAD multiplier for outlier cutoff (default: 2.0)
  --help, -h        Show this help message
""")
    sys.exit(0)

import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle

# ── Parse arguments ──────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
meta_path = script_dir / "meta.txt"
rec_dir = script_dir.parent / "recordings"
save_path = None
do_outlier = True
outlier_threshold = 2.0

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--meta" and i + 1 < len(args):
        meta_path = Path(args[i + 1])
        i += 2
    elif args[i] == "--recordings" and i + 1 < len(args):
        rec_dir = Path(args[i + 1])
        i += 2
    elif args[i] == "--save" and i + 1 < len(args):
        save_path = Path(args[i + 1])
        i += 2
    elif args[i] == "--no-outlier":
        do_outlier = False
        i += 1
    elif args[i] == "--threshold" and i + 1 < len(args):
        outlier_threshold = float(args[i + 1])
        i += 2
    else:
        print(f"Unknown argument: {args[i]}")
        sys.exit(1)

# ── Load meta.txt ────────────────────────────────────────────────
entries = []
with open(meta_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Format: filename-label (split on last hyphen)
        last_dash = line.rfind("-")
        if last_dash == -1:
            print(f"Skipping malformed line: {line}")
            continue
        filename = line[:last_dash]
        label = line[last_dash + 1:]
        entries.append((filename, label))

print(f"Loaded {len(entries)} entries from {meta_path}")
for fname, label in entries:
    print(f"  {label:>10s} <- {fname}")

# ── Load FFT data ───────────────────────────────────────────────
def load_fft_csv(filepath):
    """Load FFT CSV, return (freqs, timestamps, magnitude_db_matrix)."""
    with open(filepath) as f:
        header = f.readline().strip()
    cols = header.split(",")
    freqs = np.array([float(c) for c in cols[1:]])
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    timestamps = data[:, 0]
    magnitude_db = data[:, 1:]
    return freqs, timestamps, magnitude_db

X_all = []
y_all = []
file_indices = []  # track which file each sample came from

for idx, (filename, label) in enumerate(entries):
    filepath = rec_dir / filename
    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping")
        continue
    freqs, timestamps, mag_db = load_fft_csv(filepath)
    n_snapshots = mag_db.shape[0]
    print(f"  Loaded {filename}: {n_snapshots} snapshots, {mag_db.shape[1]} freq bins")
    for row in mag_db:
        X_all.append(row)
        y_all.append(label)
        file_indices.append(idx)

X = np.array(X_all)
y = np.array(y_all)
file_indices = np.array(file_indices)

# Restrict to 7-16 kHz range
freq_mask = (freqs >= 7000) & (freqs <= 16000)
X = X[:, freq_mask]
freqs = freqs[freq_mask]
print(f"  Cropped to {freqs[0]:.0f}-{freqs[-1]:.0f} Hz: {X.shape[1]} bins")

print(f"\nDataset before filtering: {X.shape[0]} samples, {X.shape[1]} features")
labels_unique, counts = np.unique(y, return_counts=True)
for label, count in zip(labels_unique, counts):
    print(f"  {label}: {count} samples")

# ── Outlier removal (per class, MAD-based) ───────────────────────
def remove_outliers(X, y, threshold=2.0):
    """Remove outlier spectra per class using MAD on distance to centroid.

    For each class:
    1. Compute the centroid (median spectrum).
    2. Compute each spectrum's Euclidean distance to the centroid.
    3. Compute the MAD (median absolute deviation) of those distances.
    4. Remove spectra where distance > median_dist + threshold * MAD.

    Returns mask of inliers (True = keep).
    """
    keep = np.ones(len(X), dtype=bool)
    for label in np.unique(y):
        cls_mask = y == label
        cls_X = X[cls_mask]

        # Centroid = median (robust to outliers unlike mean)
        centroid = np.median(cls_X, axis=0)

        # Euclidean distance from each spectrum to centroid
        dists = np.linalg.norm(cls_X - centroid, axis=1)

        # MAD-based cutoff
        median_dist = np.median(dists)
        mad = np.median(np.abs(dists - median_dist))
        if mad < 1e-9:
            # All distances nearly identical, nothing to remove
            continue
        cutoff = median_dist + threshold * mad

        # Mark outliers
        cls_indices = np.where(cls_mask)[0]
        outlier_mask = dists > cutoff
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"  {label:>10s}: removing {n_outliers}/{len(cls_X)} outliers "
                  f"(cutoff={cutoff:.1f}, max_dist={dists.max():.1f}, "
                  f"median_dist={median_dist:.1f}, MAD={mad:.1f})")
            keep[cls_indices[outlier_mask]] = False
        else:
            print(f"  {label:>10s}: no outliers (max_dist={dists.max():.1f}, cutoff={cutoff:.1f})")

    return keep

if do_outlier:
    print(f"\n--- Outlier Detection (MAD threshold={outlier_threshold:.1f}) ---")
    inlier_mask = remove_outliers(X, y, threshold=outlier_threshold)
    n_removed = (~inlier_mask).sum()
    X = X[inlier_mask]
    y = y[inlier_mask]
    file_indices = file_indices[inlier_mask]
    print(f"  Removed {n_removed} outliers total, {X.shape[0]} samples remaining")
else:
    print("\nOutlier detection disabled.")

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
labels_unique, counts = np.unique(y, return_counts=True)
for label, count in zip(labels_unique, counts):
    print(f"  {label}: {count} samples")

# ── Compute per-class centroids and outlier cutoffs (on cleaned data) ──
# These are saved into the model so real-time classification can reject
# incoming spectra that don't resemble any known class.
print("\n--- Per-class outlier stats (for real-time rejection) ---")
class_centroids = {}
class_outlier_cutoffs = {}
for label in sorted(np.unique(y)):
    cls_X = X[y == label]
    centroid = np.median(cls_X, axis=0)
    dists = np.linalg.norm(cls_X - centroid, axis=1)
    median_dist = np.median(dists)
    mad = np.median(np.abs(dists - median_dist))
    # Use a generous cutoff for real-time (3x MAD) since live conditions vary
    cutoff = median_dist + 3.0 * mad if mad > 1e-9 else dists.max() * 1.5
    class_centroids[label] = centroid
    class_outlier_cutoffs[label] = float(cutoff)
    print(f"  {label:>10s}: median_dist={median_dist:.1f}  MAD={mad:.1f}  cutoff={cutoff:.1f}")

# ── Feature scaling ──────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train and evaluate ───────────────────────────────────────────
# Stratified 5-fold CV (snapshots from each class split across folds)
print("\n--- Stratified 5-Fold Cross-Validation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred_lofo = cross_val_predict(clf_cv, X_scaled, y, cv=skf)
y_prob_lofo = cross_val_predict(clf_cv, X_scaled, y, cv=skf, method='predict_proba')

print(classification_report(y, y_pred_lofo, zero_division=0))

# Per-sample confidence (probability of predicted class)
confidence = np.max(y_prob_lofo, axis=1)
print(f"Confidence: mean={confidence.mean():.3f}  min={confidence.min():.3f}  max={confidence.max():.3f}")
print(f"  Per class:")
for label in sorted(np.unique(y)):
    mask = y == label
    c = confidence[mask]
    print(f"    {label:>8s}: mean={c.mean():.3f}  min={c.min():.3f}")
print("Confusion matrix:")
cm_labels = sorted(np.unique(y))
cm = confusion_matrix(y, y_pred_lofo, labels=cm_labels)
# Print with labels
header = "          " + " ".join(f"{l:>8s}" for l in cm_labels)
print(header)
for i, label in enumerate(cm_labels):
    row = f"{label:>8s}  " + " ".join(f"{cm[i, j]:8d}" for j in range(len(cm_labels)))
    print(row)

# ── Train final model on all data ────────────────────────────────
print("\n--- Training final model on all data ---")
clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
clf_final.fit(X_scaled, y)

# Feature importance - show top frequency bands
importances = clf_final.feature_importances_
top_k = 20
top_indices = np.argsort(importances)[-top_k:][::-1]
print(f"\nTop {top_k} most important frequency bins:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. {freqs[idx]:8.1f} Hz  (importance: {importances[idx]:.4f})")

# ── Compute per-class confidence thresholds from CV ──────────────
conf_thresholds = {}
for label in sorted(np.unique(y)):
    mask = y == label
    c = confidence[mask]
    conf_thresholds[label] = {
        "min": float(c.min()),
        "mean": float(c.mean()),
        "std": float(c.std()),
    }

# ── Save model ───────────────────────────────────────────────────
model_data = {
    "classifier": clf_final,
    "scaler": scaler,
    "freqs": freqs,
    "labels": cm_labels,
    "freq_range_hz": (7000, 16000),
    "confidence_thresholds": conf_thresholds,
    "class_centroids": class_centroids,
    "class_outlier_cutoffs": class_outlier_cutoffs,
}
save_dest = save_path or (script_dir / "model.pkl")
with open(save_dest, "wb") as f:
    pickle.dump(model_data, f)
print(f"\nModel saved to {save_dest}")
print(f"  Keys: {list(model_data.keys())}")
print(f"  Confidence thresholds:")
for label, vals in conf_thresholds.items():
    print(f"    {label:>8s}: min={vals['min']:.3f}  mean={vals['mean']:.3f}  std={vals['std']:.3f}")
