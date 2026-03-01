#!/usr/bin/env python3
"""Plot average FFT per class with shaded spread (after outlier removal)."""

import sys
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: plot_classes.py [--meta PATH] [--recordings DIR] [--out PATH]
                       [--no-outlier] [--threshold N]

Plot average FFT per class with shaded +/- 1 std spread.
Outlier spectra are removed before plotting (same logic as classify.py).
""")
    sys.exit(0)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'Helvetica'
sns.set_palette('deep')

script_dir = Path(__file__).resolve().parent
meta_path = script_dir / "meta.txt"
rec_dir = script_dir.parent / "recordings"
out_path = script_dir / "class_ffts.png"
do_outlier = True
outlier_threshold = 2.0

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--meta" and i + 1 < len(args):
        meta_path = Path(args[i + 1]); i += 2
    elif args[i] == "--recordings" and i + 1 < len(args):
        rec_dir = Path(args[i + 1]); i += 2
    elif args[i] == "--out" and i + 1 < len(args):
        out_path = Path(args[i + 1]); i += 2
    elif args[i] == "--no-outlier":
        do_outlier = False; i += 1
    elif args[i] == "--threshold" and i + 1 < len(args):
        outlier_threshold = float(args[i + 1]); i += 2
    else:
        i += 1

# Load meta
entries = []
with open(meta_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        last_dash = line.rfind("-")
        if last_dash == -1:
            continue
        entries.append((line[:last_dash], line[last_dash + 1:]))

# Load data
def load_fft_csv(filepath):
    with open(filepath) as f:
        header = f.readline().strip()
    cols = header.split(",")
    freqs = np.array([float(c) for c in cols[1:]])
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return freqs, data[:, 1:]

def remove_outliers_single(mag_db, threshold=2.0):
    """Remove outlier spectra from a single class using MAD."""
    centroid = np.median(mag_db, axis=0)
    dists = np.linalg.norm(mag_db - centroid, axis=1)
    median_dist = np.median(dists)
    mad = np.median(np.abs(dists - median_dist))
    if mad < 1e-9:
        return mag_db, 0
    cutoff = median_dist + threshold * mad
    keep = dists <= cutoff
    n_removed = (~keep).sum()
    return mag_db[keep], n_removed

# Collect per class
class_data = {}
freqs = None
for filename, label in entries:
    filepath = rec_dir / filename
    if not filepath.exists():
        continue
    f, mag_db = load_fft_csv(filepath)
    if freqs is None:
        freqs = f
    class_data[label] = mag_db

# Crop to 7-16 kHz
mask = (freqs >= 7000) & (freqs <= 16000)
freqs_crop = freqs[mask]

colors = sns.color_palette('deep', len(class_data))
fig, ax = plt.subplots(figsize=(12, 6))

for idx, (label, mag_db) in enumerate(class_data.items()):
    mag_crop = mag_db[:, mask]
    if do_outlier:
        mag_crop, n_removed = remove_outliers_single(mag_crop, outlier_threshold)
        if n_removed > 0:
            print(f"  {label}: removed {n_removed} outliers, {mag_crop.shape[0]} remaining")
    mean = np.mean(mag_crop, axis=0)
    std = np.std(mag_crop, axis=0)
    lo = mean - std
    hi = mean + std
    color = colors[idx]
    ax.plot(freqs_crop, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(freqs_crop, lo, hi, color=color, alpha=0.15)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Pulse FFT by Class (mean +/- 1 std, outliers removed)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
