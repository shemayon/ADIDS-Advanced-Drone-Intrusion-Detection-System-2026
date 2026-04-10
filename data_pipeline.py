"""
data_pipeline.py  —  A-DIDS Data Engineering Pipeline
Loads 137 raw CSVs from the ISOT Drone Dataset, infers attack_type
from the directory name, and exports a clean Parquet file.

Usage (from repo root):
    python3 data_pipeline.py [--base-path PATH] [--output PATH]
"""

import argparse
import glob
import os
import sys

import pandas as pd

# Add repo root to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import FEATURES, BENIGN_DIRS, DATA_PATH

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="A-DIDS Data Pipeline")
parser.add_argument("--base-path", default="extracted",
                    help="Root directory containing the extracted ISOT dataset")
parser.add_argument("--output", default=DATA_PATH,
                    help="Output path for the Parquet file")
args = parser.parse_args()

print("=" * 60)
print("  A-DIDS Data Pipeline")
print("=" * 60)

# ── 1. Discover CSV files ─────────────────────────────────────
pattern = os.path.join(args.base_path, "**", "new_feature_csv", "**", "*.csv")
files = glob.glob(pattern, recursive=True)
print(f"\n[1/4] Discovered {len(files)} CSV files in: {args.base_path}")

if not files:
    print(f"[ERROR] No CSVs found. Check that the dataset is extracted to '{args.base_path}'.")
    sys.exit(1)

# ── 2. Load & label each file ─────────────────────────────────
print(f"\n[2/4] Loading and labelling files ...")
dfs = []
skipped = 0

for f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        print(f"  [WARN] Skipping {f}: {e}")
        skipped += 1
        continue

    # Infer attack type from parent directory name
    attack_type = os.path.basename(os.path.dirname(f))
    df["attack_type"] = "benign" if attack_type in BENIGN_DIRS else attack_type
    dfs.append(df)

print(f"  Loaded: {len(dfs)} files  |  Skipped: {skipped}")

# ── 3. Merge, label, filter ───────────────────────────────────
print(f"\n[3/4] Merging and preprocessing ...")
df = pd.concat(dfs, ignore_index=True)
print(f"  Raw shape: {df.shape}")

df["label"] = (df["attack_type"] != "benign").astype(int)

# Validate all selected features exist
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"[ERROR] Missing feature columns: {missing}")
    sys.exit(1)

df = df[FEATURES + ["label"]]
df = df.dropna()

print(f"  Final shape: {df.shape}")
print(f"  Benign rows : {(df['label'] == 0).sum():,}")
print(f"  Attack rows : {(df['label'] == 1).sum():,}")

# ── 4. Export Parquet ─────────────────────────────────────────
print(f"\n[4/4] Saving to: {args.output}")
os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
df.to_parquet(args.output, index=False)
print(f"  [✓] Saved: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n[DONE] Dataset ready at: {args.output}")
