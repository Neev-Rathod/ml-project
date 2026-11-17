"""
wp1_data_pipeline/run_wp1.py
=============================
Main entry point for Work Package 1.

What this script does
---------------------
1. Loads PTB-XL metadata (ptbxl_database.csv + scp_statements.csv).
2. Optionally loads raw waveforms and extracts the 84-dim feature vectors.
   (If a cached features.npz already exists, it skips extraction.)
3. Saves features.npz and labels.npz to --out_dir so other WPs can load them.
4. Runs the sensor noise & subspace analysis (k-Means purity, PCA scree plot).

Usage
-----
    python run_wp1.py --data_path D:\ml-project\dataset --out_dir ..\outputs\wp1

    # Skip waveform loading if features.npz already exists:
    python run_wp1.py --data_path D:\ml-project\dataset \
                      --out_dir ..\outputs\wp1          \
                      --skip_extraction

Arguments
---------
--data_path       Path to the PTB-XL root directory (required)
--out_dir         Where to save features.npz / plots  (default: outputs/wp1)
--sampling_rate   100 or 500                          (default: 100)
--skip_extraction Load pre-existing features.npz instead of re-extracting
--max_records     Limit to first N records (useful for quick smoke-testing)
--seed            Random seed                          (default: 42)
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from the project root when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wp1_data_pipeline.loader import load_metadata, load_waveforms
from wp1_data_pipeline.feature_extractor import (
    extract_features_batch,
    feature_names,
)
from wp1_data_pipeline.noise_analysis import run_noise_analysis
from shared.data_utils import encode_labels, assign_gold_untrusted


def parse_args():
    p = argparse.ArgumentParser(description="WP1 – Data Pipeline & Sensor Noise Analysis")
    p.add_argument("--data_path",       required=False,
                   default=os.environ.get("PTBXL_PATH", ""),
                   help="Path to PTB-XL root (or set PTBXL_PATH env var)")
    p.add_argument("--out_dir",         default=os.path.join("..", "outputs", "wp1"))
    p.add_argument("--sampling_rate",   type=int, default=100, choices=[100, 500])
    p.add_argument("--skip_extraction", action="store_true",
                   help="Load features.npz from out_dir instead of re-extracting")
    p.add_argument("--max_records",     type=int, default=None,
                   help="Limit records for quick testing (e.g. 500)")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.data_path:
        print("ERROR: Provide --data_path or set PTBXL_PATH environment variable.")
        sys.exit(1)

    data_path = Path(args.data_path)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_file = out_dir / "features.npz"
    labels_file   = out_dir / "labels.npz"

    # ── Step 1: Load metadata ─────────────────────────────────────────────
    print("[WP1] Loading PTB-XL metadata …")
    meta = load_metadata(data_path)
    print(f"[WP1] {len(meta)} records found.")

    # Diagnostic superclass distribution
    dist = meta["diagnostic_superclass"].value_counts(dropna=False)
    print("[WP1] Diagnostic superclass distribution:")
    print(dist.to_string())

    # Noise flag summary
    for flag in ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]:
        n = meta[flag].fillna(False).astype(bool).sum()
        print(f"[WP1]   {flag}: {n} records ({100 * n / len(meta):.1f}%)")

    # ── Step 2: Feature extraction (or load cache) ─────────────────────
    if args.skip_extraction and features_file.exists() and labels_file.exists():
        print(f"[WP1] Loading cached features from {features_file} …")
        feat_data  = np.load(features_file, allow_pickle=True)
        label_data = np.load(labels_file,   allow_pickle=True)
        X          = feat_data["X"]
        y          = label_data["y"]
        fold       = label_data["fold"]
        gold_mask  = label_data["gold_mask"]

    else:
        print("[WP1] Extracting features from raw waveforms …")

        # Optionally limit record count
        if args.max_records:
            meta = meta.iloc[: args.max_records]

        waveforms = load_waveforms(
            meta, data_path, sampling_rate=args.sampling_rate
        )

        X = extract_features_batch(waveforms, fs=args.sampling_rate)

        # Encode labels
        y = encode_labels(meta["diagnostic_superclass"])

        fold = meta["strat_fold"].values.astype(int)

        # Build gold / untrusted mask for training folds
        train_rows = meta[meta["strat_fold"].isin(range(1, 9))]
        gm_train, _ = assign_gold_untrusted(train_rows)
        gold_mask = np.zeros(len(meta), dtype=bool)
        train_idx = meta.index.isin(train_rows.index)
        # Align: gm_train is in train row order
        gold_mask[np.where(train_idx)[0]] = gm_train

        # Save
        fnames = np.array(feature_names())
        np.savez_compressed(
            features_file,
            X=X,
            feature_names=fnames,
        )
        np.savez_compressed(
            labels_file,
            y=y,
            fold=fold,
            gold_mask=gold_mask,
            ecg_ids=meta.index.values,
        )
        print(f"[WP1] Saved features  → {features_file}")
        print(f"[WP1] Saved labels    → {labels_file}")
        print(f"[WP1] Feature matrix shape: {X.shape}")
        print(f"[WP1] Gold Set size (train folds): {gold_mask[np.isin(fold, range(1,9))].sum()}")
        print(f"[WP1] Untrusted Pool size (train folds): {(~gold_mask[np.isin(fold, range(1,9))]).sum()}")

    # ── Step 3: Noise analysis ────────────────────────────────────────────
    print("[WP1] Running sensor noise & subspace analysis …")
    run_noise_analysis(
        X       = X,
        y       = y,
        fold    = fold,
        meta    = meta,
        out_dir = out_dir,
        seed    = args.seed,
    )

    print("\n[WP1] Done. Outputs written to:", out_dir)
    print("[WP1] ─────────────────────────────────────────────────────────")
    print("[WP1] Other WPs should load these files:")
    print(f"[WP1]   features : {features_file}")
    print(f"[WP1]   labels   : {labels_file}")


if __name__ == "__main__":
    main()
