# WP1 – Data Pipeline & Sensor Noise Analysis

**Person:** Neev Rathod (2024A7PS0487G)  
**Status:** ✅ Complete

---

## What This WP Does

WP1 is the **foundation** every other Work Package depends on. It:

1. Loads PTB-XL metadata and resolves each record to a single diagnostic superclass.
2. Extracts an **84-dimensional feature vector** from each raw 12-lead waveform.
3. Saves `features.npz` and `labels.npz` so WP2–WP5 never need to touch raw waveforms.
4. Runs the **Sensor Noise & Subspace Analysis** (Proposal §5.1).

**Results and analysis are documented in [`outputs/wp1/README.md`](../outputs/wp1/README.md).**

---

## Files in This Folder

| File                   | Purpose                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `run_wp1.py`           | **Entry point** – run this first                                                     |
| `loader.py`            | Reads ptbxl_database.csv, resolves diagnostic superclasses, loads raw WFDB waveforms |
| `feature_extractor.py` | Computes all 84 features (View 1 time-domain + View 2 frequency-domain)              |
| `noise_analysis.py`    | k-Means cluster purity across 6 variants; PCA scree plot                             |
| `README.md`            | This file                                                                            |

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick smoke test (~1 min)

```bash
cd wp1_data_pipeline
python run_wp1.py --data_path D:\ml-project\dataset --out_dir ..\outputs\wp1 --max_records 500
```

### Full run (all 21,799 records, ~15–30 min)

```bash
python run_wp1.py --data_path D:\ml-project\dataset --out_dir ..\outputs\wp1
```

### Re-run analysis only (features already extracted)

```bash
python run_wp1.py --data_path D:\ml-project\dataset --out_dir ..\outputs\wp1 --skip_extraction
```

---

## Outputs Produced

After a full run, `outputs/wp1/` will contain:

| File                         | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| `features.npz`               | Key `X` – shape (N, 84) float32                       |
| `labels.npz`                 | Keys `y`, `fold`, `gold_mask`, `ecg_ids`              |
| `split_summary.csv`          | Per-class record counts for each data split           |
| `feature_names.csv`          | Index → feature name + view                           |
| `feature_stats.csv`          | Mean / std / min / max per feature                    |
| `noise_analysis_results.csv` | k-Means purity and ARI for each variant               |
| `noise_cluster_purity.png`   | Bar chart: purity for all vs. clean vs. noisy records |
| `pca_explained_variance.png` | Cumulative PCA variance scree plot                    |
| `README.md`                  | Full results documentation with interpretation        |

---

## How Other WPs Load the Data

```python
from shared.data_utils import load_standard_split

split = load_standard_split(
    "outputs/wp1/features.npz",
    "outputs/wp1/labels.npz",
)

X_L, y_L       = split["X_train_L"], split["y_train_L"]   # Gold Set D_L
X_U, y_U       = split["X_train_U"], split["y_train_U"]   # Untrusted Pool D_U
X_val, y_val   = split["X_val"],     split["y_val"]
X_test, y_test = split["X_test"],    split["y_test"]
```

**Class encoding:** 0=NORM, 1=MI, 2=STTC, 3=CD, 4=HYP

**View slices for WP4 Co-Training:**

```python
X_view1 = X[:, 0:42]   # Time-domain features
X_view2 = X[:, 42:84]  # Frequency-domain features
```

---

## Feature Layout (84 dimensions)

| Indices | View   | Contents                                     |
| ------- | ------ | -------------------------------------------- |
| 0–23    | View 1 | Per-lead mean & std amplitude (2 × 12 leads) |
| 24–26   | View 1 | HRV: mean RR, std RR, RMSSD (from Lead II)   |
| 27      | View 1 | QRS amplitude range                          |
| 28      | View 1 | Zero-crossing rate (mean over leads)         |
| 29–30   | View 1 | Signal skewness, kurtosis (mean over leads)  |
| 31–33   | View 1 | Hjorth activity, mobility, complexity        |
| 34–41   | View 1 | Padding zeros                                |
| 42–77   | View 2 | Welch PSD band powers (3 bands × 12 leads)   |
| 78      | View 2 | Spectral entropy (mean over leads)           |
| 79      | View 2 | Spectral roll-off (mean over leads)          |
| 80–83   | View 2 | db4 wavelet detail energies L1–L4 (Lead II)  |

---

## Troubleshooting

| Problem                      | Fix                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: wfdb`  | `pip install wfdb`                                                                                       |
| `ModuleNotFoundError: pywt`  | `pip install PyWavelets`                                                                                 |
| Memory error on full dataset | Use `--max_records 5000`                                                                                 |
| `PTBXL_PATH` not set         | Pass `--data_path D:\ml-project\dataset` explicitly                                                      |
| NaN in features              | The extractor handles zero-std leads gracefully; run with `--skip_extraction` to reload the fixed output |
