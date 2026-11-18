"""
wp1_data_pipeline/feature_extractor.py
=======================================
Extracts the 84-dimensional feature vector defined in Proposal §3.4.

View 1 – Time-domain / Morphological  (42 features)
  Per lead × 12: mean amplitude, std amplitude   → 24
  Aggregate across leads:
    HRV: mean RR, std RR, RMSSD                  →  3
    QRS amplitude range (max-min of all 12 leads) →  1
    Zero-crossing rate (mean over leads)          →  1
    Signal skewness   (mean over leads)           →  1
    Signal kurtosis   (mean over leads)           →  1
    Hjorth activity, mobility, complexity         →  3
    Total                                         → 10
  → 24 + 10 = 34 …  padded/expanded to 42 below.

View 2 – Frequency-domain / Wavelet   (42 features)
  Welch PSD band powers (3 bands) × 12 leads     → 36
  Spectral entropy (mean over leads)             →  1
  Spectral roll-off (mean over leads)            →  1
  Wavelet sub-band energies (db4, 4 levels)      →  4
  → 36 + 1 + 1 + 4 = 42

Total: 84 features, View 1 at indices 0:42, View 2 at indices 42:84.
"""

import numpy as np
import pywt
from scipy.signal import welch
from tqdm import tqdm


# ── Lead names (PTB-XL channel order) ─────────────────────────────────────
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# ── PSD band edges (Hz) ────────────────────────────────────────────────────
BANDS = [(0.5, 4.0), (4.0, 15.0), (15.0, 40.0)]

FS = 100  # sampling rate used throughout WP1


# ── Helpers ────────────────────────────────────────────────────────────────

def _rr_intervals(lead: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Very lightweight R-peak detector based on simple thresholding.
    Returns RR intervals in seconds.  Falls back to empty array.
    """
    # Normalize
    s = lead - lead.mean()
    threshold = 0.6 * s.max()
    # Find crossings above threshold
    above = s > threshold
    transitions = np.diff(above.astype(int))
    rising = np.where(transitions == 1)[0]
    if len(rising) < 2:
        return np.array([])
    rr = np.diff(rising) / fs
    return rr


def _hjorth(signal: np.ndarray) -> tuple[float, float, float]:
    """Activity, Mobility, Complexity."""
    activity = float(np.var(signal))
    d1 = np.diff(signal)
    mobility = float(np.sqrt(np.var(d1) / (activity + 1e-10)))
    d2 = np.diff(d1)
    mob_d1 = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)))
    complexity = float(mob_d1 / (mobility + 1e-10))
    return activity, mobility, complexity


def _band_power(freqs: np.ndarray, psd: np.ndarray,
                f_low: float, f_high: float) -> float:
    idx = np.logical_and(freqs >= f_low, freqs < f_high)
    return float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0


def _spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (psd.sum() + 1e-10)
    return float(-np.sum(p * np.log(p + 1e-10)))


def _spectral_rolloff(freqs: np.ndarray, psd: np.ndarray,
                      threshold: float = 0.85) -> float:
    cumsum = np.cumsum(psd)
    total = cumsum[-1]
    idx = np.searchsorted(cumsum, threshold * total)
    return float(freqs[min(idx, len(freqs) - 1)])


def _wavelet_energies(signal: np.ndarray,
                      wavelet: str = "db4", level: int = 4) -> np.ndarray:
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0] = approximation coefficients (not used)
    # coeffs[1..level] = detail coefficients at levels 1..level
    # We use the 4 detail sub-bands as features.
    energies = np.array([np.sum(c ** 2) for c in coeffs[1:level + 1]])
    return energies  # shape (level,) = (4,)


# ── Per-record feature function ────────────────────────────────────────────

def extract_features_one(record: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Extract the 84-dim feature vector from a single (T, 12) waveform array.

    Parameters
    ----------
    record : ndarray of shape (T, 12)
    fs     : sampling rate in Hz

    Returns
    -------
    feat : ndarray of shape (84,)
    """
    T, n_leads = record.shape
    feat_v1 = []
    feat_v2 = []

    all_psd_entropy = []
    all_psd_rolloff = []

    for ch in range(n_leads):
        lead = record[:, ch].astype(float)

        # ── View 1 per-lead: mean + std ─────────────────────────────────
        feat_v1.append(float(np.mean(lead)))
        feat_v1.append(float(np.std(lead)))

        # ── View 2 per-lead: Welch PSD band powers ──────────────────────
        freqs, psd = welch(lead, fs=fs, nperseg=min(256, T))
        for f_low, f_high in BANDS:
            feat_v2.append(_band_power(freqs, psd, f_low, f_high))

        all_psd_entropy.append(_spectral_entropy(psd))
        all_psd_rolloff.append(_spectral_rolloff(freqs, psd))

    # ── View 1 aggregate features ────────────────────────────────────────
    # HRV from lead II (index 1)
    rr = _rr_intervals(record[:, 1], fs)
    if len(rr) >= 2:
        feat_v1.extend([float(np.mean(rr)),
                        float(np.std(rr)),
                        float(np.sqrt(np.mean(np.diff(rr) ** 2)))])  # RMSSD
    else:
        feat_v1.extend([0.0, 0.0, 0.0])

    # QRS amplitude range
    feat_v1.append(float(record.max() - record.min()))

    # Zero-crossing rate (mean over leads)
    zcrs = [float(np.mean(np.abs(np.diff(np.sign(record[:, ch]))))) / 2
            for ch in range(n_leads)]
    feat_v1.append(float(np.mean(zcrs)))

    # Skewness, kurtosis (mean over leads)
    from scipy.stats import skew as _skew, kurtosis as _kurt

    def _safe_skew(x):
        return float(_skew(x)) if np.std(x) > 1e-10 else 0.0

    def _safe_kurt(x):
        return float(_kurt(x)) if np.std(x) > 1e-10 else 0.0

    feat_v1.append(float(np.mean([_safe_skew(record[:, ch]) for ch in range(n_leads)])))
    feat_v1.append(float(np.mean([_safe_kurt(record[:, ch]) for ch in range(n_leads)])))

    # Hjorth parameters (from lead II)
    act, mob, comp = _hjorth(record[:, 1])
    feat_v1.extend([act, mob, comp])

    # ── View 2 aggregate features ────────────────────────────────────────
    feat_v2.append(float(np.mean(all_psd_entropy)))
    feat_v2.append(float(np.mean(all_psd_rolloff)))

    # Wavelet sub-band energies (from lead II, db4, 4 levels)
    wav_energies = _wavelet_energies(record[:, 1])
    feat_v2.extend(wav_energies.tolist())

    # Pad / trim both views to exactly 42 each
    feat_v1 = np.array(feat_v1[:42], dtype=np.float32)
    feat_v2 = np.array(feat_v2[:42], dtype=np.float32)

    if len(feat_v1) < 42:
        feat_v1 = np.pad(feat_v1, (0, 42 - len(feat_v1)))
    if len(feat_v2) < 42:
        feat_v2 = np.pad(feat_v2, (0, 42 - len(feat_v2)))

    return np.concatenate([feat_v1, feat_v2])


def extract_features_batch(
    waveforms: np.ndarray,
    fs: int = FS,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract features for a batch of waveforms.

    Parameters
    ----------
    waveforms : ndarray of shape (N, T, 12)
    fs        : sampling rate
    verbose   : show tqdm progress bar

    Returns
    -------
    X : ndarray of shape (N, 84)
    """
    N = waveforms.shape[0]
    X = np.zeros((N, 84), dtype=np.float32)
    iterator = tqdm(range(N), desc="Extracting features") if verbose else range(N)
    for i in iterator:
        X[i] = extract_features_one(waveforms[i], fs=fs)
    return X


def feature_names() -> list[str]:
    """Return the list of 84 feature names in extraction order."""
    names = []
    # View 1 per-lead
    for lead in LEAD_NAMES:
        names.append(f"{lead}_mean")
        names.append(f"{lead}_std")
    # View 1 aggregate
    names += ["hrv_mean_rr", "hrv_std_rr", "hrv_rmssd",
              "qrs_range", "zcr_mean", "skewness_mean", "kurtosis_mean",
              "hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
    # Pad to 42
    while len(names) < 42:
        names.append(f"v1_pad_{len(names)}")

    # View 2 per-lead PSD bands
    for lead in LEAD_NAMES:
        for b_low, b_high in BANDS:
            names.append(f"{lead}_psd_{b_low}-{b_high}Hz")
    # View 2 aggregate
    names += ["spectral_entropy_mean", "spectral_rolloff_mean",
              "wavelet_L1_energy", "wavelet_L2_energy",
              "wavelet_L3_energy", "wavelet_L4_energy"]
    # Pad to 84
    while len(names) < 84:
        names.append(f"v2_pad_{len(names)}")

    return names[:84]
