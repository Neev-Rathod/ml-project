"""
wp1_data_pipeline/loader.py
============================
Loads PTB-XL raw waveforms and metadata.
Provides the low-level building block used by feature_extractor.py.
"""

import ast
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from tqdm import tqdm

from shared.data_utils import SUPERCLASSES, CLASS_TO_IDX


def load_metadata(data_path: str | Path) -> pd.DataFrame:
    """
    Read ptbxl_database.csv and scp_statements.csv, then attach a
    'diagnostic_superclass' column containing a single string from
    {NORM, MI, STTC, CD, HYP}, or None if ambiguous / multi-class.

    Parameters
    ----------
    data_path : path to the PTB-XL root directory

    Returns
    -------
    meta : pd.DataFrame indexed by ecg_id
    """
    data_path = Path(data_path)

    meta = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(data_path / "scp_statements.csv", index_col=0)
    diagnostic_scp = scp[scp["diagnostic"] == 1]

    def _aggregate(scp_dict: dict) -> str | None:
        classes = set()
        for code in scp_dict:
            if code in diagnostic_scp.index:
                cls = diagnostic_scp.loc[code, "diagnostic_class"]
                if cls in CLASS_TO_IDX:
                    classes.add(cls)
        if len(classes) == 1:
            return classes.pop()
        return None  # ambiguous or no diagnostic code

    meta["diagnostic_superclass"] = meta["scp_codes"].apply(_aggregate)
    return meta


def load_waveforms(
    meta: pd.DataFrame,
    data_path: str | Path,
    sampling_rate: int = 100,
    indices: list[int] | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load raw waveform arrays for specified records.

    Parameters
    ----------
    meta          : DataFrame from load_metadata()
    data_path     : PTB-XL root directory
    sampling_rate : 100 or 500
    indices       : list of integer positional indices into meta to load;
                    loads all records if None
    verbose       : show tqdm progress bar

    Returns
    -------
    X : ndarray of shape (n_records, 1000, 12)   [at 100 Hz]
          or              (n_records, 5000, 12)   [at 500 Hz]
    """
    data_path = Path(data_path)
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    if indices is not None:
        rows = meta.iloc[indices]
    else:
        rows = meta

    signals = []
    iterator = tqdm(rows[col], desc="Loading waveforms") if verbose else rows[col]
    for fname in iterator:
        record = wfdb.rdsamp(str(data_path / fname))
        signals.append(record[0])   # shape (T, 12)

    return np.array(signals, dtype=np.float32)
