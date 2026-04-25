"""Train/validation/test splitting: random stratified vs time-ordered (IEEE / fusion)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

SplitMode = Literal["random", "temporal"]


def ieee_train_valid_arrays(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split_mode: SplitMode = "random",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Return (X_train, X_valid, y_train, y_valid, ids_train, ids_valid).
    y_train / y_valid are numpy int arrays for sklearn compatibility.
    """
    id_col = "TransactionID" if "TransactionID" in df.columns else None

    if split_mode == "temporal" and "TransactionDT" in df.columns:
        order = np.argsort(df["TransactionDT"].values)
        X_o = X.iloc[order].reset_index(drop=True)
        y_o = y.iloc[order].reset_index(drop=True)
        if id_col:
            ids_o = df[id_col].iloc[order].reset_index(drop=True)
        else:
            ids_o = pd.Series(np.arange(len(X_o)))
        cut = int(len(X_o) * (1.0 - test_size))
        if 2 <= cut < len(X_o) - 1:
            X_train, X_valid = X_o.iloc[:cut], X_o.iloc[cut:]
            y_train = y_o.iloc[:cut].values
            y_valid = y_o.iloc[cut:].values
            ids_train, ids_valid = ids_o.iloc[:cut], ids_o.iloc[cut:]
            return X_train, X_valid, y_train, y_valid, ids_train, ids_valid

    strat = y if len(np.unique(y)) > 1 else None
    if id_col is not None:
        ids_all = df[id_col]
        X_train, X_valid, y_train, y_valid, ids_train, ids_valid = train_test_split(
            X, y, ids_all, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
        )
        return X_train, X_valid, y_train.values, y_valid.values, ids_train, ids_valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
    )
    ids_train = pd.Series(X_train.index.values)
    ids_valid = pd.Series(X_valid.index.values)
    return X_train, X_valid, y_train.values, y_valid.values, ids_train, ids_valid


def fusion_temporal_train_val_test(
    score_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_frac_of_train: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split: sort by ``transaction_dt``, ``TransactionDT``, or ``time_step`` if present,
    else by ``record_id``. Last ``test_size`` fraction → test; remaining train split into dev/val.
    """
    df = score_df.copy()
    time_col = None
    for c in ("transaction_dt", "TransactionDT", "time_step"):
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        df = df.sort_values("record_id").reset_index(drop=True)
    else:
        df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    cut = int(n * (1.0 - test_size))
    train_pool = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()
    if len(train_pool) < 10 or len(test_df) < 5:
        raise ValueError("Temporal split produced too few rows; use random split.")

    strat = train_pool["target"] if "target" in train_pool.columns else None
    dev_df, val_df = train_test_split(
        train_pool,
        test_size=val_frac_of_train,
        random_state=random_state,
        stratify=strat,
    )
    return dev_df, val_df, test_df


def fusion_random_train_val_test(
    score_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_frac_of_train: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        score_df,
        test_size=test_size,
        random_state=random_state,
        stratify=score_df["target"],
    )
    strat = train_df["target"]
    dev_df, val_df = train_test_split(
        train_df,
        test_size=val_frac_of_train,
        random_state=random_state,
        stratify=strat,
    )
    return dev_df, val_df, test_df
