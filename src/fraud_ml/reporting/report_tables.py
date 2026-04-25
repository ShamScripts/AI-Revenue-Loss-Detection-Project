"""Build IEEE / Elliptic / ablation metric tables for reports (CSV + Streamlit).

IEEE Logistic Regression / Random Forest rows use the **same fused cohort** as Stage 4
(``record_id`` in ``final_hybrid_scores.csv``), with train split = cohort minus fusion test IDs.
Training rows are capped at 80k (stratified) and use lighter sklearn settings than Stage 2 so
Stage 4 stays tractable on full IEEE; replace those cells from your paper if you need exact
Stage-2 hyperparameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_ml.config.paths import get_paths

RANDOM_STATE = 42
TARGET_IEEE = "isFraud"

W_GBDT_DEFAULT = 0.45
W_DNN_DEFAULT = 0.40
W_ANOM_DEFAULT = 0.15


def _metrics_row(y_true, y_prob, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
    }


def _pick_id_col(df: pd.DataFrame) -> str | None:
    for c in ("TransactionID", "transaction_id", "txId", "tx_id"):
        if c in df.columns:
            return c
    return None


def _table_row(model: str, m: dict[str, float] | None) -> dict:
    if m is None:
        return {
            "Model": model,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1-Score": np.nan,
            "AUC": np.nan,
        }
    return {
        "Model": model,
        "Precision": m["Precision"],
        "Recall": m["Recall"],
        "F1-Score": m["F1-Score"],
        "AUC": m["AUC"],
    }


def _train_lr_rf_ieee_test(
    ieee_path: Path,
    cohort_ids: set,
    test_ids: set,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Fit LR/RF on IEEE cohort rows not in test_ids; evaluate on fusion test_ids."""
    if not ieee_path.is_file():
        return None, None
    df = pd.read_csv(ieee_path)
    id_col = _pick_id_col(df)
    if id_col is None or TARGET_IEEE not in df.columns:
        return None, None

    cohort_ids = {x for x in cohort_ids}
    test_ids = {x for x in test_ids}
    train_ids = cohort_ids - test_ids

    df = df[df[id_col].isin(cohort_ids)].copy()
    if len(df) < 20:
        return None, None

    feat_cols = [c for c in df.columns if c not in (id_col, TARGET_IEEE)]
    X = df[feat_cols].select_dtypes(include=[np.number]).copy()
    y = df[TARGET_IEEE].astype(int)
    tid = df[id_col].values

    train_mask = np.array([t in train_ids for t in tid])
    test_mask = np.array([t in test_ids for t in tid])
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        return None, None

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    _max_tr = 80_000
    if len(X_train) > _max_tr:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=_max_tr, stratify=y_train, random_state=RANDOM_STATE
        )

    # Lighter than Stage 2 so Stage 4 + report tables finish in reasonable time on full IEEE.
    lr = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    random_state=RANDOM_STATE,
                    solver="saga",
                ),
            ),
        ]
    )
    lr.fit(X_train, y_train)
    lr_p = lr.predict_proba(X_test)[:, 1]
    lr_m = _metrics_row(y_test, lr_p, 0.5)

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=12,
                    min_samples_leaf=20,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_p = rf.predict_proba(X_test)[:, 1]
    rf_m = _metrics_row(y_test, rf_p, 0.5)

    return lr_m, rf_m


def _row_from_stage5(r: pd.Series, display_name: str) -> dict:
    def g(key: str) -> float:
        if key not in r.index:
            return float("nan")
        v = r[key]
        if pd.isna(v):
            return float("nan")
        return float(v)

    return {
        "Model": display_name,
        "Precision": g("precision"),
        "Recall": g("recall"),
        "F1-Score": g("f1"),
        "AUC": g("roc_auc"),
    }


def _elliptic_report_table(out_dir: Path, elliptic_path: Path) -> pd.DataFrame:
    """Prefer Stage 5 ``elliptic_graph_experiments.csv`` (temporal split + GNN); else LR/RF only."""
    exp_path = out_dir / "elliptic_graph_experiments.csv"
    if exp_path.is_file():
        exp = pd.read_csv(exp_path).drop_duplicates(subset=["model"]).set_index("model")
        name_map = [
            ("Logistic Regression (tabular)", "Logistic Regression"),
            ("Random Forest (tabular)", "Random Forest"),
            ("GNN (2-layer GCN, Elliptic graph)", "GNN [8]"),
            ("FraudGT-style MLP (tabular encoder)", "FraudGT [9]"),
        ]
        rows_out: list[dict] = []
        for src, disp in name_map:
            if src in exp.index:
                rows_out.append(_row_from_stage5(exp.loc[src], disp))
            else:
                rows_out.append(_table_row(disp, None))
        rows_out.append(_table_row("Proposed Hybrid Model", None))
        return pd.DataFrame(rows_out)

    rows = [
        _table_row("Logistic Regression", None),
        _table_row("Random Forest", None),
        _table_row("GNN [8]", None),
        _table_row("FraudGT [9]", None),
        _table_row("Proposed Hybrid Model", None),
    ]
    if not elliptic_path.is_file():
        return pd.DataFrame(rows)

    df = pd.read_csv(elliptic_path)
    if "class_label" not in df.columns:
        return pd.DataFrame(rows)

    sub = df[df["class_label"].isin(["licit", "illicit"])].copy()
    if len(sub) < 50:
        return pd.DataFrame(rows)

    sub["target"] = (sub["class_label"] == "illicit").astype(int)
    feat_cols = [c for c in sub.columns if c.startswith("feature_")]
    feat_cols += [c for c in ("time_step", "in_degree", "out_degree", "total_degree", "degree_in_out_ratio") if c in sub.columns]
    feat_cols = [c for c in feat_cols if c in sub.columns]
    if not feat_cols:
        return pd.DataFrame(rows)

    X = sub[feat_cols].select_dtypes(include=[np.number]).copy()
    y = sub["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    lr = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    lr.fit(X_train, y_train)
    lr_m = _metrics_row(y_test, lr.predict_proba(X_test)[:, 1], 0.5)

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=10,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_m = _metrics_row(y_test, rf.predict_proba(X_test)[:, 1], 0.5)

    return pd.DataFrame(
        [
            _table_row("Logistic Regression", lr_m),
            _table_row("Random Forest", rf_m),
            _table_row("GNN [8]", None),
            _table_row("FraudGT [9]", None),
            _table_row("Proposed Hybrid Model", None),
        ]
    )


def build_and_save_report_tables(
    score_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_threshold: float,
    project_root: Path | None = None,
    w_gbdt: float = W_GBDT_DEFAULT,
    w_dnn: float = W_DNN_DEFAULT,
    w_anom: float = W_ANOM_DEFAULT,
) -> dict[str, Path]:
    """
    Write CSVs under processed_data/:
    - report_table_1_ieee_cis.csv
    - report_table_2_elliptic.csv
    - report_table_3_ablation.csv
    """
    paths = get_paths(project_root)
    out_dir = paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ieee_path = out_dir / "ieee_train_eda_ready.csv"
    elliptic_path = out_dir / "elliptic_transactions_cleaned.csv"

    cohort_ids = set(score_df["record_id"].values)
    test_ids = set(test_df["record_id"].values)
    lr_m, rf_m = _train_lr_rf_ieee_test(ieee_path, cohort_ids, test_ids)

    y_true = test_df["target"].astype(int).values

    gbdt_m = _metrics_row(y_true, test_df["gbdt_score"].values, 0.5)
    dnn_m = _metrics_row(y_true, test_df["dnn_score"].values, 0.5)
    anom_m = _metrics_row(y_true, test_df["anomaly_score"].values, 0.5)
    hybrid_m = _metrics_row(y_true, test_df["hybrid_weighted_score"].values, best_threshold)

    table1 = pd.DataFrame(
        [
            _table_row("Logistic Regression", lr_m),
            _table_row("Random Forest", rf_m),
            _table_row("XGBoost (GBDT) [3]", gbdt_m),
            _table_row("Deep Neural Network [7]", dnn_m),
            _table_row("Isolation Forest [7]", anom_m),
            _table_row("Proposed Hybrid Model", hybrid_m),
        ]
    )

    table2 = _elliptic_report_table(out_dir, elliptic_path)

    # Ablation: normalized component scores on same test split as fusion
    g = test_df["gbdt_score"].values
    d = test_df["dnn_score"].values
    a = test_df["anomaly_score"].values
    s_gd = (w_gbdt * g + w_dnn * d) / (w_gbdt + w_dnn)
    s_ga = (w_gbdt * g + w_anom * a) / (w_gbdt + w_anom)

    ab_m_gbdt = _metrics_row(y_true, g, 0.5)
    ab_m_dl = _metrics_row(y_true, d, 0.5)
    ab_m_ad = _metrics_row(y_true, a, 0.5)
    ab_m_gd = _metrics_row(y_true, s_gd, 0.5)
    ab_m_ga = _metrics_row(y_true, s_ga, 0.5)
    ab_m_full = _metrics_row(y_true, test_df["hybrid_weighted_score"].values, best_threshold)

    table3 = pd.DataFrame(
        [
            _table_row("GBDT Only [3]", ab_m_gbdt),
            _table_row("Deep Learning Only [7]", ab_m_dl),
            _table_row("Anomaly Detection Only [7]", ab_m_ad),
            _table_row("GBDT + Deep Learning", ab_m_gd),
            _table_row("GBDT + Anomaly Detection", ab_m_ga),
            _table_row("Full Hybrid Model", ab_m_full),
        ]
    ).rename(columns={"Model": "Model Variant"})

    p1 = out_dir / "report_table_1_ieee_cis.csv"
    p2 = out_dir / "report_table_2_elliptic.csv"
    p3 = out_dir / "report_table_3_ablation.csv"

    def _round_metrics(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        name_cols = {"Model", "Model Variant"}
        for c in out.columns:
            if c not in name_cols and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].round(4)
        return out

    for p, df in ((p1, _round_metrics(table1)), (p2, _round_metrics(table2)), (p3, _round_metrics(table3))):
        df.to_csv(p, index=False, na_rep="")

    # Optional: test IDs for reproducibility
    ids_path = out_dir / "fusion_test_record_ids.csv"
    test_df[["record_id"]].to_csv(ids_path, index=False)

    print("Saved report tables:", p1.name, p2.name, p3.name)
    return {"ieee": p1, "elliptic": p2, "ablation": p3, "test_ids": ids_path}
