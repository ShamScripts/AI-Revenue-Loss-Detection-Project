"""Stage 1: load IEEE-CIS + Elliptic, preprocess, EDA plots, write processed_data/."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

from fraud_ml.config.paths import get_paths

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

MISSING_DROP_THRESHOLD = 0.90


def log_shape(step: str, df: pd.DataFrame) -> None:
    print(f"[{step}] shape = {df.shape[0]:,} rows × {df.shape[1]:,} cols")


def _save_fig(name: str, save_plots: bool, figures_dir: Path) -> None:
    if save_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close()


def run(
    project_root: Path | None = None,
    save_plots: bool = True,
) -> None:
    paths = get_paths(project_root)
    PROJECT_ROOT = paths.root
    DATA_ROOT = paths.data_root
    PROCESSED_DIR = paths.processed_dir
    figures_dir = paths.figures

    IEEE_DIR = paths.ieee_dir
    ieee_train_transaction_path = IEEE_DIR / "train_transaction.csv"
    ieee_train_identity_path = IEEE_DIR / "train_identity.csv"
    ELLIPTIC_DIR = paths.elliptic_dir
    elliptic_features_path = ELLIPTIC_DIR / "elliptic_txs_features.csv"
    elliptic_classes_path = ELLIPTIC_DIR / "elliptic_txs_classes.csv"
    elliptic_edges_path = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for p in [ieee_train_transaction_path, elliptic_features_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
    print("DATA_ROOT:", DATA_ROOT.resolve())
    print("PROCESSED_DIR:", PROCESSED_DIR.resolve())
    print("MISSING_DROP_THRESHOLD:", MISSING_DROP_THRESHOLD)

    train_transaction = pd.read_csv(ieee_train_transaction_path)
    train_identity = pd.read_csv(ieee_train_identity_path)
    log_shape("After load: train_transaction", train_transaction)
    log_shape("After load: train_identity", train_identity)
    print(train_transaction.head(3))
    print("TransactionID in identity:", "TransactionID" in train_identity.columns)
    print(train_identity.head(3))

    shape_before = train_transaction.shape
    ieee_train = train_transaction.merge(
        train_identity, on="TransactionID", how="left", suffixes=("", "_id")
    )
    log_shape("After merge (before cleaning)", ieee_train)
    print(f"  train_transaction was: {shape_before}")
    n_match = ieee_train["TransactionID"].isin(train_identity["TransactionID"]).sum()
    print(f"Transactions with at least one identity row: {n_match:,} / {len(ieee_train):,}")
    print(
        "Duplicate TransactionIDs in train_transaction:",
        train_transaction["TransactionID"].duplicated().sum(),
    )

    if "isFraud" in ieee_train.columns:
        print(ieee_train["isFraud"].value_counts(dropna=False))
        print("Fraud rate (%):", ieee_train["isFraud"].mean() * 100)

    miss_pct = (ieee_train.isna().sum() / len(ieee_train) * 100).sort_values(ascending=False)
    print("Top 15 columns by % missing:")
    print(miss_pct.head(15))
    print("Duplicate rows:", ieee_train.duplicated().sum())
    num_cols_preview = ieee_train.select_dtypes(include=[np.number]).columns[:15]
    print(ieee_train[num_cols_preview].describe().T.head(12))

    miss_pct_all = (ieee_train.isna().sum() / len(ieee_train) * 100).sort_values(ascending=False)
    top20 = miss_pct_all.head(20).reset_index()
    top20.columns = ["column", "pct_missing"]

    def planned_action(row):
        if row["pct_missing"] > MISSING_DROP_THRESHOLD * 100:
            return "drop (>{}% missing)".format(int(MISSING_DROP_THRESHOLD * 100))
        col = row["column"]
        if col in ("TransactionID", "isFraud"):
            return "keep (id/target)"
        if ieee_train[col].dtype in (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8):
            return "impute (median)"
        return "impute (Unknown)"

    top20["planned_action"] = top20.apply(planned_action, axis=1)
    print(top20)
    ieee_missing_summary_csv = PROCESSED_DIR / "ieee_missing_top20_summary.csv"
    top20.to_csv(ieee_missing_summary_csv, index=False)
    print("Saved:", ieee_missing_summary_csv)

    miss_pct_full = ieee_train.isna().mean()
    drop_cols = miss_pct_full[miss_pct_full > MISSING_DROP_THRESHOLD].index.tolist()
    drop_cols = [c for c in drop_cols if c not in ("TransactionID", "isFraud")]
    print(f"Dropping {len(drop_cols)} columns with >{MISSING_DROP_THRESHOLD * 100:.0f}% missing")

    shape_before_drop = ieee_train.shape
    ieee_clean = ieee_train.drop(columns=drop_cols, errors="ignore")
    log_shape("After dropping sparse columns", ieee_clean)
    print(f"  before: {shape_before_drop}, removed {shape_before_drop[1] - ieee_clean.shape[1]} columns")

    id_col = "TransactionID"
    target_col = "isFraud"
    feature_df = ieee_clean.drop(columns=[target_col], errors="ignore")
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != id_col]
    cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != id_col]

    shape_before_imp = ieee_clean.shape
    num_imputer = SimpleImputer(strategy="median")
    ieee_clean[num_cols] = num_imputer.fit_transform(ieee_clean[num_cols])
    for c in cat_cols:
        ieee_clean[c] = ieee_clean[c].astype("object").fillna("Unknown")

    log_shape("After imputation", ieee_clean)
    print(
        "Remaining NA in feature columns:",
        ieee_clean.drop(columns=[target_col], errors="ignore").isna().sum().sum(),
    )
    print(f"  shape unchanged rows/cols vs pre-impute: {shape_before_imp} -> {ieee_clean.shape}")

    print("Duplicate rows after cleaning:", ieee_clean.duplicated().sum())
    for c in ieee_clean.select_dtypes(include=["float64"]).columns:
        ieee_clean[c] = pd.to_numeric(ieee_clean[c], downcast="float")
    for c in cat_cols:
        ieee_clean[c] = ieee_clean[c].astype("category")

    log_shape("After dtype optimization", ieee_clean)

    num_features = [
        c
        for c in ieee_clean.select_dtypes(include=[np.number]).columns
        if c not in (id_col, target_col)
    ]
    cat_features = [
        c
        for c in ieee_clean.select_dtypes(include=["category", "object"]).columns
        if c != id_col
    ]
    print("Numerical (before new features):", len(num_features))
    print("Categorical:", len(cat_features))

    SEC_PER_DAY = 86400
    ieee_clean["rel_day"] = (ieee_clean["TransactionDT"] // SEC_PER_DAY).astype(int)
    ieee_clean["rel_hour"] = ((ieee_clean["TransactionDT"] % SEC_PER_DAY) // 3600).astype(int)
    ieee_clean["is_weekend_like"] = (ieee_clean["rel_day"] % 7).isin([5, 6]).astype(int)
    ieee_clean["amt_log"] = np.log1p(ieee_clean["TransactionAmt"].astype(float))
    mean_day = ieee_clean.groupby("rel_day", observed=True)["TransactionAmt"].transform("mean")
    ieee_clean["amt_to_rel_day_mean_ratio"] = ieee_clean["TransactionAmt"] / mean_day.replace(0, np.nan)
    ieee_clean["amt_to_rel_day_mean_ratio"] = ieee_clean["amt_to_rel_day_mean_ratio"].fillna(1.0)
    if "card1" in ieee_clean.columns:
        mean_card = ieee_clean.groupby("card1", observed=True)["TransactionAmt"].transform("mean")
        ieee_clean["amt_to_card1_mean_ratio"] = ieee_clean["TransactionAmt"] / mean_card.replace(0, np.nan)
        ieee_clean["amt_to_card1_mean_ratio"] = ieee_clean["amt_to_card1_mean_ratio"].fillna(1.0)

    log_shape("After time + amount feature engineering", ieee_clean)

    fig, ax = plt.subplots()
    sns.countplot(x="isFraud", data=ieee_clean, ax=ax)
    ax.set_title("isFraud distribution")
    _save_fig("stage01_isFraud_distribution", save_plots, figures_dir)
    print("Fraud %:", ieee_clean["isFraud"].mean() * 100)

    cols_plot = [c for c in ["TransactionAmt", "amt_log", "dist1", "dist2"] if c in ieee_clean.columns]
    for i, c in enumerate(cols_plot):
        plt.figure()
        sns.histplot(ieee_clean[c], bins=50, kde=True)
        plt.title(f"Distribution: {c}")
        _save_fig(f"stage01_hist_{c}", save_plots, figures_dir)

    plt.figure()
    sns.boxplot(x="isFraud", y="TransactionAmt", data=ieee_clean, showfliers=False)
    plt.title("TransactionAmt by isFraud (outliers hidden)")
    _save_fig("stage01_boxplot_amt_by_fraud", save_plots, figures_dir)

    top_miss = (
        (ieee_train.isna().sum() / len(ieee_train) * 100)).sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 5))
    top_miss.sort_values().plot(kind="barh")
    plt.xlabel("% missing (original merged, before drop/impute)")
    plt.title("Top 20 columns by missing %")
    plt.tight_layout()
    _save_fig("stage01_top20_missing", save_plots, figures_dir)

    for col in [c for c in ["ProductCD", "card4", "card6", "DeviceType"] if c in ieee_clean.columns]:
        t = ieee_clean.groupby(col, observed=True)["isFraud"].agg(["mean", "count"])
        t = t.sort_values("mean", ascending=False).head(10)
        print(col)
        print(t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ieee_clean.groupby("rel_hour").size().plot(ax=axes[0], title="Transactions by rel_hour")
    ieee_clean.groupby("rel_hour")["isFraud"].mean().plot(ax=axes[1], title="Fraud rate by rel_hour")
    plt.tight_layout()
    _save_fig("stage01_rel_hour", save_plots, figures_dir)

    sub = ["TransactionAmt", "amt_log", "TransactionDT", "rel_hour", "rel_day", "amt_to_rel_day_mean_ratio"]
    if "amt_to_card1_mean_ratio" in ieee_clean.columns:
        sub.append("amt_to_card1_mean_ratio")
    sub = [c for c in sub if c in ieee_clean.columns]
    v_cols = [c for c in ieee_clean.columns if c.startswith("V")][:5]
    sub = sub + v_cols
    sub = [
        c
        for c in sub
        if c in ieee_clean.columns
        and str(ieee_clean[c].dtype) not in ("category", "object")
    ]
    cm = ieee_clean[sub + ["isFraud"]].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="vlag", center=0)
    plt.title("Correlation — selected numeric features + isFraud")
    plt.tight_layout()
    _save_fig("stage01_corr_heatmap", save_plots, figures_dir)

    num_for_corr = ieee_clean.select_dtypes(include=[np.number]).columns.tolist()
    num_for_corr = [c for c in num_for_corr if c not in (id_col, target_col)]
    cor_series = ieee_clean[num_for_corr].corrwith(ieee_clean[target_col], numeric_only=True).dropna()
    cor_series = cor_series.reindex(cor_series.abs().sort_values(ascending=False).index)
    top_n = min(25, len(cor_series))
    print("Top features by absolute Pearson correlation with isFraud (numeric columns):")
    print(cor_series.head(top_n).to_frame("corr_with_isFraud"))

    ieee_out_full = PROCESSED_DIR / "ieee_train_merged_cleaned.csv"
    ieee_out_eda = PROCESSED_DIR / "ieee_train_eda_ready.csv"
    ieee_clean.to_csv(ieee_out_full, index=False)
    ieee_clean.to_csv(ieee_out_eda, index=False)
    log_shape("Saved IEEE files (in-memory shape)", ieee_clean)
    print("Saved:", ieee_out_full)
    print("Saved:", ieee_out_eda)

    feat_cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 166)]
    elliptic_features = pd.read_csv(elliptic_features_path, header=None, names=feat_cols)
    elliptic_classes = pd.read_csv(elliptic_classes_path)
    elliptic_edges = pd.read_csv(elliptic_edges_path)
    log_shape("Elliptic features", elliptic_features)
    log_shape("Elliptic classes", elliptic_classes)
    log_shape("Elliptic edges", elliptic_edges)
    print(elliptic_features.head(2))
    print(elliptic_classes.head(10))
    print(elliptic_edges.head())

    print("Unique raw class values (sample):", elliptic_classes["class"].unique()[:20])
    print(elliptic_classes["class"].value_counts())
    elliptic_classes["txId"] = elliptic_classes["txId"].astype(np.int64)
    elliptic_features["txId"] = elliptic_features["txId"].astype(np.int64)

    elliptic_tx = elliptic_features.merge(elliptic_classes, on="txId", how="left")
    log_shape("After merge features + classes", elliptic_tx)
    print("Missing class after merge:", elliptic_tx["class"].isna().sum())
    print(elliptic_tx["class"].value_counts(dropna=False))

    def normalize_elliptic_class(x):
        if pd.isna(x):
            return "unknown"
        s = str(x).strip().lower()
        if s in ("1", "licit"):
            return "licit"
        if s in ("2", "illicit"):
            return "illicit"
        if s in ("unknown", "3"):
            return "unknown"
        return "unknown"

    elliptic_tx["class_label"] = elliptic_tx["class"].map(normalize_elliptic_class)
    print(elliptic_tx["class_label"].value_counts())

    feat_only = [c for c in elliptic_tx.columns if c.startswith("feature_")]
    print("NA in engineered feature columns:", elliptic_tx[feat_only].isna().sum().sum())

    out_deg = elliptic_edges.groupby("txId1").size().rename("out_degree")
    in_deg = elliptic_edges.groupby("txId2").size().rename("in_degree")
    all_nodes = pd.Index(elliptic_tx["txId"].unique())
    deg_df = pd.DataFrame({"txId": all_nodes})
    deg_df = deg_df.merge(out_deg.reset_index().rename(columns={"txId1": "txId"}), on="txId", how="left")
    deg_df = deg_df.merge(in_deg.reset_index().rename(columns={"txId2": "txId"}), on="txId", how="left")
    deg_df["out_degree"] = deg_df["out_degree"].fillna(0).astype(int)
    deg_df["in_degree"] = deg_df["in_degree"].fillna(0).astype(int)
    deg_df["total_degree"] = deg_df["out_degree"] + deg_df["in_degree"]
    deg_df["is_isolated"] = (deg_df["in_degree"] == 0) & (deg_df["out_degree"] == 0)
    deg_df["degree_in_out_ratio"] = deg_df["in_degree"] / (deg_df["out_degree"] + 1)
    thr = deg_df["total_degree"].quantile(0.90)
    deg_df["high_degree"] = (deg_df["total_degree"] >= thr).astype(int)
    elliptic_tx = elliptic_tx.merge(deg_df, on="txId", how="left")
    print(
        "Edges:",
        len(elliptic_edges),
        "| Unique sources:",
        elliptic_edges["txId1"].nunique(),
        "| Unique targets:",
        elliptic_edges["txId2"].nunique(),
    )
    log_shape("After degree + graph features", elliptic_tx)
    print(
        elliptic_tx[
            ["in_degree", "out_degree", "total_degree", "is_isolated", "degree_in_out_ratio", "high_degree"]
        ].describe()
    )

    order = ["licit", "illicit", "unknown"]
    vc = elliptic_tx["class_label"].value_counts()
    vc = vc.reindex([o for o in order if o in vc.index], fill_value=0)
    sns.barplot(x=vc.index, y=vc.values)
    plt.title("Elliptic class counts")
    _save_fig("stage01_elliptic_class_counts", save_plots, figures_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    elliptic_tx.groupby("time_step").size().plot(ax=axes[0], title="Transactions per time_step")
    elliptic_tx[elliptic_tx["class_label"] == "illicit"].groupby("time_step").size().plot(
        ax=axes[1], title="Illicit count per time_step"
    )
    plt.tight_layout()
    _save_fig("stage01_elliptic_timestep", save_plots, figures_dir)

    for fname in ["feature_1", "feature_2", "feature_3"]:
        if fname in elliptic_tx.columns:
            sns.histplot(elliptic_tx[fname], bins=40, kde=True)
            plt.title(fname)
            _save_fig(f"stage01_elliptic_{fname}", save_plots, figures_dir)

    for col in ["in_degree", "out_degree", "total_degree", "degree_in_out_ratio"]:
        plt.figure()
        for lab, sub in elliptic_tx.groupby("class_label"):
            if lab == "unknown":
                continue
            sns.kdeplot(sub[col], label=lab, common_norm=False)
        plt.legend()
        plt.title(f"{col} (licit vs illicit)")
        _save_fig(f"stage01_elliptic_kde_{col}", save_plots, figures_dir)

    small = (
        ["time_step", "in_degree", "out_degree", "total_degree", "degree_in_out_ratio"]
        + [f"feature_{i}" for i in range(1, 6)]
    )
    small = [c for c in small if c in elliptic_tx.columns]
    sub_il = elliptic_tx[elliptic_tx["class_label"].isin(["licit", "illicit"])]
    cm2 = sub_il[small].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm2, cmap="coolwarm", center=0)
    plt.title("Elliptic: selected feature correlation (licit + illicit only)")
    plt.tight_layout()
    _save_fig("stage01_elliptic_corr", save_plots, figures_dir)

    compare = sub_il.groupby("class_label")[
        ["time_step", "in_degree", "out_degree", "total_degree", "degree_in_out_ratio"]
    ].mean()
    print(compare)

    elliptic_out = PROCESSED_DIR / "elliptic_transactions_cleaned.csv"
    extra_graph = ["is_isolated", "degree_in_out_ratio", "high_degree"]
    cols_save = (
        ["txId", "time_step", "class_label", "class", "in_degree", "out_degree", "total_degree"]
        + extra_graph
        + [c for c in elliptic_tx.columns if c.startswith("feature_")]
    )
    cols_save = [c for c in cols_save if c in elliptic_tx.columns]
    log_shape("Elliptic — columns to save", elliptic_tx[cols_save])
    elliptic_tx[cols_save].to_csv(elliptic_out, index=False)
    print("Saved:", elliptic_out)

    preprocessing_config = {
        "pipeline": "fraud_ml.pipeline.stage01_data",
        "missing_drop_threshold": MISSING_DROP_THRESHOLD,
        "ieee": {
            "numeric_imputation": "median (sklearn SimpleImputer)",
            "categorical_imputation": "fillna(Unknown)",
            "dropped_columns_high_missing": drop_cols,
            "n_dropped_columns": len(drop_cols),
            "output_files": [str(PROCESSED_DIR / "ieee_train_merged_cleaned.csv")],
            "final_shape": list(ieee_clean.shape),
            "target_distribution": ieee_clean["isFraud"].value_counts().to_dict(),
            "engineered_columns_ieee": [
                "rel_day",
                "rel_hour",
                "is_weekend_like",
                "amt_log",
                "amt_to_rel_day_mean_ratio",
                "amt_to_card1_mean_ratio",
            ],
        },
        "elliptic": {
            "class_mapping": "raw -> licit | illicit | unknown",
            "eda_keeps_unknown": True,
            "modeling_note": "Typically train on licit vs illicit; handle unknown separately.",
            "graph_features": [
                "in_degree",
                "out_degree",
                "total_degree",
                "is_isolated",
                "degree_in_out_ratio",
                "high_degree (>= p90 total_degree)",
            ],
            "output_files": [str(PROCESSED_DIR / "elliptic_transactions_cleaned.csv")],
            "final_shape": list(elliptic_tx.shape),
            "class_distribution": elliptic_tx["class_label"].value_counts().to_dict(),
        },
    }

    config_path = PROCESSED_DIR / "preprocessing_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_config, f, indent=2, default=str)
    print("Saved:", config_path)

    print("\n" + "=" * 60)
    print("DATA READY FOR NEXT STAGE")
    print("=" * 60)
    print("\nIEEE-CIS")
    print("  Rows, cols:", ieee_clean.shape)
    print("  Target isFraud:", ieee_clean["isFraud"].value_counts().to_dict())
    print("  Files:", ieee_out_full.name, ",", ieee_out_eda.name)
    print("\nElliptic")
    print("  Rows, cols:", elliptic_tx.shape)
    print("  class_label:", elliptic_tx["class_label"].value_counts().to_dict())
    print("  File:", elliptic_out.name)
    print("\nConfig:", config_path.name)
    print("=" * 60)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
