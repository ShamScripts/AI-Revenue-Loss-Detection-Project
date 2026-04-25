"""Stage 4: fuse scores, tune threshold, save final metrics and scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fraud_ml.config.paths import get_paths
from fraud_ml.pipeline.split_utils import (
    SplitMode,
    fusion_random_train_val_test,
    fusion_temporal_train_val_test,
)
from fraud_ml.reporting.report_tables import build_and_save_report_tables

sns.set(style="whitegrid")
RANDOM_STATE = 42
W_GBDT = 0.45
W_DNN = 0.40
W_ANOM = 0.15


def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_score(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def metrics_from_proba(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run(
    project_root: Path | None = None,
    save_plots: bool = True,
    *,
    split_mode: SplitMode = "random",
) -> None:
    paths = get_paths(project_root)
    PROCESSED_DIR = paths.processed_dir
    figures_dir = paths.figures
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    gbdt_path = PROCESSED_DIR / "gbdt_preds.csv"
    dnn_anom_path = PROCESSED_DIR / "hybrid_dnn_anomaly_preds.csv"
    ieee_path = PROCESSED_DIR / "ieee_train_eda_ready.csv"

    missing_required = [str(p.name) for p in [gbdt_path, dnn_anom_path] if not p.exists()]
    if missing_required:
        raise FileNotFoundError(
            "Missing required files in processed_data: "
            + ", ".join(missing_required)
            + "\nRun stages 2 and 3 first."
        )

    gbdt_df = pd.read_csv(gbdt_path)
    dnn_df = pd.read_csv(dnn_anom_path)
    ieee_df = pd.read_csv(ieee_path) if ieee_path.exists() else None

    print("gbdt_df shape:", gbdt_df.shape)
    print("dnn_df shape:", dnn_df.shape)
    print("ieee_df shape:", None if ieee_df is None else ieee_df.shape)

    id_candidates = ["TransactionID", "transaction_id", "txId", "tx_id"]
    target_candidates = ["isFraud", "target", "label", "y"]
    id_g = pick_first_existing(gbdt_df, id_candidates)
    id_d = pick_first_existing(dnn_df, id_candidates)
    gbdt_score_col = pick_first_existing(gbdt_df, ["gbdt_pred_proba", "pred_proba", "score"])
    dnn_score_col = pick_first_existing(dnn_df, ["dnn_pred_proba", "pred_proba", "dl_pred_proba"])
    anom_score_col = pick_first_existing(dnn_df, ["anomaly_score", "iforest_score", "anomaly"])
    hybrid03_col = pick_first_existing(dnn_df, ["hybrid_score"])

    if gbdt_score_col is None:
        raise ValueError("Could not find a GBDT score column in gbdt_preds.csv")
    if dnn_score_col is None:
        raise ValueError("Could not find a DNN score column in hybrid_dnn_anomaly_preds.csv")
    if anom_score_col is None:
        raise ValueError("Could not find an anomaly score column in hybrid_dnn_anomaly_preds.csv")

    if id_g and id_d:
        dnn_pick = [id_d, dnn_score_col, anom_score_col] + ([hybrid03_col] if hybrid03_col else [])
        if "TransactionDT" in dnn_df.columns:
            dnn_pick.append("TransactionDT")
        score_df = dnn_df[dnn_pick].copy()
        score_df = score_df.merge(
            gbdt_df[[id_g, gbdt_score_col]].copy(), left_on=id_d, right_on=id_g, how="left"
        )
        if id_g != id_d:
            score_df.drop(columns=[id_g], inplace=True)
        score_df.rename(columns={id_d: "record_id"}, inplace=True)
    else:
        min_len = min(len(gbdt_df), len(dnn_df))
        score_df = pd.DataFrame(
            {
                "record_id": np.arange(min_len),
                dnn_score_col: dnn_df[dnn_score_col].iloc[:min_len].values,
                anom_score_col: dnn_df[anom_score_col].iloc[:min_len].values,
                gbdt_score_col: gbdt_df[gbdt_score_col].iloc[:min_len].values,
            }
        )
        if hybrid03_col:
            score_df[hybrid03_col] = dnn_df[hybrid03_col].iloc[:min_len].values

    target_col = pick_first_existing(score_df, target_candidates)
    if target_col is None and "isFraud" in dnn_df.columns:
        score_df["isFraud"] = dnn_df["isFraud"].values[: len(score_df)]
        target_col = "isFraud"

    if target_col is None and ieee_df is not None:
        id_i = pick_first_existing(ieee_df, id_candidates)
        if "isFraud" in ieee_df.columns:
            if id_i and "record_id" in score_df.columns and id_i in ieee_df.columns:
                score_df = score_df.merge(ieee_df[[id_i, "isFraud"]], left_on="record_id", right_on=id_i, how="left")
                if id_i != "record_id":
                    score_df.drop(columns=[id_i], inplace=True)
            else:
                score_df["isFraud"] = ieee_df["isFraud"].iloc[: len(score_df)].values
            target_col = "isFraud"

    if target_col is None:
        raise ValueError(
            "Target label not found. Ensure isFraud exists in hybrid_dnn_anomaly_preds.csv or ieee_train_eda_ready.csv."
        )

    score_df.rename(
        columns={
            gbdt_score_col: "gbdt_score_raw",
            dnn_score_col: "dnn_score_raw",
            anom_score_col: "anomaly_score_raw",
            target_col: "target",
        },
        inplace=True,
    )
    if hybrid03_col and hybrid03_col in score_df.columns:
        score_df.rename(columns={hybrid03_col: "hybrid03_score_raw"}, inplace=True)

    score_df = score_df.dropna(subset=["gbdt_score_raw", "dnn_score_raw", "anomaly_score_raw", "target"]).copy()
    score_df["target"] = score_df["target"].astype(int)
    score_df["gbdt_score"] = normalize_score(score_df["gbdt_score_raw"])
    score_df["dnn_score"] = normalize_score(score_df["dnn_score_raw"])
    score_df["anomaly_score"] = normalize_score(score_df["anomaly_score_raw"])
    if "hybrid03_score_raw" in score_df.columns:
        score_df["hybrid03_score"] = normalize_score(score_df["hybrid03_score_raw"])

    print("Final score_df shape:", score_df.shape)

    if split_mode == "temporal" and "TransactionDT" in score_df.columns:
        print("Fusion split mode: temporal (ordered by TransactionDT)")
        dev_df, val_df, test_df = fusion_temporal_train_val_test(
            score_df, test_size=0.2, val_frac_of_train=0.25, random_state=RANDOM_STATE
        )
    elif split_mode == "temporal":
        print("[warn] TransactionDT not in fused table — using random stratified fusion split.")
        dev_df, val_df, test_df = fusion_random_train_val_test(
            score_df, test_size=0.2, val_frac_of_train=0.25, random_state=RANDOM_STATE
        )
    else:
        print("Fusion split mode: random stratified")
        dev_df, val_df, test_df = fusion_random_train_val_test(
            score_df, test_size=0.2, val_frac_of_train=0.25, random_state=RANDOM_STATE
        )
    print("dev:", dev_df.shape, "val:", val_df.shape, "test:", test_df.shape)

    (PROCESSED_DIR / "stage04_experiment_config.json").write_text(
        json.dumps({"split_mode": split_mode, "stage": 4}, indent=2),
        encoding="utf-8",
    )

    for frame in [dev_df, val_df, test_df, score_df]:
        frame["hybrid_weighted_score"] = (
            W_GBDT * frame["gbdt_score"] + W_DNN * frame["dnn_score"] + W_ANOM * frame["anomaly_score"]
        )

    thresholds = np.linspace(0.05, 0.95, 91)
    f1_vals = []
    for t in thresholds:
        pred = (val_df["hybrid_weighted_score"] >= t).astype(int)
        f1_vals.append(f1_score(val_df["target"], pred, zero_division=0))

    best_idx = int(np.argmax(f1_vals))
    best_threshold = float(thresholds[best_idx])
    best_val_f1 = float(f1_vals[best_idx])
    print(f"Best threshold (val F1): {best_threshold:.3f}")
    print(f"Best validation F1: {best_val_f1:.4f}")

    model_scores = {
        "GBDT": test_df["gbdt_score"],
        "DNN": test_df["dnn_score"],
        "Anomaly": test_df["anomaly_score"],
        "Hybrid_Weighted": test_df["hybrid_weighted_score"],
    }
    if "hybrid03_score" in test_df.columns:
        model_scores["Hybrid_From_Stage03"] = test_df["hybrid03_score"]

    rows = []
    for name, score in model_scores.items():
        threshold = best_threshold if name == "Hybrid_Weighted" else 0.5
        m = metrics_from_proba(test_df["target"], score, threshold=threshold)
        m["model"] = name
        m["threshold"] = threshold
        rows.append(m)

    results_df = pd.DataFrame(rows)[["model", "threshold", "roc_auc", "pr_auc", "precision", "recall", "f1"]]
    results_df = results_df.sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    print(results_df.to_string())

    y_true = test_df["target"].astype(int).values
    y_prob = test_df["hybrid_weighted_score"].values
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report (Hybrid Weighted):\n")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Hybrid Weighted - Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "stage04_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, score in model_scores.items():
        fpr, tpr, _ = roc_curve(test_df["target"], score)
        auc_val = roc_auc_score(test_df["target"], score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.title("ROC Curves (Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    for name, score in model_scores.items():
        precision, recall, _ = precision_recall_curve(test_df["target"], score)
        pr_auc = average_precision_score(test_df["target"], score)
        plt.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")
    plt.title("Precision-Recall Curves (Test)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.tight_layout()
    if save_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "stage04_roc_pr.png", dpi=120, bbox_inches="tight")
    plt.close()

    all_scores_out = score_df.copy()
    all_scores_out["hybrid_weighted_score"] = (
        W_GBDT * all_scores_out["gbdt_score"]
        + W_DNN * all_scores_out["dnn_score"]
        + W_ANOM * all_scores_out["anomaly_score"]
    )
    all_scores_out["hybrid_weighted_pred"] = (all_scores_out["hybrid_weighted_score"] >= best_threshold).astype(int)

    results_path = PROCESSED_DIR / "final_hybrid_comparison_metrics.csv"
    scores_path = PROCESSED_DIR / "final_hybrid_scores.csv"
    threshold_path = PROCESSED_DIR / "final_hybrid_threshold.txt"

    results_df.to_csv(results_path, index=False)
    all_scores_out.to_csv(scores_path, index=False)
    threshold_path.write_text(f"{best_threshold:.6f}", encoding="utf-8")

    print("Saved:", results_path)
    print("Saved:", scores_path)
    print("Saved:", threshold_path)

    try:
        build_and_save_report_tables(
            score_df,
            test_df,
            best_threshold,
            project_root=paths.root,
            w_gbdt=W_GBDT,
            w_dnn=W_DNN,
            w_anom=W_ANOM,
        )
    except Exception as e:
        print(f"[warn] Could not build report tables: {e}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
