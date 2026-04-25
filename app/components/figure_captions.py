"""Human-readable titles and one-line explanations for saved pipeline figures."""

from __future__ import annotations

import re
from pathlib import Path

# Exact stem (no extension) -> (short title, one-line explanation)
_EXACT: dict[str, tuple[str, str]] = {
    "stage01_isFraud_distribution": (
        "IEEE — Fraud vs non-fraud counts",
        "Bar chart of class frequency; shows strong imbalance typical of fraud detection.",
    ),
    "stage01_boxplot_amt_by_fraud": (
        "IEEE — Transaction amount by label",
        "Boxplot of TransactionAmt for fraud vs non-fraud (outliers hidden for readability).",
    ),
    "stage01_top20_missing": (
        "IEEE — Missing data concentration",
        "Horizontal bar of the 20 columns with highest missing % before drop/impute.",
    ),
    "stage01_rel_hour": (
        "IEEE — Activity by hour of day",
        "Left: transaction volume by relative hour; right: fraud rate by hour.",
    ),
    "stage01_corr_heatmap": (
        "IEEE — Feature correlation snapshot",
        "Heatmap among selected numeric features and isFraud (Pearson).",
    ),
    "stage01_elliptic_class_counts": (
        "Elliptic — Class distribution",
        "Counts of licit vs illicit vs unknown after label normalization.",
    ),
    "stage01_elliptic_timestep": (
        "Elliptic — Activity over time_step",
        "Transaction volume and illicit counts across graph time steps.",
    ),
    "stage01_elliptic_corr": (
        "Elliptic — Feature correlation (licit + illicit)",
        "Correlation heatmap on labeled subset for structural insight.",
    ),
    "stage02_gbdt_importance": (
        "GBDT — Feature importance",
        "Top features by tree ensemble importance (LightGBM or XGBoost).",
    ),
    "stage03_dnn_auc": (
        "Deep model — Training AUC curves",
        "Train vs validation AUC by epoch (TensorFlow) or loss curve (MLP fallback).",
    ),
    "stage03_dnn_confusion": (
        "Deep model — Validation confusion matrix",
        "Classification of validation rows at 0.5 threshold on neural probabilities.",
    ),
    "stage03_anomaly_kde": (
        "Anomaly — Score density by class",
        "KDE of normalized Isolation Forest scores for fraud vs normal on validation.",
    ),
    "stage04_confusion_matrix": (
        "Fusion — Test confusion matrix",
        "Hybrid weighted model at the tuned threshold on held-out test data.",
    ),
    "stage04_roc_pr": (
        "Fusion — ROC & precision–recall",
        "Test-set ROC and PR curves comparing component models and the hybrid score.",
    ),
}


def caption_for_figure(path: Path) -> tuple[str, str]:
    """
    Return (title, explanation) for a figure path.
    Uses filename stem heuristics when no exact match exists.
    """
    stem = path.stem

    if stem in _EXACT:
        return _EXACT[stem]

    # stage01_hist_{column}
    m = re.match(r"stage01_hist_(.+)$", stem)
    if m:
        col = m.group(1)
        return (
            f"IEEE — Distribution: {col}",
            f"Histogram with KDE for `{col}` to inspect skew and tails.",
        )

    # stage01_elliptic_{feature}
    m = re.match(r"stage01_elliptic_(feature_\d+)$", stem)
    if m:
        return (
            f"Elliptic — Histogram: {m.group(1)}",
            "Distribution of the selected Elliptic feature (raw feature space).",
        )

    # stage01_elliptic_kde_{col}
    m = re.match(r"stage01_elliptic_kde_(.+)$", stem)
    if m:
        col = m.group(1)
        return (
            f"Elliptic — Density: {col}",
            f"KDE of `{col}` comparing licit vs illicit (unknown excluded).",
        )

    # Generic stage prefixes
    if stem.startswith("stage01_"):
        return (
            f"IEEE / Elliptic — {stem.replace('stage01_', '').replace('_', ' ').title()}",
            "Exploratory figure from Stage 1 (data loading & preprocessing).",
        )
    if stem.startswith("stage02_"):
        return (
            f"GBDT — {stem.replace('stage02_', '').replace('_', ' ').title()}",
            "Figure from Stage 2 (baselines & gradient boosting).",
        )
    if stem.startswith("stage03_"):
        return (
            f"Deep / anomaly — {stem.replace('stage03_', '').replace('_', ' ').title()}",
            "Figure from Stage 3 (neural model & Isolation Forest).",
        )
    if stem.startswith("stage04_"):
        return (
            f"Fusion — {stem.replace('stage04_', '').replace('_', ' ').title()}",
            "Figure from Stage 4 (score fusion & final evaluation).",
        )

    return (
        path.name,
        "Pipeline or exploratory figure; see project report for full interpretation.",
    )
