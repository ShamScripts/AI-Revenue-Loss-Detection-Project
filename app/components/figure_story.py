"""Narrative inference + business meaning for pipeline figures (text only — does not alter plots)."""

from __future__ import annotations

# stem -> (inference, business_meaning)
_STEM_STORY: dict[str, tuple[str, str]] = {
    "stage04_roc_pr": (
        "ROC summarizes rank ordering across thresholds; PR highlights performance under class imbalance. "
        "The deep model often leads ROC-AUC / PR-AUC, while the hybrid stays competitive and is tuned for F1.",
        "High AUC supports prioritization queues, but business value is tied to a chosen alert rate — the hybrid targets "
        "practical precision–recall trade-offs, not ranking alone.",
    ),
    "stage04_confusion_matrix": (
        "Confusion counts at the tuned hybrid threshold show how errors split between false alarms and missed fraud.",
        "Each false negative is potential revenue leakage; each false positive consumes analyst capacity — the matrix makes that trade-off explicit.",
    ),
    "stage03_anomaly_kde": (
        "Isolation Forest scores separate licit vs fraudulent mass to varying degrees; tails often flag rare behaviors.",
        "Anomaly scores surface unknown fraud modes; fused with supervised channels they improve robustness without replacing them.",
    ),
    "stage03_dnn_auc": (
        "Training vs validation AUC by epoch indicates whether the deep model is learning generalizable structure.",
        "Stable validation curves reduce deployment risk; compare against the DNN row in TABLE I for end-to-end test behavior.",
    ),
    "stage03_dnn_confusion": (
        "Neural channel confusion at 0.5 on validation probabilities shows where the DNN is aggressive or conservative.",
        "Guards against over-trusting raw neural scores before fusion and threshold tuning.",
    ),
    "stage02_gbdt_importance": (
        "Tree ensembles concentrate risk on a subset of transactional signals — useful for monitoring and policy rules.",
        "Business teams can align controls with top drivers (MCC, device, amount patterns) while models continue to retrain.",
    ),
    "stage02_shap_summary": (
        "SHAP summarizes marginal contributions — directionally consistent with importance but localized to score ranges.",
        "Supports investigator trust and post-hoc review without changing underlying model scores.",
    ),
    "stage01_isFraud_distribution": (
        "The class histogram underscores extreme imbalance — naive accuracy is misleading.",
        "Justifies recall-aware metrics, hybrid design, and cost-sensitive alerting in production narratives.",
    ),
    "stage01_boxplot_amt_by_fraud": (
        "Spend distributions differ between fraud and non-fraud, but overlap remains — univariate rules are insufficient.",
        "Motivates multivariate models and fusion rather than amount-only rules.",
    ),
    "stage05_elliptic_gcn_score_hist": (
        "GCN probabilities on Elliptic test nodes show separation between licit and illicit under the graph-aware setup.",
        "Relational modeling targets coordinated leakage patterns that tabular-only scores may underweight.",
    ),
}


def inference_and_business_for_stem(stem: str) -> tuple[str, str]:
    if stem in _STEM_STORY:
        return _STEM_STORY[stem]
    if stem.startswith("stage01_"):
        return (
            "Exploratory view of raw or lightly processed inputs; use with TABLE I–II for supervised outcomes.",
            "Data quality and drift monitoring reduce silent revenue leakage from stale features or broken ingestion.",
        )
    if stem.startswith("stage02_"):
        return (
            "GBDT diagnostics explain which tabular signals dominate tree decisions for this training run.",
            "Align investigator playbooks with dominant features while watching for adversarial drift on those fields.",
        )
    if stem.startswith("stage03_"):
        return (
            "Deep or anomaly channel visualization — read jointly with TABLE III ablations for channel contributions.",
            "Single channels are rarely sufficient; fusion exists to blend complementary error profiles.",
        )
    if stem.startswith("stage04_"):
        return (
            "Fusion-stage evaluation artifact — compare curves and matrices to TABLE I and TABLE III.",
            "Operational thresholds should be revisited as fraud economics or staffing constraints change.",
        )
    if stem.startswith("stage05_"):
        return (
            "Elliptic / graph experiment figure — complements TABLE II baselines.",
            "Graph-aware scoring can reveal structured fraud rings tied to revenue loss networks.",
        )
    return (
        "Pipeline figure — interpret alongside the report tables for the same experiment cohort.",
        "Link model behavior to investigation throughput and expected leakage exposure at your operating point.",
    )
