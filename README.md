# AI Revenue Leakage Detection — Hybrid ML System

End-to-end **revenue leakage / payment fraud** detection on **IEEE-CIS** transactional data, with optional **Elliptic** Bitcoin graph analytics, **gradient-boosted trees**, **deep learning**, **anomaly detection**, and a **hybrid fusion** stage.

---

## Problem statement

**Revenue leakage** in digital payments appears as a small fraction of high-risk transactions hidden in massive legitimate traffic. Class imbalance, noisy identity fields, and nonlinear fraud patterns make single-model solutions brittle. This project frames detection as a **supervised fraud classification** problem with **complementary signals** (tabular GBDT scores, neural risk scores, unsupervised anomaly scores) fused for **better precision–recall tradeoffs** than any single channel alone.

---

## Datasets

| Source | Role | Location |
|--------|------|----------|
| **IEEE-CIS Fraud Detection** | Primary tabular train slice: transactions + identity merge, engineered features, stages 1–4 | `DATASET_ieee-cis-elliptic/` (see bundle README) |
| **Elliptic++** (Bitcoin) | Graph-oriented EDA and Stage 5 experiments (temporal holdout) | Same bundle; cleaned table in `processed_data/elliptic_transactions_cleaned.csv` |

After running the pipeline, **processed** IEEE tables, predictions, and configs live under **`processed_data/`** (included in this submission where present).

---

## Methodology

1. **EDA & data prep (Stage 1)** — Merge IEEE transaction + identity, missingness-driven column drops, imputation, time/amount feature engineering, class and temporal EDA; Elliptic cleaning and graph-derived degree features.
2. **Feature engineering** — Ratios to peer/card means, calendar features, log transforms; high-missing columns removed with documented thresholds (`preprocessing_config.json`).
3. **Models (Stages 2–3)** — **GBDT** (LightGBM / XGBoost) with optional SMOTE/tuning; **attention / MLP-style DNN** baselines; **Isolation Forest** anomaly scores on normalized features; intermediate prediction CSVs for downstream fusion.
4. **Hybrid system (Stage 4)** — Weighted or learned combination of GBDT, DNN, and anomaly channels with **threshold tuning** on validation (F1-oriented under imbalance); test metrics and score exports.
5. **Graph extension (Stage 5)** — Elliptic temporal split and graph-model experiments (`elliptic_graph_experiments.csv`).

---

## Models used

| Family | Implementation | Notes |
|--------|----------------|--------|
| **GBDT** | LightGBM / XGBoost via `stage02_gbdt.py` | Strong tabular baseline; high recall, lower precision at default 0.5 threshold |
| **Deep learning** | TensorFlow/Keras attention + MLP paths in `stage03_deep_anomaly.py` | Strong ROC-AUC / balanced precision–recall vs GBDT |
| **Anomaly detection** | Normalized **Isolation Forest** scores | Captures outliers; weak alone on fraud label |
| **Hybrid** | Fusion in `stage04_fusion.py` | Calibrated combination beats single-channel F1 on bundled run |

---

## Results summary (bundled run)

Values from `processed_data/report_table_1_ieee_cis.csv` and `final_hybrid_comparison_metrics.csv` (IEEE test, this checkout):

| Model | Precision | Recall | F1 | AUC (ROC) |
|-------|-----------|--------|-----|-----------|
| XGBoost (GBDT) | 0.260 | 0.822 | 0.395 | 0.943 |
| Deep neural network | 0.827 | 0.611 | 0.702 | 0.946 |
| Isolation forest | 0.290 | 0.122 | 0.172 | 0.769 |
| **Proposed hybrid** | **0.801** | **0.630** | **0.705** | **0.945** |

Additional **PR-AUC** for the hybrid (weighted) on the same run: **0.736** (`final_hybrid_comparison_metrics.csv`).

---

## Why the hybrid works best

- **GBDT** excels at sharp nonlinear splits on mixed-type tabular features but at **0.5 threshold** favors **recall** over **precision** on imbalanced fraud.
- The **DNN** learns smooth, high-dimensional representations and achieves **strong precision** with moderate recall.
- **Anomaly scores** add a complementary “unknown pattern” signal but are **noisy** as a standalone classifier.
- **Fusion** reweights channels so that high-confidence DNN and tree signals are not drowned out by the anomaly channel, yielding **highest F1** and a practical precision–recall balance for operations.

---

## Streamlit dashboard

- **Entry:** `app/streamlit_app.py` (same home experience as `app/app.py`).
- **Home** — Hero KPIs, pipeline stepper, dataset overview, key equations, inlined **Tables I–IV**, figure gallery with short inferences, business framing, threshold explorer.
- **Pages** (`app/pages/`) — EDA gallery, GBDT/SHAP, deep + anomaly, fusion downloads, pipeline runner, Elliptic graph, reports.

Theme and layout: `app/assets/custom.css`, `.streamlit/config.toml`.

### Screenshots (optional)

Add PNGs under `docs/dashboard_screenshots/` and link them here for your report; figures from the pipeline remain under `figures/`.

---

## How to run

**Python:** 3.10, 3.11, or 3.12 (see `requirements.txt` header).

```bash
cd AI_Revenue_Project
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Windows shortcut:** double-click `run_dashboard.bat` from this folder (runs the same command).

**Full pipeline (optional, long run):**

```bash
python main.py
```

---

## Folder structure

```text
AI_Revenue_Project/
├── README.md                 # This file
├── requirements.txt
├── pyproject.toml
├── main.py                   # CLI pipeline entry
├── run_dashboard.bat         # Windows launcher
├── LICENSE
├── .gitignore
├── .streamlit/config.toml
├── app/
│   ├── streamlit_app.py      # Primary Streamlit entry
│   ├── app.py                # Alternate entry
│   ├── assets/, components/, pages/
├── src/fraud_ml/             # Pipeline package
├── DATASET_ieee-cis-elliptic/   # Raw data bundle
├── processed_data/           # Cleaned CSVs, preds, metrics, configs
├── figures/                  # Stage plots (EDA → graph)
├── docs/                     # Demo, checklists, stage report
├── manuscript/               # Draft narrative (excludes obsolete TeX)
├── scripts/                  # e.g. appendix table embed helper
├── data/, models/, results/  # README stubs → see paths above
├── .github/workflows/        # CI smoke (optional)
└── ML_Report_Latest.tex      # LaTeX report (if present)
```

---

## License and contributing

See `LICENSE` and `CONTRIBUTING.md`.
