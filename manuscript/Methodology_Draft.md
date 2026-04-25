# Methodology Draft

## 1) Problem Definition

This project targets fraud / revenue leakage detection as an imbalanced binary classification problem, with anomaly detection support for rare or previously unseen behavior.

- **IEEE-CIS target:** `isFraud` (0/1)
- **Elliptic target:** mapped class labels (`illicit` vs `licit`, `unknown` handled per experiment design)
- **Primary metrics:** Recall, Precision, F1-score, ROC-AUC (plus PR-AUC where available)

## 2) Data Sources

### IEEE-CIS (Tabular)
- `train_transaction.csv`
- `train_identity.csv`
- Joined on `TransactionID`

### Elliptic (Graph + Tabular)
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

## 3) Preprocessing Pipeline (stage 1)

1. Load raw files from `DATASET_ieee-cis-elliptic/`
2. Merge transaction/identity and features/classes
3. Missing-value handling:
   - Drop very sparse columns (threshold-based)
   - Numeric imputation: median
   - Categorical imputation: `Unknown`
4. Label normalization and consistency checks
5. Feature engineering:
   - IEEE: time-based, amount-based, and ratio features
   - Elliptic: in/out/total degree and graph summary flags
6. EDA for imbalance, distributions, correlation, and degree patterns
7. Save prepared files into `processed_data/`

## 4) Modeling Strategy

### Stage 2: Baselines + GBDT
- Baseline models: Logistic Regression, Decision Tree, Random Forest
- Main model: LightGBM (XGBoost fallback)
- Produce probability outputs for downstream fusion (`gbdt_preds.csv`)

### Stage 3: Deep Learning + Anomaly
- Attention-based DNN on processed numeric inputs
- Isolation Forest for anomaly scoring
- Produce DNN probability and anomaly scores (`hybrid_dnn_anomaly_preds.csv`)

### Stage 4: Hybrid Fusion + Final Evaluation
- Merge component outputs (GBDT + DNN + anomaly)
- Compute weighted fusion score
- Tune threshold on validation split
- Evaluate against component-only models

## 5) Class Imbalance Handling

- Stratified splits for model development/evaluation
- Class weights and/or sampling strategies where appropriate
- No leakage policy: balancing/sampling only on training data

## 6) Evaluation Protocol

Models are compared on a common holdout test split using:
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix

Threshold-dependent metrics are reported with the selected threshold policy (default 0.5 for component models, tuned threshold for final hybrid).

## 7) Reproducibility and Artifacts

- Random seeds specified in pipeline code (`src/fraud_ml/`)
- Intermediate model outputs persisted in `processed_data/`
- Final comparison and threshold exported by stage 4
