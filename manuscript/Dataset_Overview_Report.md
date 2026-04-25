# Dataset Overview Report

**Project:** Hybrid Revenue Leakage / Fraud Detection System · **Stage 1:** `fraud_ml.pipeline.stage01_data`

---

## 1. Introduction

This project trains a **hybrid** stack—**GBDT** (XGBoost/LightGBM), **attention-based deep learning**, and **anomaly detection**—on **two datasets**: **IEEE-CIS** (tabular e-commerce fraud) and **Elliptic** (Bitcoin transactions as a graph). Stage 1 loads, merges, cleans, and engineers features; outputs live under `processed_data/`. This document summarizes **data scale**, **quality issues**, **preprocessing decisions**, and **which model each dataset supports**.

---

## 2. Dataset 1: IEEE-CIS Fraud Detection

Each row is **one transaction** (`TransactionID`). Source files: `train_transaction.csv` (core + `V*` features), `train_identity.csv` (device/id fields)—merged in Stage 1.

### 2.1 IEEE-CIS summary

| Metric | Value |
|--------|-------|
| Rows | 590,540 |
| Columns (transaction file only) | 394 |
| Merge | Left join **Transaction + Identity** on `TransactionID` |
| Rows after merge | 590,540 (unchanged) |
| Target | `isFraud` (binary) |
| Fraud count / rate | 20,663 / **~3.5%** (569,877 non-fraud) |

### 2.2 Data quality & actions

| Issue | Observation | Action (Stage 1) |
|-------|-------------|----------------------|
| Missing values | Many columns **>90%** missing (esp. post–identity merge) | **Drop** those columns |
| Moderate missing | Identity / categorical fields | **Median** (numeric), **`"Unknown"`** (non-numeric) |
| Imbalance | Fraud ~3.5% | Documented; **metrics / resampling / weights** in later notebooks |
| Dimensionality | Hundreds of columns pre-clean | Sparse-column drop + later modeling choices |

### 2.3 Features (at a glance)

| Block | Examples |
|-------|----------|
| Transaction | `TransactionDT`, `TransactionAmt`, `ProductCD`, card/addr, `C*`, `D*`, `V*` |
| Identity (merged) | `DeviceType`, `id_01`–`id_38`, … (sparse for many rows) |
| Engineered (Stage 1) | `rel_day`, `rel_hour`, `is_weekend_like`, `amt_log`, `amt_to_rel_day_mean_ratio`, `amt_to_card1_mean_ratio` |

### 2.4 Key EDA insights (Stage 1)

- Fraud is **highly imbalanced** (~**3.5%** positive class).
- **`TransactionAmt`** is **heavily skewed**; `amt_log` and ratio features stabilize tails.
- **Identity** columns are **sparse** but often **informative** where present (heterogeneous fraud rates by category / device).
- **Temporal** signal: **`rel_hour`** and volume/fraud patterns over relative time are **non-uniform**.
- Full correlation matrix **not** used (too many columns); **subset heatmap + top numeric correlations with `isFraud`** for interpretability (linear; fraud can be nonlinear).

### 2.5 Challenges (short)

| Challenge | Note |
|-----------|------|
| High missingness | After identity merge |
| High dimensionality | Before/after sparse drop |
| Class imbalance | PR-style metrics essential |

---

## 3. Dataset 2: Elliptic Bitcoin

Each row is **one transaction node** (`txId`). Files: features (no header in our copy), `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.

### 3.1 Elliptic summary

| Metric | Value |
|--------|-------|
| Nodes (transactions) | 203,769 |
| Edges (directed) | 234,355 |
| Local features | **165** (`feature_1`…`feature_165`) + `time_step` |
| Classes | **licit / illicit / unknown** (normalized in Stage 1) |

### 3.2 Graph-derived features (Stage 1)

| Feature | Meaning |
|---------|---------|
| In-degree | Count of incoming edges |
| Out-degree | Count of outgoing edges |
| Total degree | In + out (overall connectivity) |
| `is_isolated` | No incident edges (in and out degree 0) |
| `degree_in_out_ratio` | in_degree / (out_degree + 1) (flow balance) |
| `high_degree` | 1 if total_degree ≥ **90th percentile** (hub-like) |

### 3.3 Data quality & actions

| Issue | Observation | Action (Stage 1) |
|-------|-------------|----------------------|
| Unknown labels | Present in raw data | **Kept for EDA**; supervised modeling often **licit vs illicit** only |
| Graph complexity | Large edge list | **Degree + simple flags** (not full GNN in Stage 1) |
| Class mix | Imbalance across licit/illicit/unknown | Documented; **handled in modeling** |
| Feature NA | Typically lighter than IEEE | Checked; degrees from full edgelist |

### 3.4 Key EDA insights (Stage 1)

- **Three-class** structure (**unknown** = partial supervision).
- **Degree distributions** differ between licit and illicit (connectivity signal).
- **`time_step`**: volume and illicit counts **vary over time** (non-stationarity / temporal splits later).

---

## 4. Preprocessing summary (unified)

| Step | IEEE-CIS | Elliptic |
|------|----------|----------|
| Merge | Transaction ⋊ Identity on `TransactionID` | Features ⋈ classes on `txId` |
| Column removal | **>90%** missing → **drop** | Not the main issue |
| Missing handling | **Median** (numeric), **`"Unknown"`** (categorical) | Minimal on features; **graph from full edgelist** |
| Label handling | Binary `isFraud` | Normalize to `class_label`; **EDA keeps all 3 classes** |
| Feature engineering | Time + amount ratios (see §2.3) | Degrees + isolation + ratio + high-degree |

---

## 5. Model relevance mapping

| Component | Dataset (primary) | Purpose |
|-----------|-------------------|---------|
| **GBDT** | IEEE-CIS | Strong baseline; **structured** tabular patterns & interactions |
| **Deep learning (attention)** | IEEE-CIS | **Nonlinear** boundaries; embeddings on categoricals + dense numerics |
| **Anomaly detection** | Elliptic (also tail cases on IEEE later) | **Rare** illicit behavior, **unusual** connectivity / isolation / hubs |

---

## 6. Combined justification (why both datasets)

| Aspect | IEEE-CIS | Elliptic |
|--------|----------|----------|
| Data type | **Tabular** (+ merged identity) | **Graph** (nodes + edges) |
| Dominant signal | Transaction / device **attributes** | **Relational** + local features |
| Role in hybrid | **Structured learning** (GBDT + DL) | **Anomaly + relational** features (degrees, hubs, flow) |
| Pattern | Point-level fraud in rich features | Network-level + **time_step** structure |

**In two lines:** IEEE-CIS supplies **dense supervised tabular fraud** for trees and neural models; Elliptic supplies **explicit edges** and **partial labels**, matching **anomaly** and **graph-aware** workstreams. Together they cover **attribute-only** and **relation-explicit** regimes without duplicating the same benchmark.

---

## 7. Final prepared data (Stage 1)

| Output | Contents |
|--------|----------|
| `ieee_train_merged_cleaned.csv`, `ieee_train_eda_ready.csv` | Cleaned merge + imputation + engineered columns |
| `elliptic_transactions_cleaned.csv` | Labels + 165 features + graph scalars |
| `ieee_missing_top20_summary.csv` | Missingness documentation |
| `preprocessing_config.json` | Thresholds, policies, shapes (after a full run) |

**Readiness:** Tables are **modeling-ready** for Stages 2+ (splits, encoding, scaling, imbalance, GBDT/DL/anomaly).

---

## 8. Summary

IEEE-CIS: **large imbalanced tabular** fraud task with **heavy missingness** managed by **>90% column drop** and **median / Unknown** imputation, plus **time and amount** engineering—ideal for **GBDT** and **DL**. Elliptic: **203k nodes**, **234k edges**, **165** features, **three-way labels**, with **degree and hub/isolation** features—ideal for **anomaly** and **relational** analysis alongside tabular baselines. The **hybrid** system is justified by **complementary data geometries**: **column-rich transactions** vs **edge-rich Bitcoin flows**, aligned with Stage 1 outputs.

---

*Figures tie to stage 1 outputs and optional `preprocessing_config.json` after execution.*
