# Processed outputs (local — not in Git)

CSV and JSON artifacts are **generated** by the pipeline (`python main.py`).

Examples:

- `ieee_train_eda_ready.csv`, `elliptic_transactions_cleaned.csv`
- `gbdt_preds.csv`, `hybrid_dnn_anomaly_preds.csv`
- `final_hybrid_comparison_metrics.csv`, `final_hybrid_scores.csv`, `final_hybrid_threshold.txt`
- `report_table_1_ieee_cis.csv`, `report_table_2_elliptic.csv`, `report_table_3_ablation.csv`
- `elliptic_graph_experiments.csv` (Stage 5)
- `stage02_experiment_config.json`, `stage03_experiment_config.json`, `stage04_experiment_config.json`, `stage05_experiment_config.json`
- `preprocessing_config.json`

After cloning, run the pipeline to populate this directory. The Streamlit dashboard reads these files if present.
