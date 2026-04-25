# Models directory (convention)

Tree models (LightGBM / XGBoost) and neural nets in this project are trained in the pipeline and scored on disk as **prediction CSVs** and **metrics tables** under `processed_data/` (for example `gbdt_preds.csv`, `hybrid_dnn_anomaly_preds.csv`, `final_hybrid_scores.csv`). Serialized `.pkl` / `.keras` bundles are not required for the bundled dashboard, which reads those artifacts.
