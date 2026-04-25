# Conclusion and Future Scope

## Conclusion

This project implements a staged hybrid pipeline for fraud/revenue leakage detection:

1. Data preparation and EDA on IEEE-CIS and Elliptic datasets
2. Baseline and GBDT benchmarking
3. Attention-based deep learning + anomaly scoring
4. Final fusion and thresholded evaluation

The final hybrid stage is designed to combine the strengths of:
- **GBDT:** robust tabular discrimination
- **Deep learning:** nonlinear pattern capture
- **Anomaly detection:** rare/unknown behavior sensitivity

Based on final metrics from stage 4, the conclusion should explicitly report:
- whether hybrid improves F1/ROC-AUC
- whether recall improves for minority detection
- any precision tradeoff introduced by higher sensitivity

## Current Limitations

- Primarily random stratified splitting (time-aware validation can be stronger for transaction data)
- Fixed-weight fusion by default (meta-learning/stacking can improve)
- Limited calibration and operating-point optimization for business cost constraints
- Explainability can be expanded (e.g., SHAP for final fused decisions)

## Future Scope

- Real-time deployment pipeline with streaming inference
- Temporal cross-validation and drift monitoring
- Learned fusion (stacking/meta-model) instead of static weights
- Graph neural network extension for Elliptic relational modeling
- Enhanced anomaly scoring (autoencoder/contrastive approaches)
- Explainable AI dashboards for analyst and compliance workflows
- Multi-dataset/domain transfer and robustness testing
