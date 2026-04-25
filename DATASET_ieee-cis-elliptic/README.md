# Datasets (local only — not in Git)

Place downloaded data here so paths match the pipeline:

```
DATASET_ieee-cis-elliptic/
├── ieee-fraud-detection/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   ├── test_identity.csv
│   └── sample_submission.csv
└── elliptic-dataset/
    └── elliptic_bitcoin_dataset/
        ├── elliptic_txs_features.csv
        ├── elliptic_txs_classes.csv
        └── elliptic_txs_edgelist.csv
```

**Sources**

- **IEEE-CIS:** [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
- **Elliptic:** [Kaggle — Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (or project-provided mirror)

This folder is **gitignored**; clone the repo, then download and extract the CSVs locally.
