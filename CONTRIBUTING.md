# Contributing

This repository is primarily for **course submission and reproducibility**. If you extend it:

1. Use **Python 3.10–3.12** (see `README.md`).
2. Run the pipeline from the repo root with `PYTHONPATH=src` or `pip install -e .`.
3. Do not commit **raw datasets**, **processed CSVs**, or large **PDFs** (see `.gitignore`).
4. Keep changes focused; match existing style in `src/fraud_ml/` and `app/`.

For issues with Kaggle data access or environment setup, check `DATASET_ieee-cis-elliptic/README.md` first.
