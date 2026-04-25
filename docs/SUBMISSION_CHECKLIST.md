# Submission checklist (GitHub)

Use this before sharing the repository with instructors or reviewers.

**Two modes**

- **Code-only / privacy:** keep large folders **out of git** (restore `.gitignore` for `processed_data/`, `figures/`, `DATASET_ieee-cis-elliptic/` if needed). Reviewers clone and regenerate locally.
- **Hosted demo (Streamlit Cloud):** commit **`processed_data/`**, **`figures/`**, and **`DATASET_ieee-cis-elliptic/`** as needed so the live app shows data. See **[`DEMO.md`](DEMO.md)**.

---

- [ ] **Python 3.10–3.12** venv; `pip install -r requirements.txt` succeeds.
- [ ] **Data**: `DATASET_ieee-cis-elliptic/` populated for a full local run (and committed **only** if you chose demo mode above).
- [ ] **Pipeline**: `python main.py` (or `--skip-elliptic-graph` if omitting Stage 5) completes.
- [ ] **Optional**: `pip install torch` then `python main.py --stage 5` for Elliptic GCN metrics.
- [ ] **Git**: For **code-only** pushes, `git status` shows **no** large CSVs / `processed_data/*.csv` / raw dataset staged. For **demo** pushes, confirm you intend large files and branch size.
- [ ] **README** at repo root is up to date; **LICENSE** present if required by course.
- [ ] Remove machine-specific paths or secrets (none should be in code).
