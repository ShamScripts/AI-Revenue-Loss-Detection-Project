# Demo checklist (Streamlit + GitHub)

Use this when you want the **hosted app** to show the same tables, KPIs, and figures as on your machine.

## 1. Commit what the dashboard reads

The app resolves the project root via `DATASET_ieee-cis-elliptic/` and loads:

| Path | Purpose |
|------|--------|
| `processed_data/` | CSV / JSON artifacts (tables, fusion scores, report tables) |
| `figures/` | PNGs for the figure galleries |
| `DATASET_ieee-cis-elliptic/` | Raw / prepared data the pipeline expects (if you rely on it on the host) |

These paths are **tracked in git** for demo use (they are **not** listed in `.gitignore`). After a big first push, clones may be slow — that is normal.

## 2. Streamlit Community Cloud (typical)

1. Push this repo to GitHub (including the folders above).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo and branch.
3. **Main file:** `app/app.py`  
4. **App URL:** leave default or set a subdomain.
5. **Python:** **3.10–3.12** (match `requirements.txt`; 3.11 is a safe default).
6. Deploy and wait for the build. If the build times out, the dependency stack is heavy — retry or slim `requirements.txt` for a “dashboard-only” branch later.

Secrets are only needed if you move data to S3/GDrive; this project reads **local paths** under the repo.

## 3. Quick local demo (before / after deploy)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app/app.py
```

Open the URL Streamlit prints; use the stepper to walk **Overview → Results**.

## 4. If the hosted app is still empty

- Confirm those folders exist **on GitHub** (browse the repo in the browser).
- Confirm the deployed branch is the one you pushed.
- Check the Cloud **Logs** tab for missing file or import errors.

## 5. Course / privacy note

If the demo must **not** ship real data, use a separate branch with **sample-only** CSVs/PNGs and keep full runs private — do not rely on this doc for legal review of the datasets.
