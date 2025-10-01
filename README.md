# Fact Refresh (Daily Multilingual News DB)

This repository hosts a **daily, self-updating** news database + FAISS index published to **Hugging Face Datasets**.
Use it from **Colab** with one line: `load_dataset("<YOUR-HF-USERNAME>/news_articles_daily", split="train")`.

## Contents
- `refresh_db.py`: Orchestrates scrape → preprocess (summarize) → build embeddings → FAISS → publish to HF
- `scraper/sources.py`: Put your real scrapers here (BeautifulSoup/Newspaper3k/RSS/Selenium). Return normalized dicts.
- `scraper/preprocess.py`: Language detection + summarization (DistilBART). Add translation if needed.
- `embed/e5_embed.py`: E5-base embeddings (swap to multilingual e5 if needed).
- `embed/faiss_utils.py`: Build/read FAISS index.
- `.github/workflows/refresh.yml`: Daily cron via GitHub Actions (00:00 UTC = 05:30 IST).

## Quick Start
1. Create a **Hugging Face dataset** repo, e.g. `<your-username>/news_articles_daily` (private or public).
2. Create an HF **Access Token** (Write). Keep it secret.
3. Add the token to GitHub repo **Settings → Secrets and variables → Actions** as `HF_TOKEN`.
4. Edit `refresh_db.py` to set `HF_DATASET_ID` (or pass via env in workflow).
5. Replace TODOs in `scraper/sources.py` with your scraping logic.
6. Commit & push. Trigger the workflow in **Actions**, or wait for the daily cron.
7. In Colab:
   ```python
   !pip install datasets faiss-cpu pandas pyarrow
   from datasets import load_dataset
   ds = load_dataset("<your-username>/news_articles_daily", split="train")
   df = ds.to_pandas()
   ```

## Notes
- Default summarizer: `sshleifer/distilbart-cnn-12-6` (CPU okay for modest volume). You can set `SKIP_SUMMARY=1` to bypass.
- Default embeddings: `intfloat/e5-base`. For multilingual, swap to `intfloat/multilingual-e5-base` in `embed/e5_embed.py`.
- The index & meta are uploaded alongside the dataset as files `faiss.index` and `meta.json`.
