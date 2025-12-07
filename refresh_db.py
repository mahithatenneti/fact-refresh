import os
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfFolder, create_repo, upload_file
from datetime import datetime
from dateutil import tz

from scraper.sources import scrape_all
from scraper.preprocess import normalize_record

# ========= SETTINGS ============
HF_DATASET_ID = "mahi1010/news_articles_daily_m"
ART_PARQUET = "data/news_articles.parquet"
FAISS_PATH  = "data/faiss.index"
META_PATH   = "data/meta.json"

def ist_now():
    utc = datetime.utcnow().replace(tzinfo=tz.tzutc())
    ist = utc.astimezone(tz.gettz("Asia/Kolkata"))
    return ist.strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_hf_repo():
    create_repo(HF_DATASET_ID, repo_type="dataset", exist_ok=True, private=True)

def load_previous_df():
    try:
        ds = load_dataset(HF_DATASET_ID, split="train")
        df = ds.to_pandas()
        print(f"[INFO] Loaded previous: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[WARN] No previous dataset found ({e})")
        return pd.DataFrame(columns=[
            "uid","title","summary","lang","source","date","url"
        ])

def dedup_merge(old_df, new_df):
    all_df = pd.concat([old_df, new_df], ignore_index=True)

    # date normalization
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce", utc=True)

    # remove duplicates based on uid, keep newest
    all_df = all_df.sort_values("date", ascending=False)
    all_df = all_df.drop_duplicates(subset=["uid"], keep="first")

    # cleanup
    all_df["summary"] = all_df["summary"].fillna("")
    return all_df.reset_index(drop=True)

def main():
    print(f"[{ist_now()}] Starting refresh")

    ensure_hf_repo()

    # ---------- SCRAPE ----------
    raw = scrape_all()
    print(f"[INFO] scraped: {len(raw)}")

    processed = [normalize_record(r) for r in raw if r.get("uid")]

    new_df = pd.DataFrame([{
        "uid": r.get("uid"),
        "title": r.get("title",""),
        "summary": r.get("summary",""),
        "lang": r.get("lang","en"),
        "source": r.get("source",""),
        "date": r.get("date") or datetime.utcnow().isoformat(),
        "url": r.get("url",""),
    } for r in processed])

    # ---------- LOAD OLD + MERGE ----------
    old_df = load_previous_df()
    merged = dedup_merge(old_df, new_df)
    print(f"[INFO] final merged rows: {len(merged)}")

    # ---------- SAVE TO HF ----------
    ds = Dataset.from_pandas(merged)
    ds.push_to_hub(
        HF_DATASET_ID,
        private=True,
        split="train",
        max_shard_size="500MB"
    )

    print(f"[{ist_now()}] Refresh complete!")


if __name__ == "__main__":
    main()
