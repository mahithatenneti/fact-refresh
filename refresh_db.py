import os
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfFolder, create_repo
from datetime import datetime
from dateutil import tz

from scraper.sources import scrape_all
from scraper.preprocess import normalize_record

# ====== CONFIG (env or edit here) ======
HF_TOKEN = os.getenv("HF_TOKEN")                # set in GitHub Actions secrets
HF_DATASET_ID = os.getenv(
    "HF_DATASET_ID",
    "your-username/news_articles_daily"
)
ART_PARQUET = "data/news_articles.parquet"


def ist_now():
    utc = datetime.utcnow().replace(tzinfo=tz.tzutc())
    ist = utc.astimezone(tz.gettz("Asia/Kolkata"))
    return ist.strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_hf_repo():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set (GitHub Actions secret)")
    HfFolder.save_token(HF_TOKEN)
    # keep dataset private (change private=False if you want it public)
    create_repo(HF_DATASET_ID, repo_type="dataset", exist_ok=True, private=True)


def load_previous_df():
    """Load previous HF dataset (if exists) so we can append & deduplicate."""
    try:
        ds = load_dataset(HF_DATASET_ID, split="train")
        return ds.to_pandas()
    except Exception:
        return pd.DataFrame(
            columns=["uid", "title", "summary", "lang", "source", "date", "url", "text_for_embed"]
        )


def dedup_merge(old_df, new_df):
    all_df = pd.concat([old_df, new_df], ignore_index=True)

    # normalize datetimes → force UTC to avoid mixed tz error
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce", utc=True)

    # keep the newest per uid
    all_df = all_df.sort_values("date", ascending=False, na_position="last")
    all_df = all_df.drop_duplicates(subset=["uid"], keep="first")

    # enforce dtypes / fill
    for c in ["title", "summary", "lang", "source", "url", "text_for_embed"]:
        if c in all_df.columns:
            all_df[c] = all_df[c].fillna("")

    return all_df


def main():
    print(f"[{ist_now()}] start refresh")
    ensure_hf_repo()

    # 1) SCRAPE
    raw_items = scrape_all()
    print(f"scraped items: {len(raw_items)}")
    processed = [normalize_record(r) for r in raw_items if r.get("uid")]

    new_df = pd.DataFrame([
        {
            "uid":     r.get("uid"),
            "title":   r.get("title", ""),
            "summary": r.get("summary", ""),
            "lang":    r.get("lang", "en"),
            "source":  r.get("source", ""),
            "date":    r.get("date") or datetime.utcnow().isoformat(),
            "url":     r.get("url", ""),
        }
        for r in processed if r.get("uid")
    ])

    # create text_for_embed = title + ". " + summary
    new_df["text_for_embed"] = (
        new_df["title"].fillna("").str.strip() + ". " +
        new_df["summary"].fillna("").str.strip()
    ).str.strip()

    # 2) MERGE with previous dataset
    old_df = load_previous_df()
    merged = dedup_merge(old_df, new_df)
    print(f"merged rows total: {len(merged)} (old={len(old_df)}, new={len(new_df)})")

    # 3) Save parquet locally (for backup / local inspection)
    os.makedirs("data", exist_ok=True)
    merged.to_parquet(ART_PARQUET, index=False)

    # 4) Push to HF as (text-only) dataset split
    ds = Dataset.from_pandas(merged, preserve_index=False)
    ds.push_to_hub(HF_DATASET_ID, private=True)

    print(f"[{ist_now()}] refresh complete")


if __name__ == "__main__":
    main()
