import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfFolder, create_repo
from datetime import datetime
from dateutil import tz

from scraper.sources import scrape_all
from scraper.preprocess import normalize_record

# ====== CONFIG ======
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
    create_repo(HF_DATASET_ID, repo_type="dataset", exist_ok=True, private=True)


def main():
    print(f"[{ist_now()}] start refresh")
    ensure_hf_repo()

    # 1) SCRAPE
    raw_items = scrape_all()
    print(f"scraped items: {len(raw_items)}")

    processed = [normalize_record(r) for r in raw_items if r.get("uid")]

    df = pd.DataFrame([
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

    # text_for_embed = title + ". " + summary  (used later in Colab)
    df["text_for_embed"] = (
        df["title"].fillna("").str.strip() + ". " +
        df["summary"].fillna("").str.strip()
    ).str.strip()

    # 2) Save parquet locally
    os.makedirs("data", exist_ok=True)
    df.to_parquet(ART_PARQUET, index=False)

    # 3) Push *only this fresh batch* to HF
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(HF_DATASET_ID, private=True)

    print(f"[{ist_now()}] refresh complete | rows={len(df)}")


if __name__ == "__main__":
    main()
