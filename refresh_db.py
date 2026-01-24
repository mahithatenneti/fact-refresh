import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo
from datetime import datetime, timezone
from dateutil import tz

from scraper.sources import scrape_all
from scraper.preprocess import normalize_record

HF_DATASET_ID = "mahi1010/news_articles_daily_m"

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
            "uid","title","summary","lang","source","date","url","text_for_embed",
            "published_at","scraped_at"
        ])

def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Backward compatibility: if only "date" exists, treat it as published_at
    if "published_at" not in df.columns:
        df["published_at"] = df.get("date", None)

    # scraped_at may not exist for old rows
    if "scraped_at" not in df.columns:
        df["scraped_at"] = None

    # parse both
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["scraped_at"]   = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

    # keep "date" column synced for older notebooks
    df["date"] = df["published_at"]

    return df

def dedup_merge(old_df, new_df):
    all_df = pd.concat([old_df, new_df], ignore_index=True)

    all_df = standardize_dates(all_df)

    # Use best timestamp for sorting:
    # prefer published_at; if missing use scraped_at
    all_df["sort_ts"] = all_df["published_at"]
    all_df.loc[all_df["sort_ts"].isna(), "sort_ts"] = all_df.loc[all_df["sort_ts"].isna(), "scraped_at"]

    all_df = all_df.sort_values("sort_ts", ascending=False)

    # keep newest per uid
    all_df = all_df.drop_duplicates(subset=["uid"], keep="first")

    # cleanup
    for c in ["title","summary","lang","source","url","text_for_embed"]:
        if c not in all_df.columns:
            all_df[c] = ""
    all_df["summary"] = all_df["summary"].fillna("")

    # drop helper
    all_df = all_df.drop(columns=["sort_ts"], errors="ignore")

    return all_df.reset_index(drop=True)

def main():
    print(f"[{ist_now()}] Starting refresh")
    ensure_hf_repo()

    # ---------- SCRAPE ----------
    raw = scrape_all()
    print(f"[INFO] scraped: {len(raw)}")

    processed = [normalize_record(r) for r in raw if r.get("uid")]

    now_utc = datetime.now(timezone.utc).isoformat()

    new_df = pd.DataFrame([{
        "uid": r.get("uid"),
        "title": r.get("title",""),
        "summary": r.get("summary",""),
        "lang": r.get("lang","en"),
        "source": r.get("source",""),
        "url": r.get("url",""),
        "text_for_embed": r.get("text_for_embed", None),

        # keep old column name for notebook compatibility
        raw_date = r.get("date")
        safe_date = raw_date if (raw_date is not None and str(raw_date).strip() != "") else now_utc
        
        "date": safe_date,                 # for backward compatibility
        "published_at": safe_date,         # published_at never null now
        "scraped_at": now_utc,

    } for r in processed])

    # parse new dates immediately
    new_df = standardize_dates(new_df)

    # ---------- LOAD OLD + MERGE ----------
    old_df = load_previous_df()
    merged = dedup_merge(old_df, new_df)
    print(f"[INFO] final merged rows: {len(merged)}")

    # ---------- QUICK DEBUG ----------
    print("[DEBUG] merged published_at max:", merged["published_at"].max())
    print("[DEBUG] merged scraped_at max:", merged["scraped_at"].max())
    print("[DEBUG] merged null published_at:", merged["published_at"].isna().sum())

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
