import os
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo
from datetime import datetime
from dateutil import tz

from scraper.sources import scrape_all
from scraper.preprocess import normalize_record

# ========= SETTINGS ============
HF_DATASET_ID = "mahi1010/news_articles_daily_m"

# --------------------------------
# TIME HELPERS
# --------------------------------
def ist_now():
    utc = datetime.utcnow().replace(tzinfo=tz.tzutc())
    ist = utc.astimezone(tz.gettz("Asia/Kolkata"))
    return ist.strftime("%Y-%m-%d %H:%M:%S %Z")

def utc_now_iso():
    # timezone-aware ISO string
    return datetime.utcnow().replace(tzinfo=tz.tzutc()).isoformat()

# --------------------------------
# HF HELPERS
# --------------------------------
def ensure_hf_repo():
    # keep your existing visibility; change private=True/False as you want
    create_repo(HF_DATASET_ID, repo_type="dataset", exist_ok=True, private=True)

def load_previous_df():
    """
    Loads existing HF dataset if present.
    Keeps ALL historical rows.
    """
    try:
        ds = load_dataset(HF_DATASET_ID, split="train")
        df = ds.to_pandas()
        print(f"[INFO] Loaded previous: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[WARN] No previous dataset found ({e})")
        return pd.DataFrame(columns=[
            "uid", "title", "summary", "lang", "source", "published_at", "scraped_at", "url", "text_for_embed"
        ])

# --------------------------------
# NORMALIZATION + MERGE
# --------------------------------
CANON_COLS = [
    "uid", "title", "summary", "lang", "source",
    "published_at", "scraped_at", "url", "text_for_embed"
]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[CANON_COLS]

def _parse_dt(series):
    # always parse to UTC aware timestamps; invalid -> NaT
    return pd.to_datetime(series, errors="coerce", utc=True)

def dedup_merge(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Goal:
    - Preserve ALL historical data
    - Add a reliable ingestion time: scraped_at
    - Keep published time separately: published_at
    - Dedup by uid, keeping the newest record:
        primary: scraped_at
        fallback: published_at
    - Fix the common bug: if published dates are stale/broken, dataset appears "old"
    """

    old_df = old_df.copy()
    new_df = new_df.copy()

    # Backward compatibility: if older dataset used "date", treat it as published_at
    if "published_at" not in old_df.columns:
        if "date" in old_df.columns:
            old_df["published_at"] = old_df["date"]
        else:
            old_df["published_at"] = ""
    if "scraped_at" not in old_df.columns:
        old_df["scraped_at"] = ""

    # If old dataset has "date" only, keep it (won't be pushed) but safe
    # Ensure all required cols exist
    old_df = _ensure_cols(old_df)

    # Ensure new_df has correct cols
    # We expect new_df to come with published_at and scraped_at already set below
    new_df = _ensure_cols(new_df)

    all_df = pd.concat([old_df, new_df], ignore_index=True)

    # Parse times
    all_df["published_at_parsed"] = _parse_dt(all_df["published_at"])
    all_df["scraped_at_parsed"]   = _parse_dt(all_df["scraped_at"])

    # Choose "freshness" time for sorting: scraped_at preferred, fallback to published_at
    all_df["fresh_time"] = all_df["scraped_at_parsed"].where(
        all_df["scraped_at_parsed"].notna(),
        all_df["published_at_parsed"]
    )

    # If even both are NaT (rare), push them to the end safely
    # by filling with very old timestamp
    all_df["fresh_time"] = all_df["fresh_time"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))

    # Sort newest first and dedup by uid
    all_df = all_df.sort_values("fresh_time", ascending=False)
    all_df = all_df.drop_duplicates(subset=["uid"], keep="first")

    # Cleanup
    all_df["title"] = all_df["title"].fillna("").astype(str)
    all_df["summary"] = all_df["summary"].fillna("").astype(str)
    all_df["source"] = all_df["source"].fillna("").astype(str)
    all_df["lang"] = all_df["lang"].fillna("en").astype(str)
    all_df["url"] = all_df["url"].fillna("").astype(str)
    all_df["text_for_embed"] = all_df["text_for_embed"].fillna("").astype(str)

    # Drop helper cols
    all_df = all_df.drop(columns=["published_at_parsed", "scraped_at_parsed", "fresh_time"], errors="ignore")

    return all_df.reset_index(drop=True)

# --------------------------------
# MAIN
# --------------------------------
def main():
    print(f"[{ist_now()}] Starting refresh")

    ensure_hf_repo()

    # ---------- SCRAPE ----------
    raw = scrape_all()
    print(f"[INFO] scraped: {len(raw)}")

    processed = [normalize_record(r) for r in raw if r.get("uid")]

    now_iso = utc_now_iso()

    # Build NEW dataframe:
    # - published_at: from source date (if any)
    # - scraped_at: now (always)
    # Keep "date" OUT of the final schema to avoid confusion
    new_rows = []
    for r in processed:
        new_rows.append({
            "uid": r.get("uid", ""),
            "title": r.get("title", "") or "",
            "summary": r.get("summary", "") or "",
            "lang": r.get("lang", "en") or "en",
            "source": r.get("source", "") or "",
            "published_at": r.get("date") or "",   # source-provided publish time
            "scraped_at": now_iso,                 # ingestion time (reliable)
            "url": r.get("url", "") or "",
            "text_for_embed": r.get("text_for_embed", "") or "",
        })

    new_df = pd.DataFrame(new_rows)
    new_df = _ensure_cols(new_df)

    # ---------- LOAD OLD + MERGE ----------
    old_df = load_previous_df()
    merged = dedup_merge(old_df, new_df)
    print(f"[INFO] final merged rows: {len(merged)}")

    # Quick sanity debug
    pub_dt = pd.to_datetime(merged["published_at"], errors="coerce", utc=True)
    scr_dt = pd.to_datetime(merged["scraped_at"], errors="coerce", utc=True)
    print(f"[DEBUG] published_at: null={pub_dt.isna().sum()}  min={pub_dt.min()}  max={pub_dt.max()}")
    print(f"[DEBUG] scraped_at:   null={scr_dt.isna().sum()}  min={scr_dt.min()}  max={scr_dt.max()}")

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
