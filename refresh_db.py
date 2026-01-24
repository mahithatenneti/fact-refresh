import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo
from datetime import datetime, timezone
from dateutil import tz

from scraper.sources import scrape_all
from scraper.preprocess import normalize_record

# ========= SETTINGS ============
HF_DATASET_ID = "mahi1010/news_articles_daily_m"

# -------------------------------
# TIME HELPERS
# -------------------------------
def ist_now():
    utc = datetime.utcnow().replace(tzinfo=tz.tzutc())
    ist = utc.astimezone(tz.gettz("Asia/Kolkata"))
    return ist.strftime("%Y-%m-%d %H:%M:%S %Z")

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

# -------------------------------
# HF REPO
# -------------------------------
def ensure_hf_repo():
    # Keeping private=True as in your original script
    create_repo(HF_DATASET_ID, repo_type="dataset", exist_ok=True, private=True)

# -------------------------------
# LOAD PREVIOUS DATA
# -------------------------------
def load_previous_df():
    try:
        ds = load_dataset(HF_DATASET_ID, split="train")
        df = ds.to_pandas()
        print(f"[INFO] Loaded previous: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[WARN] No previous dataset found ({e})")
        # include both old + new schema columns so merges never break
        return pd.DataFrame(columns=[
            "uid", "title", "summary", "lang", "source", "date", "url", "text_for_embed",
            "published_at", "scraped_at"
        ])

# -------------------------------
# STANDARDIZE / BACKWARD COMPATIBILITY
# -------------------------------
def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures these columns exist and are tz-aware datetime:
      - published_at: best estimate of article publish time (fallback-safe)
      - scraped_at  : time when our scraper saw it
    Also keeps `date` synced to published_at for older notebooks.
    """

    # Create missing columns safely
    if "published_at" not in df.columns:
        # if legacy dataset only has `date`, treat it as published_at
        df["published_at"] = df["date"] if "date" in df.columns else None

    if "scraped_at" not in df.columns:
        df["scraped_at"] = None

    if "date" not in df.columns:
        df["date"] = df["published_at"]

    # Parse
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["scraped_at"]   = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

    # Keep old `date` column synced for all notebooks that still read it
    df["date"] = df["published_at"]

    return df

# -------------------------------
# MERGE + DEDUP
# -------------------------------
def dedup_merge(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat([old_df, new_df], ignore_index=True)

    # Ensure consistent date schema
    all_df = standardize_dates(all_df)

    # Sort timestamp preference:
    # 1) published_at
    # 2) scraped_at
    all_df["sort_ts"] = all_df["published_at"]
    mask = all_df["sort_ts"].isna()
    all_df.loc[mask, "sort_ts"] = all_df.loc[mask, "scraped_at"]

    all_df = all_df.sort_values("sort_ts", ascending=False)

    # Keep newest per uid
    if "uid" in all_df.columns:
        all_df = all_df.drop_duplicates(subset=["uid"], keep="first")

    # Ensure required columns exist (avoid KeyError later)
    for c in ["uid", "title", "summary", "lang", "source", "url", "text_for_embed", "published_at", "scraped_at", "date"]:
        if c not in all_df.columns:
            all_df[c] = "" if c not in ["published_at", "scraped_at", "date"] else pd.NaT

    # Cleanup
    all_df["title"]   = all_df["title"].fillna("").astype(str)
    all_df["summary"] = all_df["summary"].fillna("").astype(str)
    all_df["lang"]    = all_df["lang"].fillna("en").astype(str)
    all_df["source"]  = all_df["source"].fillna("").astype(str)
    all_df["url"]     = all_df["url"].fillna("").astype(str)

    # Drop helper
    all_df = all_df.drop(columns=["sort_ts"], errors="ignore")

    return all_df.reset_index(drop=True)

# -------------------------------
# MAIN
# -------------------------------
def main():
    print(f"[{ist_now()}] Starting refresh")
    ensure_hf_repo()

    # ---------- SCRAPE ----------
    raw = scrape_all()
    print(f"[INFO] scraped: {len(raw)}")

    processed = [normalize_record(r) for r in raw if r.get("uid")]

    scrape_ts = now_utc_iso()

    # Build new_df with safe fallback for dates
    rows = []
    for r in processed:
        raw_date = r.get("date")

        # SAFE FALLBACK:
        # If the source parser doesn't provide a date, we still store it as "now"
        # so that the dataset continues to show latest items and doesn't become NaT-heavy.
        safe_date = raw_date if (raw_date is not None and str(raw_date).strip() != "") else scrape_ts

        rows.append({
            "uid": r.get("uid"),
            "title": r.get("title", ""),
            "summary": r.get("summary", ""),
            "lang": r.get("lang", "en"),
            "source": r.get("source", ""),
            "url": r.get("url", ""),
            "text_for_embed": r.get("text_for_embed", None),

            # legacy column for older notebooks
            "date": safe_date,

            # stable fields
            "published_at": safe_date,
            "scraped_at": scrape_ts,
        })

    new_df = pd.DataFrame(rows)

    # Parse immediately (important)
    new_df = standardize_dates(new_df)

    # ---------- LOAD OLD + MERGE ----------
    old_df = load_previous_df()
    merged = dedup_merge(old_df, new_df)
    print(f"[INFO] final merged rows: {len(merged)}")

    # ---------- DEBUG ----------
    print("[DEBUG] merged published_at max:", merged["published_at"].max())
    print("[DEBUG] merged scraped_at max  :", merged["scraped_at"].max())
    print("[DEBUG] merged null published_at:", merged["published_at"].isna().sum())
    print("[DEBUG] merged null scraped_at  :", merged["scraped_at"].isna().sum())

    # ---------- SAVE TO HF ----------
    ds = Dataset.from_pandas(merged, preserve_index=False)
    ds.push_to_hub(
        HF_DATASET_ID,
        private=True,
        split="train",
        max_shard_size="500MB"
    )

    print(f"[{ist_now()}] Refresh complete!")

if __name__ == "__main__":
    main()
