import os, pandas as pd, numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfFolder, create_repo, upload_file
from datetime import datetime
from dateutil import tz

from scraper.sources import scrape_all
from scraper.preprocess import normalize_record
from embed.e5_embed import embed_passages
from embed.faiss_utils import build_faiss

# ====== CONFIG (env or edit here) ======
HF_TOKEN = os.getenv("HF_TOKEN")                # set in GitHub Actions secrets
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "your-username/news_articles_daily")
ART_PARQUET = "data/news_articles.parquet"
FAISS_PATH  = "data/faiss.index"
META_PATH   = "data/meta.json"

def ist_now():
    utc = datetime.utcnow().replace(tzinfo=tz.tzutc())
    ist = utc.astimezone(tz.gettz("Asia/Kolkata"))
    return ist.strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_hf_repo():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set (GitHub Actions secret)")
    HfFolder.save_token(HF_TOKEN)
    create_repo(HF_DATASET_ID, repo_type="dataset", exist_ok=True)

def load_previous_df():
    try:
        ds = load_dataset(HF_DATASET_ID, split="train")
        return ds.to_pandas()
    except Exception:
        return pd.DataFrame(columns=["uid","title","summary","lang","source","date","url"])

def dedup_merge(old_df, new_df):
    all_df = pd.concat([old_df, new_df], ignore_index=True)
    # keep the newest per uid
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df = all_df.sort_values("date", ascending=False, na_position="last")
    all_df = all_df.drop_duplicates(subset=["uid"], keep="first")
    # enforce dtypes / fill
    for c in ["title","summary","lang","source","url"]:
        all_df[c] = all_df[c].fillna("")
    return all_df

def main():
    print(f"[{ist_now()}] start refresh")
    ensure_hf_repo()

    # 1) SCRAPE
    raw_items = scrape_all()
    print(f"scraped items: {len(raw_items)}")
    processed = [normalize_record(r) for r in raw_items if r.get('uid')]

    new_df = pd.DataFrame([{
        "uid": r.get("uid"),
        "title": r.get("title",""),
        "summary": r.get("summary",""),
        "lang": r.get("lang","en"),
        "source": r.get("source",""),
        "date": r.get("date") or datetime.utcnow().isoformat(),
        "url": r.get("url",""),
    } for r in processed if r.get("uid")])

    # 2) MERGE with previous dataset
    old_df = load_previous_df()
    merged = dedup_merge(old_df, new_df)
    print(f"merged rows total: {len(merged)} (old={len(old_df)}, new={len(new_df)})")

    # 3) EMBEDDINGS + FAISS (only for rows with summaries)
    ok = merged["summary"].fillna("") != ""
    summaries = merged.loc[ok, "summary"].tolist()
    if summaries:
        emb = embed_passages(summaries)
        all_emb = np.zeros((len(merged), emb.shape[1]), dtype=np.float32)
        all_emb[np.where(ok)[0]] = emb
    else:
        # No summaries yet
        all_emb = np.zeros((len(merged), 768), dtype=np.float32)

    build_faiss(all_emb, out_index=FAISS_PATH, meta_path=META_PATH)

    # 4) Save parquet locally
    os.makedirs("data", exist_ok=True)
    merged.to_parquet(ART_PARQUET, index=False)

    # 5) Push to HF as dataset split
    ds = Dataset.from_pandas(merged, preserve_index=False)
    ds.push_to_hub(HF_DATASET_ID, private=False)

    # 6) Upload FAISS + meta as files alongside dataset
    upload_file(path_or_fileobj=FAISS_PATH, path_in_repo="faiss.index",
                repo_id=HF_DATASET_ID, repo_type="dataset")
    upload_file(path_or_fileobj=META_PATH, path_in_repo="meta.json",
                repo_id=HF_DATASET_ID, repo_type="dataset")

    print(f"[{ist_now()}] refresh complete")

if __name__ == "__main__":
    main()
