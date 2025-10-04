import os, json
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify
from datasets import load_dataset

HF_DATASET_ID = os.getenv("HF_DATASET_ID", "mahi1010/news_articles_daily_m")
# For private datasets you must have HF_TOKEN in your environment when you run locally.

app = Flask(__name__)
_ds = None
_df = None

def _load_df():
    """Lazy-load the HF dataset into a pandas DataFrame (UTC dates)."""
    global _ds, _df
    if _df is not None:
        return _df
    _ds = load_dataset(HF_DATASET_ID, split="train")  # reads HF_TOKEN from env if needed
    _df = _ds.to_pandas()
    _df["date"] = pd.to_datetime(_df["date"], errors="coerce", utc=True)
    _df = _df.sort_values("date", ascending=False)
    return _df

def _to_ist(dt_utc):
    if pd.isna(dt_utc):
        return None
    return dt_utc.tz_convert("Asia/Kolkata").isoformat()

@app.get("/health")
def health():
    df = _load_df()
    latest = df["date"].max()
    return jsonify({
        "ok": True,
        "rows": int(len(df)),
        "latest_utc": latest.isoformat() if pd.notna(latest) else None,
        "latest_ist": _to_ist(latest),
        "dataset": HF_DATASET_ID
    })

@app.get("/articles")
def articles():
    """
    Simple keyword filter over title/summary with optional source/lang/since.
    Examples:
      /articles?limit=10
      /articles?q=kerala&source=The%20Hindu&limit=5
      /articles?since=2025-10-01
    """
    df = _load_df().copy()
    q      = request.args.get("q", "").strip()
    source = request.args.get("source", "").strip()
    lang   = request.args.get("lang", "").strip()
    since  = request.args.get("since", "").strip()
    limit  = int(request.args.get("limit", "20"))

    if q:
        mask = df["title"].fillna("").str.contains(q, case=False, na=False) | \
               df["summary"].fillna("").str.contains(q, case=False, na=False)
        df = df[mask]
    if source:
        df = df[df["source"].fillna("") == source]
    if lang:
        df = df[df["lang"].fillna("") == lang]
    if since:
        # interpret as date in any common format; treat as UTC midnight
        dt = pd.to_datetime(since, errors="coerce", utc=True)
        if pd.notna(dt):
            df = df[df["date"] >= dt]

    # build response
    out = []
    for _, r in df.head(limit).iterrows():
        out.append({
            "uid": r.get("uid"),
            "title": r.get("title"),
            "summary": r.get("summary"),
            "lang": r.get("lang"),
            "source": r.get("source"),
            "date_utc": r.get("date").isoformat() if pd.notna(r.get("date")) else None,
            "date_ist": _to_ist(r.get("date")),
            "url": r.get("url")
        })
    return jsonify({"count": len(out), "items": out})

if __name__ == "__main__":
    # local dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
