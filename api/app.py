# fact-refresh/api/app.py

import os, json
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify
from datasets import load_dataset

# NEW imports for retrieval
import json as _json
import numpy as np
import requests, faiss, torch
from transformers import AutoTokenizer, AutoModel

HF_DATASET_ID = os.getenv("HF_DATASET_ID", "mahi1010/news_articles_daily_m")
E5_NAME = os.getenv("E5_NAME", "intfloat/multilingual-e5-base")  # embedding model

# For private datasets you must have HF_TOKEN in your environment when you run (Actions or local).

app = Flask(__name__)
_ds = None
_df = None

# Caches
_FAISS = None
_META = None
_E5_TOK = None
_E5_MOD = None

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

# ---------- Retrieval helpers ----------
def _load_faiss_and_meta():
    """Download faiss.index and meta.json from HF once; keep in memory."""
    global _FAISS, _META
    if _FAISS is not None and _META is not None:
        return _FAISS, _META
    base = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main"
    fa = requests.get(f"{base}/faiss.index").content
    me = requests.get(f"{base}/meta.json").text
    os.makedirs("/tmp", exist_ok=True)
    open("/tmp/faiss.index", "wb").write(fa)
    open("/tmp/meta.json", "w").write(me)
    _FAISS = faiss.read_index("/tmp/faiss.index")
    _META = _json.loads(me)
    return _FAISS, _META

def _load_e5():
    """Lazy-load the E5 model for query embeddings."""
    global _E5_TOK, _E5_MOD
    if _E5_TOK is not None and _E5_MOD is not None:
        return _E5_TOK, _E5_MOD
    _E5_TOK = AutoTokenizer.from_pretrained(E5_NAME)
    _E5_MOD = AutoModel.from_pretrained(E5_NAME)
    _E5_MOD.eval()
    return _E5_TOK, _E5_MOD

def _embed_query(q: str) -> np.ndarray:
    tok, mod = _load_e5()
    with torch.no_grad():
        t = tok([f"query: {q}"], return_tensors="pt", truncation=True)
        v = mod(**t).last_hidden_state[:, 0].numpy()
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v.astype(np.float32)

# ---------- Routes ----------
@app.get("/")
def home():
    return {
        "ok": True,
        "message": "Flask is running. Try /health or /articles?limit=5 or /verify?q=your+claim&k=5",
        "endpoints": ["/health", "/articles", "/verify"]
    }

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
        dt = pd.to_datetime(since, errors="coerce", utc=True)
        if pd.notna(dt):
            df = df[df["date"] >= dt]

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

@app.get("/verify")
def verify():
    """
    Use:  /verify?q=<claim>&k=5
    Returns: top-k evidence with similarity scores (higher is closer).
    """
    claim = request.args.get("q", "").strip()
    k = int(request.args.get("k", "5"))
    if not claim:
        return jsonify({"ok": False, "error": "missing q parameter"}), 400

    df = _load_df()
    idx, meta = _load_faiss_and_meta()
    qv = _embed_query(claim)
    D, I = idx.search(qv, k)

    out = []
    for rank, i in enumerate(I[0]):
        if i < 0 or i >= len(df):
            continue
        row = df.iloc[i]
        out.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "title": row.get("title"),
            "summary": row.get("summary"),
            "source": row.get("source"),
            "date_utc": row.get("date").isoformat() if pd.notna(row.get("date")) else None,
            "date_ist": _to_ist(row.get("date")),
            "url": row.get("url"),
        })
    return jsonify({"ok": True, "claim": claim, "k": k, "results": out})

if __name__ == "__main__":
    # local dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
