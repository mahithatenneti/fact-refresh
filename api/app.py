import os, json, pandas as pd, numpy as np, requests, faiss, torch
from datetime import datetime
from flask import Flask, request, jsonify
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

HF_DATASET_ID = os.getenv("HF_DATASET_ID", "mahi1010/news_articles_daily_m")
E5_NAME = os.getenv("E5_NAME", "intfloat/multilingual-e5-base")

app = Flask(__name__)

# caches
_df = _FAISS = _META = _E5_TOK = _E5_MOD = None


# -------- dataset loader --------
def _load_df():
    global _df
    if _df is not None:
        return _df
    ds = load_dataset(HF_DATASET_ID, split="train")
    df = ds.to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    _df = df.sort_values("date", ascending=False)
    return _df


def _to_ist(dt):
    if pd.isna(dt): return None
    return dt.tz_convert("Asia/Kolkata").isoformat()


# -------- FAISS & E5 setup --------
def _load_faiss_and_meta():
    global _FAISS, _META
    if _FAISS is not None and _META is not None:
        return _FAISS, _META
    base = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main"
    os.makedirs("/tmp", exist_ok=True)
    faiss_path, meta_path = "/tmp/faiss.index", "/tmp/meta.json"
    open(faiss_path, "wb").write(requests.get(f"{base}/faiss.index").content)
    open(meta_path, "w").write(requests.get(f"{base}/meta.json").text)
    _FAISS = faiss.read_index(faiss_path)
    _META = json.load(open(meta_path))
    return _FAISS, _META


def _load_e5():
    global _E5_TOK, _E5_MOD
    if _E5_TOK and _E5_MOD:
        return _E5_TOK, _E5_MOD
    _E5_TOK = AutoTokenizer.from_pretrained(E5_NAME)
    _E5_MOD = AutoModel.from_pretrained(E5_NAME)
    _E5_MOD.eval()
    return _E5_TOK, _E5_MOD


def _embed_query(q):
    tok, mod = _load_e5()
    with torch.no_grad():
        t = tok([f"query: {q}"], return_tensors="pt", truncation=True)
        v = mod(**t).last_hidden_state[:, 0].numpy()
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v.astype(np.float32)


# -------- routes --------
@app.get("/")
def home():
    return {
        "ok": True,
        "message": "Server alive ✅  Try /health, /articles?limit=5, or /verify?q=your+claim&k=5",
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
    df = _load_df().copy()
    q = request.args.get("q", "").strip()
    src = request.args.get("source", "").strip()
    lang = request.args.get("lang", "").strip()
    since = request.args.get("since", "").strip()
    limit = int(request.args.get("limit", "20"))

    if q:
        mask = df["title"].fillna("").str.contains(q, case=False, na=False) | \
               df["summary"].fillna("").str.contains(q, case=False, na=False)
        df = df[mask]
    if src:
        df = df[df["source"].fillna("") == src]
    if lang:
        df = df[df["lang"].fillna("") == lang]
    if since:
        dt = pd.to_datetime(since, errors="coerce", utc=True)
        if pd.notna(dt):
            df = df[df["date"] >= dt]

    items = [{
        "uid": r.uid,
        "title": r.title,
        "summary": r.summary,
        "source": r.source,
        "lang": r.lang,
        "date_utc": r.date.isoformat() if pd.notna(r.date) else None,
        "date_ist": _to_ist(r.date),
        "url": r.url
    } for _, r in df.head(limit).iterrows()]

    return jsonify({"count": len(items), "items": items})


@app.get("/verify")
def verify():
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", "5"))
    if not q:
        return jsonify({"ok": False, "error": "missing q parameter"}), 400

    df = _load_df()
    idx, _ = _load_faiss_and_meta()
    qv = _embed_query(q)
    D, I = idx.search(qv, k)

    results = []
    for rank, i in enumerate(I[0]):
        if i < 0 or i >= len(df): continue
        r = df.iloc[i]
        results.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "title": r.title,
            "summary": r.summary,
            "source": r.source,
            "date_utc": r.date.isoformat() if pd.notna(r.date) else None,
            "date_ist": _to_ist(r.date),
            "url": r.url
        })
    return jsonify({"ok": True, "claim": q, "k": k, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
