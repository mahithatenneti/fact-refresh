# fact-refresh/api/app.py

import os
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify
from datasets import load_dataset

# --- retrieval / embeddings deps ---
import json as _json
import numpy as np
import requests
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# === Config ===
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "mahi1010/news_articles_daily_m")
E5_NAME = os.getenv("E5_NAME", "intfloat/multilingual-e5-base")  # embedding model
NLI_NAME = os.getenv(
    "NLI_NAME",
    "MoritzLaurer/deberta-v3-base-mnli-fever-anli-ling-binary"  # multilingual compact NLI
)
# If you prefer classic English-only, set NLI_NAME="facebook/bart-large-mnli"

# For private datasets you must have HF_TOKEN in the environment.

# (Helps reduce CPU usage on small runners)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

app = Flask(__name__)

# Caches
_ds = None
_df = None
_FAISS = None
_META = None
_E5_TOK = None
_E5_MOD = None
_NLI_TOK = None
_NLI_MOD = None


# ---------- Data loading ----------
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
    os.makedirs("/tmp", exist_ok=True)

    # small timeouts to avoid hanging
    fa_bytes = requests.get(f"{base}/faiss.index", timeout=60).content
    meta_txt = requests.get(f"{base}/meta.json", timeout=30).text

    with open("/tmp/faiss.index", "wb") as f:
        f.write(fa_bytes)
    with open("/tmp/meta.json", "w") as f:
        f.write(meta_txt)

    _FAISS = faiss.read_index("/tmp/faiss.index")
    _META = _json.loads(meta_txt)
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


# ---------- NLI (verdict) helpers ----------
def _load_nli():
    global _NLI_TOK, _NLI_MOD
    if _NLI_TOK is not None and _NLI_MOD is not None:
        return _NLI_TOK, _NLI_MOD
    _NLI_TOK = AutoTokenizer.from_pretrained(NLI_NAME)
    _NLI_MOD = AutoModelForSequenceClassification.from_pretrained(NLI_NAME)
    _NLI_MOD.eval()
    return _NLI_TOK, _NLI_MOD


def _nli_verdict(claim: str, evidence: str):
    """
    Returns: (label, conf, dist) where label in {"SUPPORTS","REFUTES","NEI"}.
    We map typical MNLI label order (entailment, neutral, contradiction) to these 3.
    """
    tok, mod = _load_nli()
    with torch.no_grad():
        enc = tok(claim, evidence, return_tensors="pt", truncation=True)
        logits = mod(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Assume order: entailment, neutral, contradiction
    if probs.shape[0] == 3:
        ent, neu, con = probs
        dist = {"SUPPORTS": float(ent), "NEI": float(neu), "REFUTES": float(con)}
    else:
        # Fallback if head is unusual: pick max as SUPPORTS
        m = float(probs.max())
        dist = {"SUPPORTS": m, "NEI": 0.0, "REFUTES": 0.0}
    label = max(dist, key=dist.get)
    conf = dist[label]
    return label, conf, dist


# ---------- Routes ----------
@app.get("/")
def home():
    return {
        "ok": True,
        "message": "Flask is running. Try /health, /articles?limit=5, /verify?q=claim&k=5, /verdict?q=claim&k=5",
        "endpoints": ["/health", "/articles", "/verify", "/verdict"],
    }


@app.get("/health")
def health():
    df = _load_df()
    latest = df["date"].max()
    faiss_ready = _FAISS is not None
    return jsonify(
        {
            "ok": True,
            "rows": int(len(df)),
            "latest_utc": latest.isoformat() if pd.notna(latest) else None,
            "latest_ist": _to_ist(latest),
            "dataset": HF_DATASET_ID,
            "faiss_loaded": faiss_ready,
            "models": {"e5": E5_NAME, "nli": NLI_NAME},
        }
    )


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
    q = request.args.get("q", "").strip()
    source = request.args.get("source", "").strip()
    lang = request.args.get("lang", "").strip()
    since = request.args.get("since", "").strip()

    try:
        limit = int(request.args.get("limit", "20"))
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 100))

    if q:
        mask = df["title"].fillna("").str.contains(q, case=False, na=False) | df[
            "summary"
        ].fillna("").str.contains(q, case=False, na=False)
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
        out.append(
            {
                "uid": r.get("uid"),
                "title": r.get("title"),
                "summary": r.get("summary"),
                "lang": r.get("lang"),
                "source": r.get("source"),
                "date_utc": r.get("date").isoformat() if pd.notna(r.get("date")) else None,
                "date_ist": _to_ist(r.get("date")),
                "url": r.get("url"),
            }
        )
    return jsonify({"count": len(out), "items": out})


@app.get("/verify")
def verify():
    """
    Use:  /verify?q=<claim>&k=5
    Returns: top-k evidence with similarity scores (higher is closer).
    """
    claim = request.args.get("q", "").strip()
    if not claim:
        return jsonify({"ok": False, "error": "missing q parameter"}), 400

    try:
        k = int(request.args.get("k", "5"))
    except ValueError:
        k = 5
    k = max(1, min(k, 20))

    df = _load_df()
    idx, _meta = _load_faiss_and_meta()
    qv = _embed_query(claim)
    D, I = idx.search(qv, k)

    out = []
    for rank, i in enumerate(I[0]):
        if i < 0 or i >= len(df):
            continue
        row = df.iloc[i]
        out.append(
            {
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "title": row.get("title"),
                "summary": row.get("summary"),
                "source": row.get("source"),
                "date_utc": row.get("date").isoformat() if pd.notna(row.get("date")) else None,
                "date_ist": _to_ist(row.get("date")),
                "url": row.get("url"),
            }
        )
    return jsonify({"ok": True, "claim": claim, "k": k, "results": out})


@app.get("/verdict")
def verdict():
    """
    Use: /verdict?q=<claim>&k=5
    1) retrieve top-k evidence with FAISS
    2) run NLI over (claim, each evidence)
    3) return best label + confidence and the evidence list
    """
    claim = request.args.get("q", "").strip()
    if not claim:
        return jsonify({"ok": False, "error": "missing q parameter"}), 400

    try:
        k = int(request.args.get("k", "5"))
    except ValueError:
        k = 5
    k = max(1, min(k, 10))

    # 1) retrieve evidence
    df = _load_df()
    idx, _meta = _load_faiss_and_meta()
    qv = _embed_query(claim)
    D, I = idx.search(qv, k)

    evidences = []
    for i in I[0]:
        if 0 <= i < len(df):
            r = df.iloc[i]
            evidences.append({
                "title": r.get("title"),
                "summary": r.get("summary"),
                "source": r.get("source"),
                "date_utc": r.get("date").isoformat() if pd.notna(r.get("date")) else None,
                "date_ist": _to_ist(r.get("date")),
                "url": r.get("url")
            })

    if not evidences:
        return jsonify({"ok": True, "claim": claim, "label": "NEI", "confidence": 0.0, "evidence": []})

    # 2) NLI scoring
    scored = []
    for ev in evidences:
        text = ev["summary"] or ev["title"] or ""
        label, conf, dist = _nli_verdict(claim, text)
        scored.append({**ev, "label": label, "confidence": conf, "distribution": dist})

    # 3) pick strongest evidence as final label
    best = max(scored, key=lambda x: x["confidence"])
    final_label = best["label"]
    final_conf  = best["confidence"]

    return jsonify({
        "ok": True,
        "claim": claim,
        "label": final_label,
        "confidence": final_conf,
        "evidence_top": best,
        "evidence_list": scored
    })


if __name__ == "__main__":
    # local dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
