import os, torch, numpy as np
from transformers import AutoTokenizer, AutoModel

E5_NAME = os.getenv("E5_NAME", "intfloat/e5-base")  # swap to intfloat/multilingual-e5-base for multilingual
_tok = None
_mod = None

def _load():
    global _tok, _mod
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(E5_NAME)
        _mod = AutoModel.from_pretrained(E5_NAME)
        _mod.eval()

def _norm(a: np.ndarray):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return (a / n).astype(np.float32)

def embed_passages(texts):
    _load()
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    with torch.no_grad():
        toks = _tok([f"passage: {t}" for t in texts], padding=True, truncation=True, return_tensors="pt")
        out  = _mod(**toks).last_hidden_state[:,0]  # CLS
        return _norm(out.numpy())

def embed_queries(texts):
    _load()
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    with torch.no_grad():
        toks = _tok([f"query: {t}" for t in texts], padding=True, truncation=True, return_tensors="pt")
        out  = _mod(**toks).last_hidden_state[:,0]
        return _norm(out.numpy())
