import os, json, faiss, numpy as np

def build_faiss(embeddings: np.ndarray, out_index="data/faiss.index", meta_path="data/meta.json"):
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D [N, D]")
    d = embeddings.shape[1]
    idx = faiss.IndexFlatIP(d)  # inner product on L2-normalized vectors equals cosine similarity
    if embeddings.shape[0] > 0:
        idx.add(embeddings.astype(np.float32))
    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    faiss.write_index(idx, out_index)
    with open(meta_path, "w") as f:
        json.dump({"dim": int(d), "count": int(embeddings.shape[0])}, f)
