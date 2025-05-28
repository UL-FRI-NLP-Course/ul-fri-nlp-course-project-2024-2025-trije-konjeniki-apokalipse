import numpy as np
from sentence_transformers import SentenceTransformer

# 1) load precomputed embeddings + docs
embs = np.load("doc_embeddings.npy")      # (N, D)
with open("docs.txt","r",encoding="utf-8") as f:
    docs = [ln.strip() for ln in f]
assert len(docs)==embs.shape[0]

# 2) init embedder
embedder = SentenceTransformer("sentence-transformers/LaBSE")

def topk(query: str, k: int = 3):
    # a) embed query
    q_emb = embedder.encode([query], convert_to_numpy=True)  # (1, D)

    # b) compute similarity (dot product)
    sims = (embs @ q_emb.T).squeeze(1)   # shape (N,)

    # c) pick top-k indices
    idxs = np.argsort(-sims)[:k]
    return [(i, float(sims[i]), docs[i]) for i in idxs]

if __name__=="__main__":
    q = "Ali je primorska avtocesta zaprta zaradi okvare?"
    for rank, (idx, score, text) in enumerate(topk(q, 3), 1):
        print(f"{rank}. doc#{idx}  score={score:.4f}\n   ▶ {text}\n")