import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1) Load instruction embeddings + JSONL
inst_embs = np.load("rag_instructions_embeddings.npy")    # (M, D)
inst_docs = []
with open("rag_instructions.jsonl", "r", encoding="utf-8") as f:
    for ln in f:
        inst_docs.append(json.loads(ln)["text"])
assert len(inst_docs) == inst_embs.shape[0]

# 2) Load roads embeddings + JSONL
roads_embs = np.load("rag_roads_embeddings.npy")          # (N, D)
roads_docs = []
with open("rag_roads.jsonl", "r", encoding="utf-8") as f:
    for ln in f:
        roads_docs.append(json.loads(ln)["text"])
assert len(roads_docs) == roads_embs.shape[0]

# 3) Init the encoder
embedder = SentenceTransformer("sentence-transformers/LaBSE")

def top1(query: str, embs: np.ndarray, docs: list[str]):
    q_emb = embedder.encode([query], convert_to_numpy=True)  # (1, D)
    sims = (embs @ q_emb.T).squeeze(1)                        # (K,)
    idx = int(np.argmax(sims))
    return idx, float(sims[idx]), docs[idx]

if __name__ == "__main__":
    q = "Kako se imenuje južna veja ljubljanske obvoznice?"  # sample query

    # retrieve top-1 from instructions
    i_idx, i_score, i_text = top1(q, inst_embs, inst_docs)
    print("→ Instruction hit:")
    print(f"  [{i_idx}] (score={i_score:.4f}) {i_text}\n")

    # retrieve top-1 from roads
    r_idx, r_score, r_text = top1(q, roads_embs, roads_docs)
    print("→ Road-segment hit:")
    print(f"  [{r_idx}] (score={r_score:.4f}) {r_text}")