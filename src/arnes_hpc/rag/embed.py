# build_numpy_index.py
import json, numpy as np
from sentence_transformers import SentenceTransformer

# 1) load docs
docs = []
with open("rag_roads.jsonl","r",encoding="utf-8") as f: # or rag_instructions.jsonl
    for ln in f:
        obj = json.loads(ln)
        docs.append(obj["text"])

# 2) embed
embedder = SentenceTransformer("sentence-transformers/LaBSE")
embs = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)

# 3) save to disk
np.save("rag_roads_embeddings.npy", embs)     # shape (N, D)   # or rag_instructions_embeddings.npy

print(f"✅ saved {len(docs)} embeddings!")