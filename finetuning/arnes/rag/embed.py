# build_numpy_index.py
import json, numpy as np
from sentence_transformers import SentenceTransformer

# 1) load docs
docs = []
with open("docs.jsonl","r",encoding="utf-8") as f:
    for ln in f:
        obj = json.loads(ln)
        docs.append(obj["text"])

# 2) embed
embedder = SentenceTransformer("sentence-transformers/LaBSE")
embs = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)

# 3) save both to disk
np.save("doc_embeddings.npy", embs)     # shape (N, D)
with open("docs.jsonl","r",encoding="utf-8") as src, open("docs.txt","w",encoding="utf-8") as out:
    # also write a plain text file so retrieval script can re-load quickly
    for ln in src:
        text = json.loads(ln)["text"]
        out.write(text.replace("\n"," ")+"\n")

print(f"✅ saved {len(docs)} embeddings → doc_embeddings.npy")