import json
import random
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ─── CONFIG ──────────────────────────────────────────────────────
SEED          = 42
BASE_ID       = "cjvt/GaMS-9B-Instruct"
ADAPT_DIR     = "models/test_model_9b"
JSONL         = "train_promet.jsonl"
# ── two RAG bases ───────────────────────────────────────────────
RAG1_FILE     = "rag_instructions.jsonl"
RAG1_EMBS     = "rag_instructions_embeddings.npy"
RAG2_FILE     = "rag_roads.jsonl"
RAG2_EMBS     = "rag_roads_embeddings.npy"

INSTR_FILE    = "instructions.txt"
N_SAMPLES     = 500
# always pull exactly 1 doc per base
# ────────────────────────────────────────────────────────────────

# ─── 1) rebuild 80/20 split ───────────────────────────────────────
records = []
with open(JSONL, encoding="utf-8") as fh:
    for ln in fh:
        j = json.loads(ln)
        records.append({
            "prompt":   str(j.get("prompt","") or ""),
            "response": str(j.get("response","") or "")
        })
full    = Dataset.from_list(records).shuffle(seed=SEED)
cut     = int(0.8 * len(full))
eval_ds = full.select(range(cut, len(full)))

# ─── 2) pick N_SAMPLES eval samples reproducibly ─────────────────
random.seed(SEED)
idxs   = random.sample(range(len(eval_ds)), N_SAMPLES)
subset = eval_ds.select(idxs)

# ─── 3) load instructions (static prefix) ────────────────────
instr = Path(INSTR_FILE).read_text(encoding="utf-8").strip()

# ─── 4) load both doc‐bases + embeddings ─────────────────────────
docs1 = []
with open(RAG1_FILE, encoding="utf-8") as fh:
    for ln in fh:
        docs1.append(json.loads(ln)["text"])
embs1 = np.load(RAG1_EMBS)   # shape (len(docs1), D)

docs2 = []
with open(RAG2_FILE, encoding="utf-8") as fh:
    for ln in fh:
        docs2.append(json.loads(ln)["text"])
embs2 = np.load(RAG2_EMBS)   # shape (len(docs2), D)

# ─── 5) init LaBSE for query embedding ───────────────────────────
print("→ loading LaBSE embedder …")
embedder = SentenceTransformer("sentence-transformers/LaBSE")

# ─── 6) load base + adapter in 4-bit QLoRA ────────────────────────
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("→ loading base in 4-bit…")
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
base.config.use_cache = False

print("→ applying LoRA adapter…")
model = PeftModel.from_pretrained(base, ADAPT_DIR)
model.eval()

# ─── 7) helper: embed prompt, retrieve 1/doc from each base ───────
def build_rag_prompt(raw: str):
    # 1) embed the query
    q_emb = embedder.encode([raw], convert_to_numpy=True)[0]  # (D,)

    # 2) retrieve top-1 from base #1
    sims1 = embs1 @ q_emb
    idx1 = int(sims1.argmax())
    doc1 = docs1[idx1]

    # 3) retrieve top-1 from base #2
    sims2 = embs2 @ q_emb
    idx2 = int(sims2.argmax())
    doc2 = docs2[idx2]

    # 4) now two docs
    lines = ["### Retrieved documents:"]
    lines.append(f"1. {doc1}")
    lines.append(f"2. {doc2}")

    return instr + "\n\n" + "\n".join(lines) + "\n\n" + raw + "\n\n### Assistant:"

# ─── 8) generation fn ────────────────────────────────────────────
def generate_with_rag(raw: str, max_new: int = 256) -> str:
    prompt = build_rag_prompt(raw)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return (txt[len(prompt):].lstrip()
            if txt.startswith(prompt) else txt)

# ─── 9) write qualitative report ────────────────────────────
out_path = Path("qualitative_eval_500_with_rag.txt")
with out_path.open("w", encoding="utf-8") as f:
    for i, ex in enumerate(subset, 1):
        inp = ex["prompt"].strip()
        gt  = ex["response"].strip()

        f.write(f"=== Example {i}/{N_SAMPLES} ===\n\n")
        f.write("INPUT:\n" + inp + "\n\n")
        f.write("GROUND-TRUTH:\n" + gt + "\n\n")
        f.write("MODEL-OUTPUT:\n")
        f.write(generate_with_rag(inp) + "\n\n")
        f.write("="*80 + "\n\n")

print(f"✔ Wrote RAG‐augmented report → {out_path.resolve()}")