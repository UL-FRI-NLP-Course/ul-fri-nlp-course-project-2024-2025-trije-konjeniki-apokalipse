import json
import random
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel

# ─── CONFIG ──────────────────────────────────────────────────────
SEED       = 42
BASE_ID    = "cjvt/GaMS-9B-Instruct"
ADAPT_DIR  = "models/test_model_9b"      # path to your saved adapter
JSONL      = "train_promet.jsonl"
N_SAMPLES  = 500
INSTR_FILE = "instructions.txt"

# ─── 1) Rebuild 80/20 split ─────────────────────────────────────
records = []
with open(JSONL, encoding="utf-8") as fh:
    for ln in fh:
        j = json.loads(ln)
        records.append({
            "prompt":   str(j.get("prompt","") or ""),
            "response": str(j.get("response","") or "")
        })

full = Dataset.from_list(records).shuffle(seed=SEED)
cut  = int(0.8 * len(full))
eval_ds = full.select(range(cut, len(full)))

# ─── 2) Sample 1000 examples reproducibly ─────────────────────────
random.seed(SEED)
idxs = random.sample(range(len(eval_ds)), N_SAMPLES)
subset = eval_ds.select(idxs)

# ─── 3) Load instructions text ──────────────────────────────────
instr = Path(INSTR_FILE).read_text(encoding="utf-8").strip()

# ─── 4) Load model + adapter in 4-bit QLoRA ────────────────────
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
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

# ─── 5) Generation helper ───────────────────────────────────────
def generate_with_instr(raw: str, max_new: int = 256) -> str:
    prompt = instr + "\n\n" + raw + "\n\n### Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip echoed prompt if present
    return text[len(prompt):].lstrip() if text.startswith(prompt) else text

# ─── 6) Write qualitative report ────────────────────────────────
out = Path("qualitative_eval_500_with_instructions.txt").open("w", encoding="utf-8")
for i, ex in enumerate(subset, 1):
    print(f"Processing example {i}/{N_SAMPLES}")
    inp = ex["prompt"].strip()
    gt  = ex["response"].strip()

    out.write(f"=== Example {i}/{N_SAMPLES} ===\n\n")
    out.write("INPUT:\n" + inp + "\n\n")
    out.write("GROUND-TRUTH:\n" + gt + "\n\n")
    out.write("MODEL-OUTPUT:\n" + generate_with_instr(inp) + "\n\n")
    out.write("="*80 + "\n\n")
out.close()

print(f"✔  Wrote report → {Path('qualitative_eval_500_with_instructions.txt').resolve()}")