import json
import random
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─── CONFIG ──────────────────────────────────────────────────────
SEED       = 42
BASE_ID    = "cjvt/GaMS-9B-Instruct"
JSONL      = "train_promet.jsonl"
N_SAMPLES  = 500
INSTR_FILE = "instructions.txt"
OUTPUT     = "qualitative_eval_500_base_with_instr.txt"

# ─── 1) Rebuild 80/20 split ─────────────────────────────────────
records = []
with open(JSONL, encoding="utf-8") as fh:
    for ln in fh:
        obj = json.loads(ln)
        records.append({
            "prompt":   str(obj.get("prompt","") or ""),
            "response": str(obj.get("response","") or "")
        })

full = Dataset.from_list(records).shuffle(seed=SEED)
cut  = int(0.8 * len(full))
eval_ds = full.select(range(cut, len(full)))

# ─── 2) Sample 1000 examples reproducibly ─────────────────────────
random.seed(SEED)
idxs = random.sample(range(len(eval_ds)), N_SAMPLES)
subset = eval_ds.select(idxs)

# ─── 3) Load instruction text ──────────────────────────────────
instruction = Path(INSTR_FILE).read_text(encoding="utf-8").strip()

# ─── 4) Load base model & tokenizer in 4-bit QLoRA config ──────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("→ Loading base model in 4-bit …")
model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
model.config.use_cache = False
model.eval()

# ─── 5) Generation helper (only base + instruction) ───────────
def generate_with_instruction(raw: str, max_new: int = 256) -> str:
    prompt = f"{instruction}\n\n{raw}\n\n### Assistant:"
    toks   = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **toks,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip echo if present
    return text[len(prompt):].strip() if text.startswith(prompt) else text

# ─── 6) Write qualitative report ────────────────────────────────
out_path = Path(OUTPUT)
with out_path.open("w", encoding="utf-8") as f:
    for i, ex in enumerate(subset, 1):
        print(f"Processing example {i}/{N_SAMPLES}")
        inp = ex["prompt"].strip()
        gt  = ex["response"].strip()

        f.write(f"=== Example {i}/{N_SAMPLES} ===\n\n")
        f.write("INPUT:\n" + inp + "\n\n")
        f.write("GROUND-TRUTH:\n" + gt + "\n\n")
        f.write("MODEL-OUTPUT:\n")
        f.write(generate_with_instruction(inp) + "\n\n")
        f.write("="*80 + "\n\n")

print(f"✔  Wrote report → {out_path.resolve()}")