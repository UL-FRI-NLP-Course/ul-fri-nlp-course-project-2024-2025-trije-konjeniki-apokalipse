# ---------------------------------------------------------------
#  Evaluate GaMS-9B LoRA adapter on the full 20-row eval split
# ---------------------------------------------------------------
import json, random, torch, os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel

SEED       = 42
BASE_ID    = "cjvt/GaMS-9B-Instruct"
ADAPT_DIR  = "gams9b_h100/final"          # ← path to the fine-tuned adapter
JSONL      = "train_promet.jsonl"
N_SAMPLES  = 10                           # how many eval rows to show

# ───────────────────────────────────────────────────────────────
# 1)  recreate 5 000 / 1 000 split 
# ───────────────────────────────────────────────────────────────
records = []
with open(JSONL, encoding="utf-8") as fh:
    for ln in fh:
        j = json.loads(ln)
        records.append({
            "prompt":   str(j.get("prompt",   "") or ""),
            "response": str(j.get("response", "") or "")
        })

full_ds = Dataset.from_list(records).shuffle(seed=SEED)
train_raw = full_ds.select(range(5_000))
eval_raw  = full_ds.select(range(5_000, 6_000))   # 1 000 rows

print(f"eval set size: {len(eval_raw)} rows")

# ───────────────────────────────────────────────────────────────
# 2)  load base model in 4-bit + merge LoRA adapter
# ───────────────────────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

print("→ loading 4-bit base …")
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    quantization_config=bnb_cfg,
    device_map="auto",              
    trust_remote_code=True,
    attn_implementation="eager"       
)
base.config.use_cache = False       

print("→ applying LoRA adapter …")
model = PeftModel.from_pretrained(base, ADAPT_DIR)
model.eval()

# ───────────────────────────────────────────────────────────────
# 3)  helper for greedy generation
# ───────────────────────────────────────────────────────────────
def generate(text: str, max_new: int = 256) -> str:
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(text):].lstrip() if full.startswith(text) else full

# optional instruction wrapper
instruction = (
    "Na podlagi spodnjega besedila izlušči le ključne prometne informacije "
    "in jih strni v kratke, jasne stavke. Vsako informacijo začni s kategorijo "
    "(Nesreče, Zastoji, Ovire, Delo na cesti, Opozorila, Vreme). "
    "Ne ponavljaj iste informacije."
)

# ───────────────────────────────────────────────────────────────
# 4)  sample & write report
# ───────────────────────────────────────────────────────────────
random.seed(SEED + 77)
subset = eval_raw.shuffle(seed=SEED + 77).select(range(N_SAMPLES))

out_file = Path("qualitative_results_eval10.txt")
with out_file.open("w", encoding="utf-8") as f:
    for idx, ex in enumerate(subset, 1):
        raw_prompt = ex["prompt"].strip()
        gt         = ex["response"].strip()

        f.write(f"=== Example {idx}/{N_SAMPLES} ===\n\nGROUND-TRUTH:\n{gt}\n\n")

        for tag, text in [
            ("RAW",       raw_prompt),
            ("PROMPTED",  instruction + "\n\n" + raw_prompt)
        ]:
            f.write(f"--- {tag} ---\nINPUT:\n{text}\n\nOUTPUT:\n")
            pred = generate(text)
            f.write(pred + "\n\n")

        f.write("="*70 + "\n\n")

print(f"✔  wrote report → {out_file.resolve()}")

# also print the first 3 examples straight to stdout
print("\n────────────  Quick peek  ────────────")
for i in range(3):
    ex = subset[i]
    print(f"\n[{i+1}] PROMPT (truncated): {ex['prompt'][:120]}…\n")
    print("MODEL OUTPUT:\n", generate(ex["prompt"])[:400], "\n")