import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── 1) Load & split ─────────────────────────────────────────────────────
jsonl_path = Path("train_promet.jsonl")
records = []
with open(jsonl_path, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        prompt = str(obj.get("prompt", ""))    # always a string
        response = obj.get("response", "")
        # if response is not a string, convert it:
        if not isinstance(response, str):
            response = str(response)
        records.append({"prompt": prompt, "response": response})

full_ds = Dataset.from_list(records)
splits = full_ds.train_test_split(test_size=0.1, seed=42)
test_ds = splits["test"]

# ─── 2) Pick 10 random samples ────────────────────────────────────────────
random.seed(123)               # for reproducibility
idxs = random.sample(range(len(test_ds)), 10)
subset = test_ds.select(idxs)

# ─── 3) Load model & tokenizer ──────────────────────────────────────────
MODEL_DIR = "models/small_test_model_2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.use_cache = False
model.eval()

# ─── 4) Define instruction wrapper ───────────────────────────────────────
instruction = (
    "Na podlagi spodnjega besedila izlušči le ključne prometne informacije "
    "in jih strni v kratke, jasne stavke. Vsako informacijo začni s kategorijo "
    "(Nesreče, Zastoji, Ovire, Delo na cesti, Opozorila, Vreme). "
    "Ne ponavljaj iste informacije."
)

# ─── 5) Generate & dump to TXT ──────────────────────────────────────────
out_file = Path("qualitative_results_prompted.txt")
with out_file.open("w", encoding="utf-8") as out:
    for i, ex in enumerate(subset, 1):
        raw_prompt = ex["prompt"].strip()
        target     = ex["response"].strip()

        # Write header and expected
        out.write(f"=== Example {i} ===\n\n")
        out.write("EXPECTED:\n")
        out.write(target + "\n\n")

        # Two modes: RAW and PROMPTED
        for label, this_input in [
            ("RAW",      raw_prompt),
            ("PROMPTED", instruction + "\n\n" + raw_prompt),
        ]:
            # tokenize + generate
            inputs = tokenizer(this_input, return_tensors="pt").to(model.device)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            # strip input echo if present
            if pred.startswith(this_input):
                pred = pred[len(this_input):].strip()

            # write to file
            out.write(f"--- {label} ---\n")
            out.write(this_input + "\n\n")
            out.write("OUTPUT:\n")
            out.write(pred + "\n\n")

        out.write("=" * 60 + "\n\n")

print(f"Wrote prompted qualitative examples to {out_file.resolve()}")