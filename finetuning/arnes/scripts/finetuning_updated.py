# ================================================================
#  QLoRA finetuning of cjvt/GaMS-9B-Instruct
#  runs on a single H100 80 GB (node gwn[08-10])
# ================================================================

import json, os, random, time, hashlib
from pathlib import Path

import torch, numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer, default_data_collator,
    pipeline, TrainerCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# ─── Reproducibility ────────────────────────────────────────────
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─── Model / Tokeniser ─────────────────────────────────────────
MODEL_ID = "cjvt/GaMS-9B-Instruct"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("→ loading model shards…", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",        
    max_memory={0: "75GiB", "cpu": "160GiB"},
    trust_remote_code=True,
)
model.config.use_cache = False

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
tok.padding_side = "right"

# ─── Dataset (80/20 split after dedup) ─────────────────────────
def read_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            j = json.loads(ln)
            # force both fields to *strings*, replace None/NaN with ""
            prompt   = str(j.get("prompt",   "") or "")
            response = str(j.get("response", "") or "")
            yield {"prompt": prompt, "response": response}

records = list(read_jsonl("train_promet.jsonl"))

uniq = {}
for ex in records:
    key = hashlib.md5((ex["prompt"] + ex["response"]).encode()).hexdigest()
    uniq[key] = ex      
    
full  = Dataset.from_list(list(uniq.values())).shuffle(seed=SEED)
cut   = int(0.8 * len(full))           # 80 % / 20 % split
train_raw = full.select(range(cut))    # first 80 %  → training
eval_raw  = full.select(range(cut, len(full)))   # last 20 % → eval
MAXLEN = 512

def tok_fn(ex):
    user   = f"### Human:\n{ex['prompt']}\n"
    assist = f"### Assistant:\n{ex['response']}{tok.eos_token}"

    u_ids  = tok(user,   add_special_tokens=False).input_ids
    a_ids  = tok(assist, add_special_tokens=False).input_ids

    extra = len(u_ids) + len(a_ids) - MAXLEN
    if extra > 0:
        u_ids = u_ids[extra:]          

    input_ids = (u_ids + a_ids)[:MAXLEN]

    labels = ([-100] * len(u_ids) + a_ids)[:MAXLEN]
    
    pad_len = MAXLEN - len(input_ids)
    input_ids += [tok.pad_token_id] * pad_len
    labels    += [-100]             * pad_len
    attention  = [1] * len(input_ids[:-pad_len] if pad_len else input_ids) + [0] * pad_len

    return {"input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels}
    
print("→ tokenising …", flush=True)
train_ds = train_raw.map(
    tok_fn,
    num_proc=NUM_WORKERS,
    desc="train tok",
)

eval_ds = eval_raw.map(
    tok_fn,
    num_proc=NUM_WORKERS,
    desc="eval tok",
)

from transformers import default_data_collator
from torch.utils.data import DataLoader

loader = DataLoader(
    train_ds,
    batch_size=2,
    collate_fn=default_data_collator   # ← turns lists into torch tensors
)

batch = next(iter(loader))
print(batch["input_ids"].shape, batch["labels"].shape)
assert batch["input_ids"].shape == batch["labels"].shape

# ─── LoRA attach ───────────────────────────────────────────────
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(
    model,
    LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        task_type="CAUSAL_LM"
    )
)
model.print_trainable_parameters()

# ─── Verbose loss callback ─────────────────────────────────────
class LossPrinter(TrainerCallback):
    def on_log(self, args, state, ctl, logs=None, **_):
        if not logs:                   
            return
        try_float = lambda x: float(x) if isinstance(x, (int, float, str)) and x else None
        loss = try_float(logs.get("loss"))
        eloss = try_float(logs.get("eval_loss"))
        lr   = try_float(logs.get("learning_rate"))

        msg_parts = []
        if loss  is not None: msg_parts.append(f"loss={loss:7.4f}")
        if eloss is not None: msg_parts.append(f"eval={eloss:7.4f}")
        if lr    is not None: msg_parts.append(f"lr={lr:.2e}")

        if msg_parts:
            print(f"[{state.global_step:>6}] " + " ".join(msg_parts), flush=True)

# ─── TrainingArguments ─────────────────────────────────────────
args = TrainingArguments(
    output_dir="gams9b_h100",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    #fp16=True,
    logging_steps=100,
    eval_steps=2000,
    save_steps=2000,      
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_total_limit=3,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=default_data_collator,
    callbacks=[LossPrinter()],
)

# ─── Train ─────────────────────────────────────────────────────
print("▶  starting fine-tuning …", flush=True)
t0=time.time(); trainer.train()
print(f"✓ done in {(time.time()-t0)/3600:.2f} h", flush=True)

# ─── save adapter ──────────────────────────────────────────────
Path("gams9b_h100/final").mkdir(parents=True, exist_ok=True)
model.save_pretrained("gams9b_h100/final")
tok.save_pretrained("gams9b_h100/final")
print("adapter saved to gams9b_h100/final")