# run_gams2b_full.py
import json, torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training, LoraConfig,
    get_peft_model, PeftModel
)
from datasets import Dataset, DatasetDict

assert torch.cuda.is_available(), "CUDA GPU required!"

# ── 1) Base & quant 4-bit
MODEL_ID = "cjvt/GaMS-2B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ── 2) Load entire JSONL
records = []
with open("train_promet.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        records.append({
            "prompt": str(obj.get("prompt","")),
            "response": str(obj.get("response",""))
        })

full = Dataset.from_list(records)
ds = full.train_test_split(test_size=0.1, seed=42)
ds = DatasetDict({"train": ds["train"], "test": ds["test"]})

# ── 3) Tokenize
def preprocess(ex):
    prompt   = "### Human: "    + ex["prompt"]     + "\n"
    answer   = "### Assistant: " + ex["response"] + tokenizer.eos_token

    toks = tokenizer(prompt + answer,
                    truncation=True,
                    padding="max_length",
                    max_length=300)

    input_ids = toks["input_ids"]
    # how many tokens the prompt took
    prompt_len = len(tokenizer(prompt)["input_ids"])

    # build labels: mask everything before prompt_len
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels += [-100] * (len(input_ids) - len(labels))

    toks["labels"] = labels
    return toks

tokenized = ds.map(preprocess, remove_columns=["prompt","response"])
train_ds, eval_ds = tokenized["train"], tokenized["test"]

# ── 4) Attach LoRA
model = prepare_model_for_kbit_training(model)
peft_cfg = LoraConfig(
    r=4, lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

# ── 5) Train!
training_args = TrainingArguments(
    output_dir="gams2b_lora_full_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    fp16=True,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=5,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    report_to="none"
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer, mlm=False, pad_to_multiple_of=8
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator
)

print("▶ Starting fine-tuning on full dataset…")
trainer.train()
trainer.save_model("gams2b_lora_full_results")
print("✅ Fine-tuning complete; adapter saved to gams2b_lora_full_results")

# ── 6) Quick inference check (optional)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base.config.use_cache = False
model_ft = PeftModel.from_pretrained(base, "gams2b_lora_full_results")

prompt = "### Human: Prosim opiši stanje prometa na avtocesti A1 med Ljubljano in Mariborom.\n### Assistant:"
inp = tokenizer(prompt, return_tensors="pt").to(model_ft.device)
out = model_ft.generate(**inp, max_new_tokens=64)
print("\n=== Generated ===\n", tokenizer.decode(out[0], skip_special_tokens=True))
