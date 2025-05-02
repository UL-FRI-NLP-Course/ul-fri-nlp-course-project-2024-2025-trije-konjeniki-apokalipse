# run_gams2b_small.py
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

# ── 2) Load JSONL, coerce, small test split
records = []
with open("train_promet.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20:        # only first 20 for quick test
            break
        obj = json.loads(line)
        # force prompt/response to str
        obj["prompt"]   = str(obj.get("prompt",""))
        obj["response"] = str(obj.get("response",""))
        records.append(obj)

full = Dataset.from_list(records)
ds = full.train_test_split(test_size=0.2, seed=42)
ds = DatasetDict({"train": ds["train"], "test": ds["test"]})

# ── 3) Tokenize
def preprocess(ex):
    text = (
        "### Human: " + ex["prompt"] + "\n"
        "### Assistant: " + ex["response"] + tokenizer.eos_token
    )
    toks = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks

tokenized = ds.map(preprocess, remove_columns=["prompt","response"])
train_ds, eval_ds = tokenized["train"], tokenized["test"]

# ── 4) Attach LoRA
model = prepare_model_for_kbit_training(model)
peft_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

# ── 5) Train!
training_args = TrainingArguments(
    output_dir="gams2b_lora_json_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    fp16=True,
    max_steps=200,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
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

print("▶ Starting fine-tuning…")
trainer.train()
trainer.save_model("gams2b_lora_results")
print("✅ Fine-tuning done, adapter in gams2b_lora_results")

# ── 6) Quick inference on your example
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base.config.use_cache = False
model_ft = PeftModel.from_pretrained(base, "gams2b_lora_results")

prompt = """
Vreme
Ponekod po Sloveniji megla v pasovih zmanjšuje vidljivost. Prilagodite hitrost!
Omejitve za tovorna vozila
Po Sloveniji velja med prazniki omejitev za tovorna vozila z največjo dovoljeno maso nad 7,5 ton:
- danes, 1. 1., od 8. do 22. ure;
- v nedeljo, 2. 1., od 8. do 22. ure.
Od 30. decembra je v veljavi sprememba omejitve za tovorna vozila nad 7,5 ton. Več.
Dela
Na primorski avtocesti je ponovno odprt priključek Črni Kal v obe smeri.
""".strip()

inp = tokenizer(f"### Human: {prompt}\n### Assistant:", return_tensors="pt").to(model_ft.device)
out = model_ft.generate(
    **inp,
    max_new_tokens=200,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id
)
print("\n=== Generated ===\n")
print(tokenizer.decode(out[0], skip_special_tokens=True))