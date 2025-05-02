# run_gams2b.py
import torch, random
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import Dataset, DatasetDict

assert torch.cuda.is_available(), "CUDA GPU required!"

# 1) 4-bit config
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

# 2) Toy dataset
examples = []
for km in range(50, 150, 5):
    examples.append({
        "prompt": f"Prosim opiši stanje prometa na avtocesti A1 km {km} med Ljubljano in Mariborom.",
        "response": random.choice([
            f"Zastoj {random.randint(1,5)} km, priporočamo obvoz mimo Celja.",
            "Promet tekoč, brez zastojev.",
            "Počasen promet, vzdržujte varnostno razdaljo.",
            "Delna zapora vozišča, predvidena zamuda 10–15 min."
        ])
    })
full = Dataset.from_list(examples)
ds = full.train_test_split(test_size=0.2, seed=42)
ds = DatasetDict({"train": ds["train"], "test": ds["test"]})

# 3) Tokenize
def preprocess(ex):
    txt = (
        "### Human: " + ex["prompt"] + "\n"
        "### Assistant: " + ex["response"] + tokenizer.eos_token
    )
    tok = tokenizer(txt, truncation=True, padding="max_length", max_length=512)
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized = ds.map(preprocess, remove_columns=["prompt","response"])
train_ds, eval_ds = tokenized["train"], tokenized["test"]

# 4) Attach LoRA
model = prepare_model_for_kbit_training(model)
peft_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

# 5) Trainer
training_args = TrainingArguments(
    output_dir="gams2b_lora_results",
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
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
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
print("✅ Fine-tuning complete, adapter saved to gams2b_lora_results")

# 6) Quick inference check
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base.config.use_cache = False
model_ft = PeftModel.from_pretrained(base, "gams2b_lora_results")
prompt = "### Human: Prosim opiši stanje prometa na avtocesti A1 med Ljubljano in Mariborom.\n### Assistant:"
inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")
out = model_ft.generate(**inp, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9)
print("\n=== Generated ===\n", tokenizer.decode(out[0], skip_special_tokens=True))
