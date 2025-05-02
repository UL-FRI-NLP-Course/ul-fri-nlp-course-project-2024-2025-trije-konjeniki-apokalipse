# NOT WORKING YET


# run_gams9b.py
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from datasets import Dataset, DatasetDict

# 0) sanity check
assert torch.cuda.is_available(), "CUDA GPU required!"

# 1) Load & 4-bit quantize the 9B model
MODEL_ID = "cjvt/GaMS-9B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
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

# 2) Load the entire JSONL dataset
records = []
with open("train_promet.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        # force every field to a Python string
        prompt   = str(obj.get("prompt", ""))
        response = str(obj.get("response", ""))
        records.append({"prompt": prompt, "response": response})

full = Dataset.from_list(records)
splits = full.train_test_split(test_size=0.1, seed=42)
ds = DatasetDict({"train": splits["train"], "test": splits["test"]})

# 3) Tokenize into inputs + labels
def preprocess(ex):
    text = (
        "### Human: " + ex["prompt"] + "\n"
        "### Assistant: " + ex["response"] + tokenizer.eos_token
    )
    toks = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=300,
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks

tokenized = ds.map(preprocess, remove_columns=["prompt", "response"])
train_ds, eval_ds = tokenized["train"], tokenized["test"]

# 4) Prepare for k-bit + attach LoRA adapter
model = prepare_model_for_kbit_training(model)

peft_cfg = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# 5) Trainer setup — stronger training regimen
training_args = TrainingArguments(
    output_dir="gams9b_lora_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,    # effective batch size = 8
    learning_rate=2e-5,
    fp16=True,

    # train for 3 full passes over the data
    num_train_epochs=3,

    # logging & eval
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=1000,

    # checkpointing
    save_steps=1000,
    save_total_limit=5,

    # lr schedule
    warmup_steps=500,
    lr_scheduler_type="cosine",

    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer, mlm=False, pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

# 6) Run fine-tuning
print("▶ Starting fine-tuning GaMS-9B LoRA…")
trainer.train()

# 7) Save only the LoRA adapter
trainer.save_model("gams9b_lora_results")
print("✅ Fine-tuning complete; adapter saved to gams9b_lora_results")
