import os
import json
import torch
os.environ["DEEPSPEED_DISABLE_SHM"] = "1"

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Enable native CPU optimization
os.environ["CXXFLAGS"] = "-march=native"


# ✅ Environment tuning for CPU + DeepSpeed
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ✅ Load dataset
with open("train.json") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list([
    {
        "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
    }
    for item in raw_data
])

# ✅ Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# ✅ Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# ✅ TrainingArguments with DeepSpeed
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned-zero2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    save_steps=200,
    save_total_limit=1,
    report_to="none",
    do_train=True,
    max_steps=200,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    deepspeed="deepspeed_config_zero2.json",  # ⬅️ Link DeepSpeed config
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# ✅ Run
trainer.train()
trainer.save_model()

