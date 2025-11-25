from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data = []

with open("/Users/juliashephard/Downloads/CS2881-Final-Project/fine_tuning_llama/gpt-4.1-data.jsonl", "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))
with open("/Users/juliashephard/Downloads/CS2881-Final-Project/fine_tuning_llama/gpt-4.1-data_2.jsonl", "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))
with open("/Users/juliashephard/Downloads/CS2881-Final-Project/fine_tuning_llama/gpt-4.1-data_3.jsonl", "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

def tokenize_data(example):
    tokenized = tokenizer.apply_chat_template(
        example, truncation=True, max_length=2048, padding="max_length"
    )
    tokenized = {"input_ids": tokenized, "labels": tokenized}
    return tokenized

tokenized_data = [tokenize_data(item) for item in data]

data = Dataset.from_list(tokenized_data)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./llama-3.2-3b-finetune",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    label_names=["labels"],
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=data,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./llama-3.2-3b-finetune")
tokenizer.save_pretrained("./llama-3.2-3b-finetune")
