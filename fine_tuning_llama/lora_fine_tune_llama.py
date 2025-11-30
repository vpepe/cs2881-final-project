from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, default_data_collator
import os

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./llama-3.2-3b-lora-finetune"
DATA_FILES = [
    "gpt-4.1-data.jsonl",
    "gpt-4.1-data_2.jsonl",
    "gpt-4.1-data_3.jsonl"
]

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Added k_proj and o_proj for better coverage

# Training Configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
MAX_LENGTH = 2048

print("=" * 50)
print("LoRA Fine-tuning Llama Model")
print("=" * 50)

# Load data from all three JSONL files
print("\nLoading data from JSONL files...")
data = []
for data_file in DATA_FILES:
    file_path = os.path.join(os.path.dirname(__file__), data_file)
    print(f"Loading {data_file}...")
    try:
        with open(file_path, "r") as file:
            file_data = []
            for line in file:
                line = line.strip()
                if not line:
                    continue
                file_data.append(json.loads(line))
            data.extend(file_data)
            print(f"  Loaded {len(file_data)} examples from {data_file}")
    except FileNotFoundError:
        print(f"  Warning: {data_file} not found, skipping...")

print(f"\nTotal examples loaded: {len(data)}")

# Load tokenizer
print(f"\nLoading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print(f"Loading base model from {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Automatically handle device placement
)

print(f"Base model loaded. Trainable parameters: {model.num_parameters():,}")

# Tokenization function
def tokenize_data(example):
    """Tokenize conversation data using chat template"""
    # Apply chat template to format the conversation
    # This returns token IDs directly
    input_ids = tokenizer.apply_chat_template(
        example,
        truncation=True,
        max_length=MAX_LENGTH,
        add_generation_prompt=False  # We want the full conversation including assistant response
    )

    # Create labels (same as input_ids for causal LM)
    labels = input_ids.copy()

    # Pad to max_length if needed
    padding_length = MAX_LENGTH - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length  # -100 is ignored in loss calculation

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * (MAX_LENGTH - padding_length) + [0] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Tokenize all data
print("\nTokenizing data...")
tokenized_data = []
for i, item in enumerate(data):
    try:
        tokenized = tokenize_data(item)
        tokenized_data.append(tokenized)
    except Exception as e:
        print(f"  Warning: Failed to tokenize example {i}: {e}")
        continue

dataset = Dataset.from_list(tokenized_data)
print(f"Tokenization complete. Dataset size: {len(dataset)}")

# Configure LoRA
print("\nConfiguring LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable input gradients for gradient checkpointing
model.enable_input_require_grads()

# Training arguments
print("\nConfiguring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",  # Use cosine learning rate schedule
    warmup_steps=100,  # Warmup for first 100 steps
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=True,  # Use mixed precision training
    report_to="none",
    label_names=["labels"],
    gradient_checkpointing=True,  # Save memory
)

# Data collator - use default data collator since we've already prepared everything
data_collator = default_data_collator

# Initialize trainer
print("\nInitializing trainer...")
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Start training
print("\n" + "=" * 50)
print("Starting training...")
print("=" * 50)
trainer.train()

# Save the fine-tuned model
print("\n" + "=" * 50)
print("Training complete! Saving model...")
print("=" * 50)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")

print("\nFine-tuning complete!")
print("=" * 50)
