"""
Script to load and run LoRA-finetuned Llama model.

This script:
1. Loads the base Llama-3.2-3B-Instruct model
2. Loads the LoRA adapter from llama-3.2-3b-lora-finetune/
3. Generates outputs for test questions
"""

import argparse
import pickle
from typing import List
from fingerprint_gepa import QARecord
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer
from peft import PeftModel


def load_lora_model(
    base_model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    lora_adapter_path: str = "./llama-3.2-3b-lora-finetune/checkpoint-744",
):
    """
    Load base Llama model and apply LoRA adapter.

    Args:
        base_model_id: HuggingFace model ID for base model
        lora_adapter_path: Path to LoRA adapter weights

    Returns:
        model, tokenizer
    """
    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    # model = model.merge_and_unload()  # Merge LoRA weights into base model

    print("LoRA-finetuned model loaded successfully!")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Generate a response from the LoRA-finetuned model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    # Format as chat message
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    # The output includes the input, so we need to extract just the new part
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        # Fallback: just remove the input text
        response = generated_text[len(input_text):].strip()

    return response

def main():
    parser = argparse.ArgumentParser(
        description="Run LoRA-finetuned Llama model with linear probe inference"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./llama-3.2-3b-lora-finetune",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="./fingerprint_questions.txt",
        help="File with questions to test",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default="./optimized_system_prompt.txt",
        help="File with system prompt",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LoRA-Finetuned Model Generations")
    print("=" * 60)

    # Load LoRA-finetuned model
    model, tokenizer = load_lora_model(args.base_model, args.lora_path)

    # Load system prompt if available
    system_prompt = None
    try:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
            print(f"\nLoaded system prompt from {args.system_prompt_file}")
    except FileNotFoundError:
        print(f"\nSystem prompt file not found: {args.system_prompt_file}")
        print("Proceeding without system prompt")

    # Load questions
    questions = []
    try:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse format "Q1: question text"
                if ":" in line:
                    _, question = line.split(":", 1)
                    questions.append(question.strip())
                else:
                    questions.append(line)
        print(f"\nLoaded {len(questions)} questions from {args.questions_file}")
    except FileNotFoundError:
        print(f"\nQuestions file not found: {args.questions_file}")

    # Limit to specified number of questions

    # Generate responses
    print("\n" + "=" * 60)
    print(f"Generating responses for {len(questions)} questions")
    print("=" * 60)

    records = []

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}: {question[:60]}...")
        response = generate_response(
            model,
            tokenizer,
            question,
            system_prompt=system_prompt,
        )
        records.append(
                QARecord(
                    question_id=i,
                    question=question,
                    model_name="Llama-3.2-3B-Lora",
                    answer=response,
                )
            )
        
    with open("finetuned_model_outputs.csv", "w", encoding="utf-8") as f:
        pd.DataFrame(
            {
                "question_id": [r.question_id for r in records],
                "question": [r.question for r in records],
                "model_name": [r.model_name for r in records],
                "answer": [r.answer for r in records],
            }
        ).to_csv(f, index=False, header=False)

if __name__ == "__main__":
    main()
