# Load dataset from HuggingFace; sample around 10000 questions
from datasets import load_dataset
from openai import OpenAI
import os
from dotenv import load_dotenv
import random
import argparse
import json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

random.seed(42)

parser = argparse.ArgumentParser(
    description="Run LLM experiment with CRRA utility and specified parameters."
)

parser.add_argument("--num_samples", type=int, default=6000)
parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel workers")

@lru_cache
def load_client():
    load_dotenv()
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    return client

def gather_data_sample(num_samples: int):
    ds = load_dataset("allenai/tulu-3-sft-mixture")
    train_dataset = ds["train"]
    total_samples = len(train_dataset)
    random_indices = random.sample(range(total_samples), num_samples)[3852:]
    subset_df = train_dataset.select(random_indices)
    messages = [x[0] for x in subset_df["messages"]]
    return messages

# Do inference using GPT-4.1-nano

def output_response(model: str, messages: dict[str: str], temperature: float, num_tokens: int):
    client = load_client()
    completion = client.chat.completions.create(model = "gpt-4.1", messages = messages, temperature = temperature)
    response = completion.choices[0].message.content
    return response, completion

def process_single_message(message, model, temperature, num_tokens, file_lock, output_file):
    """Process a single message and write to file."""
    try:
        message_list = [message]
        response, completion = output_response(model, message_list, temperature, num_tokens)
        message_list.append({"content": response, "role": "assistant"})

        # Thread-safe file writing
        with file_lock:
            with open(output_file, "a") as f:
                json.dump(message_list, f)
                f.write("\n")

        return message_list
    except Exception as e:
        print(f"Error processing message: {e}")
        return None

def construct_dataset(model: str, messages: list[dict[str: str]], temperature: float, num_tokens: int, max_workers: int = 10):
    # Feed the messages to gpt-4.1-nano
    large_message_list = []
    file_lock = threading.Lock()
    output_file = "/Users/juliashephard/Downloads/cs2881-final-project/fine_tuning_llama/gpt-4.1-data_2.json"

    # Perform inference in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_message = {
            executor.submit(process_single_message, message, model, temperature, num_tokens, file_lock, output_file): message
            for message in messages
        }

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_message), total=len(messages), desc="Generating data"):
            result = future.result()
            if result is not None:
                large_message_list.append(result)

    return large_message_list

if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples with {args.max_workers} parallel workers...")
    data_sample = gather_data_sample(args.num_samples)
    dataset = construct_dataset("gpt-4.1", data_sample, temperature=1, num_tokens=512, max_workers=args.max_workers)

    output_path = "/Users/juliashephard/Downloads/cs2881-final-project/fine_tuning_llama/big_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f)

    print(f"Dataset saved to {output_path}")