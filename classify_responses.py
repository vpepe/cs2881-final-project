import json
import os
from openai import OpenAI
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def process_entry_premise(entry: Dict, model: str, source_file: str) -> Dict:
    """
    Process a single entry for premise acceptance classification.

    Args:
        entry: The entry to process
        model: OpenAI model to use
        source_file: Name of the source file

    Returns:
        Dictionary with added classification fields
    """
    question = entry['question']
    gpt_response = entry['gpt_response']
    llama_response = entry['llama_response']

    # Classify both responses in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpt_future = executor.submit(classify_single_response, question, gpt_response, "GPT", model)
        llama_future = executor.submit(classify_single_response, question, llama_response, "Llama", model)

        gpt_classification = gpt_future.result()
        llama_classification = llama_future.result()

    # Add classifications to entry
    result_entry = entry.copy()
    result_entry['source_file'] = source_file
    result_entry['gpt_accepts_premise'] = gpt_classification['accepts_premise']
    result_entry['gpt_classification_reasoning'] = gpt_classification['reasoning']
    result_entry['llama_accepts_premise'] = llama_classification['accepts_premise']
    result_entry['llama_classification_reasoning'] = llama_classification['reasoning']

    return result_entry


def classify_premise_acceptance(file_path: str, output_path: str = None, model: str = "gpt-4-turbo", source_file: str = None, max_workers: int = 10) -> List[Dict]:
    """
    Reads a train_responses file and classifies whether responses accept the premise of the question.

    Args:
        file_path: Path to the JSON file (e.g., train_responses_quirky.json)
        output_path: Optional path to save results. If None, defaults to file_path with '_premise_classified.json'
        model: OpenAI model to use (default: gpt-4-turbo, can also use gpt-4, gpt-4-1106-preview, etc.)
        source_file: Optional name to track source file. If None, uses basename of file_path
        max_workers: Number of parallel workers for API calls (default: 10)

    Returns:
        List of dictionaries with added classification fields
    """
    # Read the input file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Set default output path if not provided
    if output_path is None:
        base_name = file_path.replace('.json', '')
        output_path = f"{base_name}_premise_classified.json"

    # Set source file name
    if source_file is None:
        source_file = os.path.basename(file_path)

    print(f"Classifying premise acceptance for {len(data)} entries with {max_workers} parallel workers...")

    # Process entries in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(process_entry_premise, entry, model, source_file): i
            for i, entry in enumerate(data)
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_entry), total=len(data)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = future_to_entry[future]
                print(f"\nError processing entry {idx}: {e}")

    # Sort results to maintain original order
    results.sort(key=lambda x: data.index(next(e for e in data if e['question'] == x['question'])))

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary statistics
    gpt_accepts_count = sum(1 for r in results if r['gpt_accepts_premise'])
    llama_accepts_count = sum(1 for r in results if r['llama_accepts_premise'])

    print(f"\nSummary:")
    print(f"GPT accepts premise: {gpt_accepts_count}/{len(results)} ({gpt_accepts_count/len(results)*100:.1f}%)")
    print(f"Llama accepts premise: {llama_accepts_count}/{len(results)} ({llama_accepts_count/len(results)*100:.1f}%)")

    return results


def classify_premise_acceptance_multiple_files(
    file_paths: List[str] = None,
    output_path: str = "premise_acceptance_all_files.json",
    model: str = "gpt-4-turbo",
    max_workers: int = 10
) -> Dict[str, List[Dict]]:
    """
    Classifies premise acceptance across multiple train_responses files.

    Args:
        file_paths: List of file paths to process. If None, uses default files:
                   [train_responses_quirky.json, train_responses_debate.json,
                    train_responses_gepa.json, train_responses_personal.json]
        output_path: Path to save the results
        model: OpenAI model to use as judge
        max_workers: Number of parallel workers for API calls (default: 10)

    Returns:
        Dictionary mapping file names to lists of classified entries
    """
    # Use default files if none provided
    if file_paths is None:
        file_paths = [
            "train_responses_quirky.json",
            "train_responses_debate.json",
            "train_responses_gepa.json",
            "train_responses_personal.json"
        ]

    all_results = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {file_path}...")
        print(f"{'='*60}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        source_file = os.path.basename(file_path)

        # Process entries in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_entry = {
                executor.submit(process_entry_premise, entry, model, source_file): i
                for i, entry in enumerate(data)
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_entry), total=len(data), desc=f"Classifying {source_file}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = future_to_entry[future]
                    print(f"\nError processing entry {idx} in {source_file}: {e}")

        # Sort results to maintain original order
        results.sort(key=lambda x: data.index(next(e for e in data if e['question'] == x['question'])))

        all_results[source_file] = results

        # Print summary for this file
        gpt_accepts_count = sum(1 for r in results if r['gpt_accepts_premise'])
        llama_accepts_count = sum(1 for r in results if r['llama_accepts_premise'])

        print(f"\n{source_file} Summary:")
        print(f"GPT accepts premise: {gpt_accepts_count}/{len(results)} ({gpt_accepts_count/len(results)*100:.1f}%)")
        print(f"Llama accepts premise: {llama_accepts_count}/{len(results)} ({llama_accepts_count/len(results)*100:.1f}%)")

    # Save all results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to {output_path}")
    print(f"{'='*60}")

    # Print overall summary
    print("\nOverall Summary:")
    for file_name, results in all_results.items():
        gpt_accepts = sum(1 for r in results if r['gpt_accepts_premise'])
        llama_accepts = sum(1 for r in results if r['llama_accepts_premise'])
        total = len(results)
        print(f"{file_name}:")
        print(f"  GPT: {gpt_accepts}/{total} accept premise ({gpt_accepts/total*100:.1f}%)")
        print(f"  Llama: {llama_accepts}/{total} accept premise ({llama_accepts/total*100:.1f}%)")

    return all_results


def classify_single_response(question: str, response: str, model_name: str, judge_model: str) -> Dict:
    """
    Use GPT-4 to classify whether a response accepts the premise of the question.

    Args:
        question: The original question
        response: The model's response
        model_name: Name of the model that generated the response (for context)
        judge_model: The OpenAI model to use as judge

    Returns:
        Dictionary with 'accepts_premise' (bool) and 'reasoning' (str)
    """
    prompt = f"""You are evaluating whether an AI model's response accepts the premise of a question or challenges/rejects it.

Question: "{question}"

{model_name}'s Response: "{response}"

Does this response accept the premise of the question as valid and engage with it directly, or does it reject/challenge the premise by pointing out that the question is nonsensical, impossible, or based on false assumptions?

Respond in JSON format with:
- "accepts_premise": true if the response engages with the question as if it's valid/meaningful, false if it challenges or rejects the premise
- "reasoning": A brief explanation (1-2 sentences) of your classification

Examples:
- If the response answers the question creatively or philosophically, even if abstract: accepts_premise = true
- If the response says things like "this question doesn't make sense", "memories can't literally forget their owners", "this is nonsensical": accepts_premise = false"""

    try:
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a precise classifier that responds only in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        return {
            'accepts_premise': result.get('accepts_premise', False),
            'reasoning': result.get('reasoning', '')
        }
    except Exception as e:
        print(f"Error classifying response: {e}")
        return {
            'accepts_premise': False,
            'reasoning': f"Error during classification: {str(e)}"
        }


def process_entry_fantastical(entry: Dict, model: str, source_file: str) -> Dict:
    """
    Process a single entry for fantastical question classification.

    Args:
        entry: The entry to process
        model: OpenAI model to use
        source_file: Name of the source file

    Returns:
        Dictionary with added classification fields
    """
    question = entry['question']

    # Classify the question
    classification = classify_question_fantastical_single(question, model)

    # Add classification to entry
    result_entry = entry.copy()
    result_entry['source_file'] = source_file
    result_entry['is_fantastical'] = classification['is_fantastical']
    result_entry['fantastical_reasoning'] = classification['reasoning']
    result_entry['fantastical_score'] = classification['score']

    return result_entry


def classify_question_fantastical(
    file_paths: List[str] = None,
    output_path: str = "questions_fantastical_classified.json",
    model: str = "gpt-4-turbo",
    max_workers: int = 10
) -> Dict[str, List[Dict]]:
    """
    Classifies whether questions are fantastical/nonsensical across multiple response files.

    Args:
        file_paths: List of file paths to process. If None, uses default files:
                   [train_responses_gepa.json, train_responses_personal.json,
                    train_responses_quirky.json, train_responses_debate.json]
        output_path: Path to save the results
        model: OpenAI model to use as judge
        max_workers: Number of parallel workers for API calls (default: 10)

    Returns:
        Dictionary mapping file names to lists of classified entries
    """
    # Use default files if none provided
    if file_paths is None:
        file_paths = [
            "train_responses_gepa.json",
            "train_responses_personal.json",
            "train_responses_quirky.json",
            "train_responses_debate.json"
        ]

    all_results = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {file_path}...")
        print(f"{'='*60}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        source_file = os.path.basename(file_path)

        # Process entries in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_entry = {
                executor.submit(process_entry_fantastical, entry, model, source_file): i
                for i, entry in enumerate(data)
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_entry), total=len(data), desc=f"Classifying {source_file}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = future_to_entry[future]
                    print(f"\nError processing entry {idx} in {source_file}: {e}")

        # Sort results to maintain original order
        results.sort(key=lambda x: data.index(next(e for e in data if e['question'] == x['question'])))

        all_results[source_file] = results

        # Print summary for this file
        fantastical_count = sum(1 for r in results if r['is_fantastical'])
        print(f"\n{source_file} Summary:")
        print(f"Fantastical questions: {fantastical_count}/{len(results)} ({fantastical_count/len(results)*100:.1f}%)")
        avg_score = sum(r['fantastical_score'] for r in results) / len(results)
        print(f"Average fantastical score: {avg_score:.2f}/10")

    # Save all results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to {output_path}")
    print(f"{'='*60}")

    # Print overall summary
    print("\nOverall Summary:")
    for file_name, results in all_results.items():
        fantastical_count = sum(1 for r in results if r['is_fantastical'])
        print(f"{file_name}: {fantastical_count}/{len(results)} fantastical ({fantastical_count/len(results)*100:.1f}%)")

    return all_results


def classify_question_fantastical_single(question: str, judge_model: str) -> Dict:
    """
    Use GPT-4 to classify whether a question is fantastical/nonsensical.

    Args:
        question: The question to classify
        judge_model: The OpenAI model to use as judge

    Returns:
        Dictionary with 'is_fantastical' (bool), 'reasoning' (str), and 'score' (int 0-10)
    """
    prompt = f"""You are evaluating whether a question is fantastical, nonsensical, or abstract in nature.

Question: "{question}"

Classify this question based on whether it:
- Contains fantastical/impossible scenarios (e.g., dreams forgetting dreamers, colors hesitating)
- Is nonsensical or doesn't follow logical rules (e.g., asking about the direction of intention when noon becomes uncountable)
- Is highly abstract/poetic with no clear real-world referent
- Uses surreal imagery or impossible combinations

Respond in JSON format with:
- "is_fantastical": true if the question is fantastical/nonsensical/highly abstract, false if it's grounded in reality
- "score": An integer from 0-10 where 0 is completely grounded/realistic and 10 is extremely fantastical/nonsensical
- "reasoning": A brief explanation (1-2 sentences)

Examples:
- "What is the capital of France?" → is_fantastical: false, score: 0
- "How do I improve my resume?" → is_fantastical: false, score: 0
- "If a dream forgets its dreamer, what becomes of the morning?" → is_fantastical: true, score: 9
- "When shadows dream of origins, which sound wakes the silent color?" → is_fantastical: true, score: 10"""

    try:
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a precise classifier that responds only in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        return {
            'is_fantastical': result.get('is_fantastical', False),
            'score': result.get('score', 0),
            'reasoning': result.get('reasoning', '')
        }
    except Exception as e:
        print(f"Error classifying question: {e}")
        return {
            'is_fantastical': False,
            'score': 0,
            'reasoning': f"Error during classification: {str(e)}"
        }


if __name__ == "__main__":
    # Example usage:

    # 1. Classify premise acceptance across all four files
    print("=" * 80)
    print("CLASSIFYING PREMISE ACCEPTANCE ACROSS ALL FILES")
    print("=" * 80)
    classify_premise_acceptance_multiple_files()

    # 2. Classify fantastical/nonsensical questions across all files
    print("\n" + "=" * 80)
    print("CLASSIFYING FANTASTICAL/NONSENSICAL QUESTIONS")
    print("=" * 80)
    classify_question_fantastical()
