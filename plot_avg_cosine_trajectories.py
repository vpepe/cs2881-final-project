import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_cosine_histories(folder_path):
    """Load all q*_cosine_history.jsonl files from a folder and compute average."""
    folder = Path(folder_path)

    # Dictionary to store cosine similarities at each eval_count
    eval_count_to_cosines = defaultdict(list)

    # Find all q*_cosine_history.jsonl files
    history_files = sorted(folder.glob("q*_cosine_history.jsonl"))

    for history_file in history_files:
        with open(history_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if data["eval_count"] < 21:
                        eval_count = data['eval_count']
                        cosine_sim = data['cosine_similarity']
                        eval_count_to_cosines[eval_count].append(cosine_sim)

    # Compute average at each eval_count
    eval_counts = sorted(eval_count_to_cosines.keys())
    avg_cosines = [np.mean(eval_count_to_cosines[ec]) for ec in eval_counts]

    return eval_counts, avg_cosines

def main():
    # Define folders to analyze
    folders = {
        'Surreal': 'gepa_question_evolution',
        'Debate': 'gepa_question_evolution_beta_debate',
        'Personal': 'gepa_question_evolution_beta_personal',
        'Quirky': 'gepa_question_evolution_beta_quirky'
    }

    # Set up the plot
    plt.figure(figsize=(12, 7))

    # Plot each trajectory
    for label, folder in folders.items():
        try:
            eval_counts, avg_cosines = load_cosine_histories(folder)
            plt.plot(eval_counts, avg_cosines, marker='o', markersize=4,
                    label=label, linewidth=2, alpha=0.8)
            print(f"{label}: Loaded {len(eval_counts)} eval steps")
        except Exception as e:
            print(f"Error loading {label} ({folder}): {e}")

    # Customize plot
    plt.xlabel('Iteration', fontsize=12)
    plt.xlim(0, 21)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    plt.title('Average Cosine Similarity Trajectories Over Question Evolution', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_file = 'avg_cosine_similarity_trajectories.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
