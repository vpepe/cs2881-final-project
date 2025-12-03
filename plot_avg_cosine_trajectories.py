import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

def load_cosine_histories(folder_path):
    """Load all q*_cosine_history.jsonl files from a folder and compute average and std."""
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

    # Compute average and standard error at each eval_count
    eval_counts = sorted(eval_count_to_cosines.keys())
    avg_cosines = [np.mean(eval_count_to_cosines[ec]) for ec in eval_counts]
    # Standard error of the mean for error bars (95% CI ~ 1.96 * SEM)
    sem_cosines = [1.96 * np.std(eval_count_to_cosines[ec]) / np.sqrt(len(eval_count_to_cosines[ec]))
                   for ec in eval_counts]

    return eval_counts, avg_cosines, sem_cosines

def get_baseline_cosine(folder_path):
    """Get the average baseline (eval_count=1) cosine similarity."""
    folder = Path(folder_path)
    history_files = sorted(folder.glob("q*_cosine_history.jsonl"))

    baseline_cosines = []
    for history_file in history_files:
        with open(history_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                if data["eval_count"] == 1:
                    baseline_cosines.append(data['cosine_similarity'])

    return np.mean(baseline_cosines) if baseline_cosines else None

def main():
    # Define folders to analyze
    folders = {
        'Debate': 'gepa_question_evolution_beta_debate',
        'Surreal': 'gepa_question_evolution',
        'Personal': 'gepa_question_evolution_beta_personal',
        'Quirky': 'gepa_question_evolution_beta_quirky',
    }

    # Color scheme matching the bar chart (only need 2 colors)
    gepa_colors = sns.color_palette("viridis", n_colors=4)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get baseline cosine similarity from baseline_questions.json calculation
    try:
        with open('baseline_cosine_similarity.json', 'r') as f:
            baseline_data = json.load(f)
            baseline_cosine = baseline_data['average_cosine_similarity']
            ax.axhline(y=baseline_cosine, color='#7f8c8d', linestyle='--',
                       linewidth=2.5, alpha=0.8, label=f'Baseline ({baseline_cosine:.3f})')
    except FileNotFoundError:
        print("Warning: baseline_cosine_similarity.json not found. Run calculate_baseline_cosine.py first.")

    # Plot each trajectory with matching colors
    for idx, (label, folder) in enumerate(folders.items()):
        try:
            eval_counts, avg_cosines, sem_cosines = load_cosine_histories(folder)

            # Plot line with error bars
            ax.plot(eval_counts, avg_cosines, marker='o', markersize=6,
                   label=label, linewidth=2.5, alpha=0.85, color=gepa_colors[idx])

            # Add 95% confidence interval shading
            ax.fill_between(eval_counts,
                           np.array(avg_cosines) - np.array(sem_cosines),
                           np.array(avg_cosines) + np.array(sem_cosines),
                           alpha=0.2, color=gepa_colors[idx])

            print(f"{label}: Loaded {len(eval_counts)} eval steps")
        except Exception as e:
            print(f"Error loading {label} ({folder}): {e}")

    # Customize plot
    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_xlim(0, 21)
    ax.set_ylabel('Average Cosine Similarity', fontsize=13)
    ax.set_title('Average Cosine Similarity Trajectories Over Question Evolution',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_file = 'avg_cosine_similarity_trajectories.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
