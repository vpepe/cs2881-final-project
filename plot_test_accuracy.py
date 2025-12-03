import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from scipy import stats

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

def wilson_confidence_interval(p, n, alpha=0.05):
    """
    Calculate Wilson score confidence interval for a proportion.
    More accurate than normal approximation, especially for extreme proportions.
    """
    z = stats.norm.ppf(1 - alpha/2)
    denominator = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denominator
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    return centre - spread, centre + spread

# Load test accuracies from each metrics file
metrics_files = {
    'Baseline': 'classifier_metrics.json',
    'Debate': 'classifier_metrics_debate.json',
    'Surreal': 'classifier_metrics_gepa.json',
    'Personal': 'classifier_metrics_personal.json',
    'Quirky': 'classifier_metrics_quirky.json'
}

test_accuracies = {}
confidence_intervals = {}
n_test_samples = 100  # From confusion matrix

for name, file_path in metrics_files.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
        acc = data['test_accuracy']
        test_accuracies[name] = acc

        # Calculate 95% CI using Wilson score interval
        ci_lower, ci_upper = wilson_confidence_interval(acc, n_test_samples)
        confidence_intervals[name] = (acc - ci_lower, ci_upper - acc)

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
models = list(test_accuracies.keys())
accuracies = list(test_accuracies.values())

# Extract error bars (lower, upper)
error_bars = np.array([confidence_intervals[name] for name in models]).T

# Color scheme: gray for baseline, purple gradient for GEPA conditions
colors = ['#7f8c8d']  # Gray for Baseline
gepa_colors = sns.color_palette("viridis", n_colors=4)  # Purple/blue gradient for GEPA conditions
colors.extend(gepa_colors)

bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85,
              yerr=error_bars, capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2, 'alpha': 0.7})

# Add chance performance line at 50%
ax.axhline(y=0.5, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label='Chance (50%)')

# Add value labels inside bars at the bottom
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., 0.52,  # Just above the 50% line
            f'{height:.1%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

ax.set_xlabel('GEPA seed prompt', fontsize=13)
ax.set_ylabel('Test accuracy', fontsize=13)
ax.set_title('Classifier accuracy comparison across seed prompts', fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0.5, 1.0)  # Start at chance performance (50%)
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Bar chart saved as 'test_accuracy_comparison.png'")
plt.show()
