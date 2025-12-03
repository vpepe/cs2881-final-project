import matplotlib.pyplot as plt
import json

# Load test accuracies from each metrics file
metrics_files = {
    'Baseline': 'classifier_metrics.json',
    'Debate': 'classifier_metrics_debate.json',
    'Surreal': 'classifier_metrics_gepa.json',
    'Personal': 'classifier_metrics_personal.json',
    'Quirky': 'classifier_metrics_quirky.json'
}

test_accuracies = {}
for name, file_path in metrics_files.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
        test_accuracies[name] = data['test_accuracy']

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
models = list(test_accuracies.keys())
accuracies = list(test_accuracies.values())

bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=10)

ax.set_xlabel('GEPA seed prompt', fontsize=12)
ax.set_ylabel('Test accuracy', fontsize=12)
ax.set_title('Classifier accuracy comparison across seed prompts', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Bar chart saved as 'test_accuracy_comparison.png'")
plt.show()
