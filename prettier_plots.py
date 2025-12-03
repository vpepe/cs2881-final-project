import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("premise_acceptance_all_files.json", "r") as f:
    premise_acceptance_data = json.load(f)

premise_data = pd.DataFrame()

for filename in premise_acceptance_data.keys():
    gpt_accepts_list = []
    llama_accepts_list = []
    data = premise_acceptance_data[filename]
    for dat in data:
        gpt_accepts_list.append(int(dat['gpt_accepts_premise']))
        llama_accepts_list.append(int(dat['llama_accepts_premise']))
    name = filename.replace("train_responses_", "").replace(".json", "")
    if name == "gepa":
        name = "surreal"
    premise_data = pd.concat([premise_data, pd.DataFrame({
        "filename": [name],
        "gpt_acceptance_rate": [np.mean(gpt_accepts_list)],
        "llama_acceptance_rate": [np.mean(llama_accepts_list)],
    })], ignore_index=True)
# plot the premise acceptance rates

import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(6, 6))

# sort by llama acceptance rate
premise_data_sorted = premise_data.sort_values(by='llama_acceptance_rate')

melted = premise_data_sorted.melt(id_vars=['filename'], value_vars=['gpt_acceptance_rate', 'llama_acceptance_rate'], var_name="model", value_name='acceptance_rate')
melted['filename'] = pd.Categorical(melted['filename'], categories=premise_data_sorted['filename'], ordered=True)
melted['model'] = melted['model'].map({'gpt_acceptance_rate': 'GPT-4', 'llama_acceptance_rate': 'LLaMA'})

ax = sns.barplot(data=melted, x='filename', y='acceptance_rate', hue='model', palette=['#2E86AB', '#A23B72'])

# Add error bars manually
plt.ylim(0, 1.2)
plt.xlabel('Seed Prompts', fontsize=12, fontweight='bold')
plt.ylabel('Premise Acceptance Rate', fontsize=12, fontweight='bold')
plt.title('Accepts Premise of Seed Question', fontsize=14, fontweight='bold', pad=20)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.legend(title='Model', title_fontsize=11, fontsize=10, frameon=True, shadow=True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("premise_acceptance_plots.png", dpi=300, bbox_inches='tight')
plt.show()

with open("questions_fantastical_classified.json", "r") as f:
    data = json.load(f)
for filename in premise_acceptance_data.keys():
    # get average fantastical score:
    fantastical_scores = []
    data_file = data[filename]
    for entry in data_file:
        fantastical_scores.append(int(entry['is_fantastical']))
    avg_fantastical_score = np.mean(fantastical_scores)
    name = filename.replace("train_responses_", "").replace(".json", "")
    if name == "gepa":
        name = "surreal"
    premise_data.loc[premise_data['filename'] == name, 'avg_fantastical_score'] = avg_fantastical_score
# plot the fantastical scores
sns.set_style("whitegrid")
plt.figure(figsize=(6, 6))
premise_data = premise_data.sort_values(by='avg_fantastical_score')
x = np.arange(len(premise_data['filename']))
plt.bar(x, premise_data['avg_fantastical_score'], color='#E63946', width=0.7, edgecolor='black', linewidth=1.2)
plt.xticks(x, premise_data['filename'], fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 1.2)
# yaxis percent
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.xlabel('Seed Prompts', fontsize=12, fontweight='bold')
plt.axhline(y=1, color='#457B9D', linestyle='--', linewidth=2, label='All questions fantastical')
plt.ylabel('Percent Labeled Fantastical', fontsize=12, fontweight='bold')
plt.title('Average Fantastical Score of Questions (per GPT-4 as Judge)', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=10, frameon=True, shadow=True)
plt.tight_layout()
plt.savefig("fantastical_score.png", dpi=300, bbox_inches='tight')
plt.show()