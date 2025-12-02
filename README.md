# Question Evolution with GEPA

Evolve questions to maximize disagreement between GPT-4.1 and Llama 3.2 3B using evolutionary optimization with LLM-based mutations.

## What This Does

This project uses **GEPA** (Genetic Evolution of Prompts via Adaptors) to evolve questions that cause maximum disagreement between two language models. The key innovation is using an **LLM as the mutation operator** instead of random text changes.

### The Mutation Model

The `reflection_lm` parameter (set to GPT-4.1) specifies which LLM performs the mutations. This model:

1. **Reads feedback** about why questions succeeded or failed
2. **Understands context** - sees actual responses from both models
3. **Reasons strategically** about how to increase disagreement
4. **Proposes intelligent variations** rather than random changes

Example evolution:
```
Gen 0: "What is the capital of France?"
       → Similarity: 0.95 ❌ (both models agree)

Gen 1: "Should France move its capital?"
       → Similarity: 0.78 ✓ (some disagreement)

Gen 2: "How would French history differ if Orleans was the capital?"
       → Similarity: 0.52 ✓✓ (good disagreement)

Gen 3: "Analyze medieval power dynamics if Reims was the Capetian seat..."
       → Similarity: 0.31 ✓✓✓ (excellent disagreement!)
```

The mutation model **learns** that factual questions → agreement, but counterfactual/complex questions → disagreement.

## Quick Start

```bash
# Install dependencies
pip install gepa openai scikit-learn numpy

# Set API keys
export OPENROUTER_API_KEY="your-key"  # For GPT-4.1 and Llama 3.2 3B
export OPENAI_API_KEY="your-key"      # For embeddings

# Run evolution
python evolve_questions.py

# Start fresh (delete previous state)
python evolve_questions.py --fresh
```

## How It Works

```
┌──────────────────────────────────────────┐
│  Question: "What is X?"                  │
└──────────────────────────────────────────┘
            ↓                    ↓
    ┌──────────┐        ┌──────────────┐
    │ GPT-4.1  │        │ Llama 3.2 3B │
    └──────────┘        └──────────────┘
            ↓                    ↓
     Response A           Response B
            ↓                    ↓
        ┌────────────────────────┐
        │  Embed & Compute       │
        │  Cosine Similarity     │
        └────────────────────────┘
                   ↓
         Score = 1 - similarity
                   ↓
        ┌────────────────────────┐
        │  Generate Feedback:    │
        │  "Models agreed too    │
        │   much, try X instead" │
        └────────────────────────┘
                   ↓
        ┌────────────────────────┐
        │  Mutation Model        │
        │  (GPT-4.1) reads       │
        │  feedback and proposes │
        │  strategic variations  │
        └────────────────────────┘
                   ↓
        Select best → Repeat
```

## Key Components

### 1. Custom GEPA Adapter (`QuestionEvolutionAdapter`)

Implements two key methods:

**`evaluate()`** - Evaluates questions on both models:
```python
def evaluate(self, batch, candidate, capture_traces):
    question = candidate["question"]
    gpt_response = get_model_response(question, "gpt-4.1")
    llama_response = get_model_response(question, "llama-3.2-3b")
    similarity = cosine_similarity(embed(gpt_response), embed(llama_response))
    score = 1.0 - similarity  # Higher = more disagreement
    return EvaluationBatch(outputs, scores, trajectories)
```

**`make_reflective_dataset()`** - Creates feedback for mutation model:
```python
def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
    # Extract trajectories and scores
    for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
        # Create rich feedback explaining why question succeeded/failed
        if score > 0.5:
            feedback = "GOOD: High disagreement. This question type works well."
        else:
            feedback = "POOR: Models agreed too much. Try more speculative questions."

        records.append({
            "Inputs": question,
            "Generated Outputs": gpt_response + llama_response,
            "Feedback": feedback,
        })

    return {component_name: records}  # Fed to mutation model
```

### 2. The Mutation Model (`reflection_lm`)

```python
result = optimize(
    seed_candidate={"question": "What is the capital of France?"},
    adapter=QuestionEvolutionAdapter(),
    reflection_lm="gpt-4.1",  # ⭐ The mutation model
    max_metric_calls=50,      # Budget: 50 evaluations
    ...
)
```

This LLM receives the feedback and strategically mutates questions to maximize disagreement.

## Why This Is Powerful

### vs Traditional Genetic Algorithms

**Traditional GA**: Random mutations
```
"What is X?" → "What is Y?" → "What is Z?" (random word swaps)
```

**GEPA**: Strategic LLM-based mutations
```
"What is X?"
→ Mutation model reads feedback: "Both models know basic facts equally"
→ Mutation model strategizes: "Try speculative or counterfactual questions"
→ "How would history change if X was Y instead?"
```

The mutation model **learns through feedback** what types of questions maximize disagreement.

## Customization

### Change Starting Question
Edit line 242 in `evolve_questions.py`:
```python
seed_candidate = {
    "question": "Your custom question here"
}
```

### Change Models Being Compared
Edit `get_model_response()` function to use different models.

### Change Mutation Model
Edit line 250:
```python
reflection_lm="claude-3-opus-20240229"  # Use Claude instead of GPT-4.1
```

### Adjust Evolution Parameters
Edit the `optimize()` call:
```python
optimize(
    reflection_lm="gpt-4.1",
    reflection_minibatch_size=2,      # Samples per mutation
    max_metric_calls=50,              # Total evaluation budget
    candidate_selection_strategy="current_best",  # Or "pareto"
    ...
)
```

## Applications

- **Model differentiation**: Find capability differences between models
- **Adversarial testing**: Discover edge cases where models disagree
- **Dataset generation**: Create challenging questions for benchmarks
- **Bias detection**: Identify topics with divergent model perspectives
- **Research**: Study how different architectures handle various question types

## Troubleshooting

### API Key Errors
Make sure both keys are set:
```bash
echo $OPENROUTER_API_KEY  # Should show your key
echo $OPENAI_API_KEY      # Should show your key (different from OpenRouter!)
```

Note: You need TWO different keys:
- `OPENROUTER_API_KEY` - For accessing GPT-4.1 and Llama 3.2 3B
- `OPENAI_API_KEY` - For embeddings (from platform.openai.com)

### Low Disagreement Scores
- Try different seed questions
- Increase `max_metric_calls` to allow more evolution
- Try a different `reflection_lm` model

### Cost Optimization
Reduce the budget:
```python
max_metric_calls=20           # Fewer evaluations
reflection_minibatch_size=1   # Fewer samples per reflection
```

## License

MIT
