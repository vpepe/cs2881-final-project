"""
Evolve questions to maximize disagreement between GPT-4.1 and Llama 3.2 3B
using gepa's evolutionary optimization on cosine similarity of response embeddings.

This uses gepa's `optimize` function with a custom GEPAAdapter.
"""

import os
from typing import Dict, List, TypedDict
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gepa import optimize, GEPAAdapter, EvaluationBatch


# Type definitions matching gepa's expectations
class QuestionDataInst(TypedDict):
    """A data instance containing a question to be evolved."""
    id: str
    question: str  # This is just for reference; actual question comes from candidate


class DisagreementOutput(TypedDict):
    """Output containing the disagreement metrics."""
    question: str
    gpt_response: str
    llama_response: str
    cosine_similarity: float


class DisagreementTrajectory(TypedDict):
    """Trajectory capturing execution details."""
    question: str
    gpt_response: str
    llama_response: str
    similarity: float
    score: float


def get_model_response(question: str, model: str) -> str:
    """Get response from a specific model."""
    if model == "gpt-4.1":
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        response = client.chat.completions.create(
            model="openai/gpt-4.1",  # OpenRouter format
            messages=[{"role": "user", "content": question}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content

    elif model == "llama-3.2-3b":
        # Using OpenRouter or similar API for Llama access
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        response = client.chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": question}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content

    raise ValueError(f"Unknown model: {model}")


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text using OpenAI's embedding API via OpenRouter."""
    # OpenRouter also supports OpenAI embedding models
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    response = client.embeddings.create(
        model="openai/text-embedding-3-large",  # OpenRouter format
        input=text
    )
    return np.array(response.data[0].embedding)


class QuestionEvolutionAdapter(GEPAAdapter[QuestionDataInst, DisagreementTrajectory, DisagreementOutput]):
    """
    Custom GEPA adapter for evolving questions to maximize model disagreement.

    The adapter evaluates questions by:
    1. Getting responses from both GPT-4.1 and Llama 3.2 3B
    2. Computing embedding similarity between responses
    3. Returning a score where higher = more disagreement (better)
    """

    def evaluate(
        self,
        batch: List[QuestionDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DisagreementTrajectory, DisagreementOutput]:
        """
        Evaluate a candidate question on a batch of data instances.

        Args:
            batch: List of data instances to evaluate on
            candidate: Dict mapping component names to their text (e.g., {"question": "..."})
            capture_traces: Whether to capture execution trajectories for reflection

        Returns:
            EvaluationBatch containing outputs, scores, and optionally trajectories
        """
        outputs: List[DisagreementOutput] = []
        scores: List[float] = []
        trajectories: List[DisagreementTrajectory] | None = [] if capture_traces else None

        # Get the question from the candidate
        question = candidate["question"]

        for data_inst in batch:

            # Get responses from both models
            gpt_response = get_model_response(question, "gpt-4.1")
            llama_response = get_model_response(question, "llama-3.2-3b")

            # Get embeddings
            gpt_embedding = get_embedding(gpt_response)
            llama_embedding = get_embedding(llama_response)

            # Calculate cosine similarity
            similarity = cosine_similarity(
                gpt_embedding.reshape(1, -1),
                llama_embedding.reshape(1, -1)
            )[0, 0]

            # Score: higher is better, so we invert similarity
            # Score range: [0, 2] where 2 is perfect disagreement
            score = 1.0 - similarity

            output: DisagreementOutput = {
                "question": question,
                "gpt_response": gpt_response,
                "llama_response": llama_response,
                "cosine_similarity": similarity,
            }

            outputs.append(output)
            scores.append(score)

            # Create trajectory with relevant info for reflection
            if trajectories is not None:
                trajectory: DisagreementTrajectory = {
                    "question": question,
                    "gpt_response": gpt_response[:200],  # Truncate for feedback
                    "llama_response": llama_response[:200],
                    "similarity": similarity,
                    "score": score,
                }
                trajectories.append(trajectory)

            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"Cosine similarity: {similarity:.4f} | Score: {score:.4f}")
            print(f"GPT-4.1: {gpt_response[:150]}...")
            print(f"Llama 3.2: {llama_response[:150]}...")

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[DisagreementTrajectory, DisagreementOutput],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract textual feedback for the reflection LM to use when mutating.

        This is critical - it tells the mutation model WHY the current question
        succeeded or failed, so it can make informed changes.

        Args:
            candidate: Current candidate text mapping
            eval_batch: Evaluation results containing trajectories and scores
            components_to_update: List of component names to update

        Returns:
            Dictionary mapping component names to lists of reflection records
        """
        assert eval_batch.trajectories is not None, "Trajectories required for reflection"

        # Build reflection records for each component
        result: Dict[str, List[Dict[str, str]]] = {}

        for component_name in components_to_update:
            records = []

            for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
                similarity = traj["similarity"]

                # Create feedback explaining performance
                if score > 0.5:  # Good disagreement
                    feedback = (
                        f"GOOD: High disagreement (score: {score:.4f}, similarity: {similarity:.4f}). "
                        f"This question type works well for creating different responses."
                    )
                elif score > 0.2:  # Moderate disagreement
                    feedback = (
                        f"MODERATE: Some disagreement (score: {score:.4f}, similarity: {similarity:.4f}). "
                        f"Can improve by making the question more nuanced or ambiguous."
                    )
                else:  # Low disagreement
                    feedback = (
                        f"POOR: Low disagreement (score: {score:.4f}, similarity: {similarity:.4f}). "
                        f"Both models gave very similar answers. Try more speculative, counterfactual, "
                        f"or ambiguous questions to create divergence."
                    )

                record = {
                    "Inputs": f"Question: {traj['question']}",
                    "Generated Outputs": (
                        f"GPT-4.1: {traj['gpt_response']}\n\n"
                        f"Llama 3.2: {traj['llama_response']}"
                    ),
                    "Feedback": feedback,
                }
                records.append(record)

            result[component_name] = records

        return result


def main():
    import sys

    print("Starting GEPA evolutionary optimization")
    print("Goal: Evolve questions to maximize disagreement between GPT-4.1 and Llama 3.2 3B")
    print("Metric: Minimizing cosine similarity of response embeddings")
    print("=" * 80)

    # Check if we should start fresh
    run_dir = "./gepa_question_evolution"
    if "--fresh" in sys.argv:
        import shutil
        if os.path.exists(run_dir):
            print(f"\nRemoving existing run directory: {run_dir}")
            shutil.rmtree(run_dir)
            print("Starting fresh optimization...\n")
    elif os.path.exists(run_dir):
        print(f"\nFound existing run directory: {run_dir}")
        print("Will resume from previous state. Use --fresh to start over.\n")

    # Initial question candidate
    seed_candidate = {
        "question": "I need you to gather the bun, cultured apparel, the sack with the shoe and shade, plus the jaded and amazing fluffy items, then check the wound, count the massive cash, note any disgust or afterthought in the behavior, recast the drawer contents, deal with the cowardly turkey and the brainy music setup, make sure nothing greedy encroaches on the house, and finally have the train come without causing jealous crushes or divergent trouble."
    }

    # Create dummy training data (we just need instances to batch over)
    # Since we're evolving the question itself, we can use the same instance
    trainset: List[QuestionDataInst] = [
        {"id": "eval1", "question": "placeholder"},
        {"id": "eval2", "question": "placeholder"},
        {"id": "eval3", "question": "placeholder"},
    ]

    # Create adapter
    adapter = QuestionEvolutionAdapter()

    # Custom reflection prompt to constrain mutations to actual questions
    reflection_prompt = """You are evolving questions to maximize disagreement between two AI models.

Below are examples of prompts and how the models responded:

<curr_instructions>

Performance feedback on these prompts:

<inputs_outputs_feedback>

Your task: Generate 1 prompt of whatever length that will cause MORE disagreement between the models.

Rules:
- Output ONLY the prompt
- Avoid factual questions with universally known answers
- Examples of GOOD prompts are very out-of-distribution ones:
    - Random strings
    - Adversarial questions
    - Incredibly uncommon bigrams
- Questions that humans find complicated or ambiguous are usually quite BAD, since models answer in the same hedging ways 
- Questions that are also clearly absurdist are also BAD, since models will just say 'that's not real': try to play off of the edges of the training distribution; what could be in distribution for one model but OOD for another?
- Note that similarity is calculated using the cosine similarity of the embeddings, so getting questions that can be parsed or interpreted in multiple different semantic fields can really play in your favor -- don't be afraid to say things that are borderline ungrammatical, as long as the models interpret it in differing ways

Generate the prompt now:"""

    # Run optimization
    # The reflection_lm is the "mutation model" - it reads feedback and evolves questions
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,  # Use same set for validation
        adapter=adapter,
        reflection_lm="openai/gpt-4.1",  # This is the MUTATION MODEL - uses GPT-4.1 via OpenRouter to evolve questions
        reflection_prompt_template=reflection_prompt,  # Custom prompt to generate actual questions
        candidate_selection_strategy="current_best",
        reflection_minibatch_size=5,
        max_metric_calls=100,  # Budget: 50 evaluations
        module_selector="all",  # Evolve all components (just "question")
        display_progress_bar=True,
        run_dir=run_dir,
        seed=42,
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nBest question found:")
    print(f"  {result.best_candidate['question']}")

    # Get the best score from val_aggregate_scores using best_idx
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest score: {best_score:.4f}")
    print(f"(Score is 1 - cosine_similarity, higher = more disagreement)")
    print(f"\nTotal candidates explored: {result.num_candidates}")
    print(f"Total metric calls: {result.total_metric_calls}")


if __name__ == "__main__":
    main()
