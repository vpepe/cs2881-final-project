"""
Evolve batches of questions to maximize cluster separation between
GPT-4.1 and Llama 3.2 3B responses using GEPA evolutionary optimization.

This version optimizes batches of n questions for classifier training,
scoring based on Euclidean distance between cluster centroids.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, TypedDict
from concurrent.futures import ThreadPoolExecutor
import openai
import numpy as np
from gepa import optimize, GEPAAdapter, EvaluationBatch


# Type definitions matching gepa's expectations
class QuestionDataInst(TypedDict):
    """A data instance containing a question to be evolved."""
    id: str
    question: str  # This is just for reference; actual questions come from candidate


class ClusterOutput(TypedDict):
    """Output containing cluster separation metrics."""
    questions: List[str]
    gpt_responses: List[str]
    llama_responses: List[str]
    gpt_centroid: List[float]
    llama_centroid: List[float]
    cluster_distance: float


class ClusterTrajectory(TypedDict):
    """Trajectory capturing execution details for reflection."""
    questions: List[str]
    gpt_responses_preview: List[str]  # Truncated
    llama_responses_preview: List[str]  # Truncated
    cluster_distance: float
    score: float


def get_model_response(question: str, model: str) -> str:
    """Get response from a specific model."""
    if model == "gpt-4.1":
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": question}],
            max_tokens=1000,
            temperature=1
        )
        return response.choices[0].message.content

    elif model == "llama-3.2-3b":
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        response = client.chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": question}],
            max_tokens=1000,
            temperature=1
        )
        return response.choices[0].message.content

    raise ValueError(f"Unknown model: {model}")


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text using OpenAI's embedding API via OpenRouter."""
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    response = client.embeddings.create(
        model="openai/text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding)


class QuestionBatchEvolutionAdapter(GEPAAdapter[QuestionDataInst, ClusterTrajectory, ClusterOutput]):
    """
    Custom GEPA adapter for evolving batches of questions to maximize cluster separation.

    The adapter evaluates question batches by:
    1. Getting responses from both GPT-4.1 and Llama 3.2 3B for all questions
    2. Computing embeddings for all responses
    3. Calculating cluster centroids for each model's responses
    4. Scoring based on Euclidean distance between centroids (higher = better)
    """

    def __init__(self, history_file: str = "candidates_history.jsonl"):
        """Initialize adapter with candidate history logging."""
        self.history_file = history_file
        self.eval_count = 0

    def evaluate(
        self,
        batch: List[QuestionDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ClusterTrajectory, ClusterOutput]:
        """
        Evaluate a candidate batch of questions.

        Args:
            batch: List of data instances to evaluate on
            candidate: Dict mapping component names to their text (e.g., {"question_0": "...", "question_1": "...", ...})
            capture_traces: Whether to capture execution trajectories for reflection

        Returns:
            EvaluationBatch containing outputs, scores, and optionally trajectories
        """
        outputs: List[ClusterOutput] = []
        scores: List[float] = []
        trajectories: List[ClusterTrajectory] | None = [] if capture_traces else None

        # Extract questions from candidate dict (question_0, question_1, ...)
        questions = [candidate[f"question_{i}"] for i in range(10)]
        n_questions = len(questions)

        print(f"\n{'='*80}")
        print(f"Evaluating batch of {n_questions} questions")
        print(f"{'='*80}")

        for data_inst in batch:
            # Collect all responses and embeddings in parallel
            gpt_responses = []
            llama_responses = []
            gpt_embeddings = []
            llama_embeddings = []

            def process_question(i_question_tuple):
                """Process a single question: get responses from both models and embeddings."""
                i, question = i_question_tuple
                print(f"\nQuestion {i+1}/{n_questions}: {question[:100]}...")

                # Get responses from both models
                gpt_response = get_model_response(question, "gpt-4.1")
                llama_response = get_model_response(question, "llama-3.2-3b")

                # Get embeddings
                gpt_embedding = get_embedding(gpt_response)
                llama_embedding = get_embedding(llama_response)

                print(f"  GPT-4.1: {gpt_response[:100]}...")
                print(f"  Llama 3.2: {llama_response[:100]}...")

                return gpt_response, llama_response, gpt_embedding, llama_embedding

            # Process all questions in parallel with 64 workers
            with ThreadPoolExecutor(max_workers=64) as executor:
                results = list(executor.map(process_question, enumerate(questions)))

            # Unpack results
            for gpt_response, llama_response, gpt_embedding, llama_embedding in results:
                gpt_responses.append(gpt_response)
                llama_responses.append(llama_response)
                gpt_embeddings.append(gpt_embedding)
                llama_embeddings.append(llama_embedding)

            # Convert to numpy arrays for calculations
            gpt_embeddings_array = np.array(gpt_embeddings)  # shape: (n_questions, embedding_dim)
            llama_embeddings_array = np.array(llama_embeddings)

            # Calculate cluster centroids (mean of all embeddings)
            gpt_centroid = np.mean(gpt_embeddings_array, axis=0)
            llama_centroid = np.mean(llama_embeddings_array, axis=0)

            # Calculate Euclidean distance between centroids
            cluster_distance = np.linalg.norm(gpt_centroid - llama_centroid)

            # Score is the cluster distance (higher = better separation)
            score = float(cluster_distance)

            output: ClusterOutput = {
                "questions": questions,
                "gpt_responses": gpt_responses,
                "llama_responses": llama_responses,
                "gpt_centroid": gpt_centroid.tolist(),
                "llama_centroid": llama_centroid.tolist(),
                "cluster_distance": cluster_distance,
            }

            outputs.append(output)
            scores.append(score)

            # Create trajectory for reflection
            if trajectories is not None:
                trajectory: ClusterTrajectory = {
                    "questions": questions,
                    "gpt_responses_preview": [r[:200] for r in gpt_responses],
                    "llama_responses_preview": [r[:200] for r in llama_responses],
                    "cluster_distance": cluster_distance,
                    "score": score,
                }
                trajectories.append(trajectory)

            print(f"\n{'='*80}")
            print(f"Cluster distance (score): {cluster_distance:.4f}")
            print(f"{'='*80}")

        # Log candidate to history file
        self._log_candidate(candidate, scores[0], outputs[0])

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def _log_candidate(self, candidate: Dict[str, str], score: float, output: ClusterOutput):
        """Log candidate to JSONL history file."""
        self.eval_count += 1

        log_entry = {
            "eval_count": self.eval_count,
            "timestamp": datetime.now().isoformat(),
            "candidate": candidate,
            "score": score,
            "cluster_distance": output["cluster_distance"],
        }

        with open(self.history_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[ClusterTrajectory, ClusterOutput],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract textual feedback for the reflection LM to use when mutating.

        This tells the mutation model how well the current batch performs
        at creating cluster separation, so it can evolve better questions.

        Args:
            candidate: Current candidate text mapping (question_0, question_1, ...)
            eval_batch: Evaluation results containing trajectories and scores
            components_to_update: List of component names to update

        Returns:
            Dictionary mapping component names to lists of reflection records
        """
        assert eval_batch.trajectories is not None, "Trajectories required for reflection"

        result: Dict[str, List[Dict[str, str]]] = {}

        for component_name in components_to_update:
            records = []

            for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
                cluster_distance = traj["cluster_distance"]
                questions = traj["questions"]
                gpt_responses = traj["gpt_responses_preview"]
                llama_responses = traj["llama_responses_preview"]

                # Calculate individual question contributions to cluster separation
                # (Use embedding distances as proxy for which questions diverge most)
                individual_divergences = []
                for i in range(len(questions)):
                    gpt_r = gpt_responses[i]
                    llama_r = llama_responses[i]
                    # Simple heuristic: longer response differences suggest more divergence
                    length_diff = abs(len(gpt_r) - len(llama_r))
                    individual_divergences.append((i, length_diff))

                # Sort by divergence (descending)
                individual_divergences.sort(key=lambda x: x[1], reverse=True)
                most_divergent_idx = individual_divergences[0][0] if individual_divergences else 0
                least_divergent_idx = individual_divergences[-1][0] if individual_divergences else 0

                # Create detailed feedback explaining performance
                if score > 50.0:  # Good separation
                    feedback = (
                        f"EXCELLENT: Strong cluster separation (distance: {cluster_distance:.4f}). "
                        f"This batch creates very distinct response patterns between models.\n\n"
                        f"WHAT WORKED: Question {most_divergent_idx + 1} appears most effective - "
                        f"GPT-4.1 and Llama responded very differently (length diff: {individual_divergences[0][1]} chars). "
                        f"Questions that are grammatically unusual or contain rare word combinations tend to work well.\n\n"
                        f"KEEP: Continue using OOD patterns like this batch."
                    )
                elif score > 30.0:  # Moderate separation
                    feedback = (
                        f"GOOD: Moderate cluster separation (distance: {cluster_distance:.4f}). "
                        f"Questions create some distinction, but could be more diverse.\n\n"
                        f"WHAT WORKED: Question {most_divergent_idx + 1} showed good divergence (length diff: {individual_divergences[0][1]} chars). "
                        f"Question {least_divergent_idx + 1} had similar responses (length diff: {individual_divergences[-1][1]} chars).\n\n"
                        f"IMPROVE: Push questions further out-of-distribution. Use more unusual syntax, rare bigrams, or semantic ambiguity."
                    )
                elif score > 15.0:  # Weak separation
                    feedback = (
                        f"FAIR: Weak cluster separation (distance: {cluster_distance:.4f}). "
                        f"Models respond somewhat similarly.\n\n"
                        f"PROBLEMS: Even the most divergent question (Q{most_divergent_idx + 1}) only achieved "
                        f"{individual_divergences[0][1]} char difference. Most questions got similar-length responses.\n\n"
                        f"TRY: More varied question types - mix different grammatical structures, use borderline-nonsensical "
                        f"phrases that could be interpreted multiple ways, combine unrelated semantic domains."
                    )
                else:  # Poor separation
                    feedback = (
                        f"POOR: Very weak cluster separation (distance: {cluster_distance:.4f}). "
                        f"Models respond very similarly across all questions.\n\n"
                        f"PROBLEMS: All questions yielded similar responses from both models. "
                        f"Largest response difference was only {individual_divergences[0][1]} chars (Q{most_divergent_idx + 1}).\n\n"
                        f"CRITICAL: Need radically different questions. Current patterns are too in-distribution. "
                        f"Try: unusual word sequences, grammatical edge cases, extremely rare vocabulary, "
                        f"or prompts that could trigger different training data coverage between models."
                    )

                # Show all 10 questions with their responses for context
                all_qa = []
                for i, (q, gpt_r, llama_r) in enumerate(zip(questions, gpt_responses, llama_responses)):
                    divergence_marker = "✓ DIVERGENT" if i == most_divergent_idx else ("✗ SIMILAR" if i == least_divergent_idx else "")
                    all_qa.append(
                        f"Q{i+1} {divergence_marker}: {q[:150]}...\n"
                        f"  GPT-4.1 ({len(gpt_r)} chars): {gpt_r}\n"
                        f"  Llama ({len(llama_r)} chars): {llama_r}"
                    )

                record = {
                    "Inputs": f"Current batch performance (overall cluster distance: {cluster_distance:.4f}):\n\n" + "\n\n".join(all_qa),
                    "Generated Outputs": (
                        f"Cluster separation: {cluster_distance:.4f}\n"
                        f"Most divergent question: Q{most_divergent_idx + 1} (diff: {individual_divergences[0][1]})\n"
                        f"Least divergent question: Q{least_divergent_idx + 1} (diff: {individual_divergences[-1][1]})"
                    ),
                    "Feedback": feedback,
                }
                records.append(record)

            result[component_name] = records

        return result


def main():
    import sys

    print("Starting GEPA evolutionary optimization for question batches")
    print("Goal: Evolve batches of questions to maximize cluster separation")
    print("Metric: Euclidean distance between response cluster centroids")
    print("=" * 80)

    # Configuration
    N_QUESTIONS = 10  # Number of questions per batch

    # Check if we should start fresh
    run_dir = "./gepa_question_evolution_beta"
    if "--fresh" in sys.argv:
        import shutil
        if os.path.exists(run_dir):
            print(f"\nRemoving existing run directory: {run_dir}")
            shutil.rmtree(run_dir)
            print("Starting fresh optimization...\n")
        # Also clear history file
        history_file = "candidates_history.jsonl"
        if os.path.exists(history_file):
            os.remove(history_file)
            print(f"Cleared history file: {history_file}\n")
    elif os.path.exists(run_dir):
        print(f"\nFound existing run directory: {run_dir}")
        print("Will resume from previous state. Use --fresh to start over.\n")

    # Initial seed batch of questions - stored as separate components for GEPA
    seed_questions = [
        "I need you to gather the bun, cultured apparel, the sack with the shoe and shade, plus the jaded and amazing fluffy items, then check the wound, count the massive cash, note any disgust or afterthought in the behavior, recast the drawer contents, deal with the cowardly turkey and the brainy music setup, make sure nothing greedy encroaches on the house, and finally have the train come without causing jealous crushes or divergent trouble.",
        "Assemble the crisp lantern, the philosophical blanket, the pocket stuffed with marbles and moonlight, along with the timid sparkling stones, review the minor fracture, tally the towering coins, observe any boredom or leftover hesitation, reorganize the cabinet's secret stash, soothe the panicked goose and the logical drum kit, guard the gate from any ravenous intruders, and summon the bus without stirring envious clashes or chaotic spirals.",
        "Collect the toasted spiral, scholarly boots, the pouch holding feathers and echoes, plus the weary yet magnificent drifting clouds, inspect the tiny scrape, measure the overflowing funds, detect any annoyance or lingering doubt, rewrite the contents of the hidden crate, calm the frightened pigeon and the calculating piano rig, keep greedy shadows away from the yard, and guide the ship in smoothly without triggering jealous sparks or wandering mishaps.",
        "Gather the honeyed cube, enlightened jacket, the tote with pebbles and starlight, and the cranky but glorious fuzzy clusters, check the shallow bruise, count the heaping gold, sense any disdain or reflective pause, retune the dresser clutter, wrangle the bashful rooster and the analytical speaker system, block any grasping vines from the cottage, and let the carriage roll in without causing possessive flurries or tangled chaos.",
        "Bring together the savory coil, mindful gloves, the duffel stuffed with whistles and dusk, plus the moody radiant plush forms, examine the fresh nick, assess the mountain of bills, notice any revulsion or lingering second guesses, reshape the drawer's odd relics, cheer up the jittery quail and the cerebral guitar array, prevent any covetous critters from slipping into the porch, and signal the tram to arrive with no jealous storms or diverging complications.",
        "Fetch the buttered ring, poetic trousers, the knapsack of yarn and twilight, along with the cynical yet marvelous soft spheres, inspect the healing cut, inventory the colossal stash, track any disgust or delayed remorse, overhaul the chest of trinkets, shoo the faint-hearted duck and the genius violin stand, shield the barn from anything graspingly territorial, and usher the wagon in without igniting jealous tangles or branching confusion.",
        "Round up the spiced wheel, academic scarf, the satchel brimming with shells and duskfall, plus the irritable but brilliant fuzzy strands, review the tender scrape, compute the bulging currency, observe any squeamishness or hesitant reflection, reframe the compartment odds and ends, steady the timid pheasant and the brainy drum cluster, protect the homestead from greedy drifters, and cue the shuttle to glide in without planting jealous frictions or stray turmoil.",
        "Collect the roasted knot, philosophical shoes, the bag filled with chalk and shadows, and the mellow astonished cotton tufts, examine the shallow cut, add up the enormous bankroll, gauge any disgust or wavering contemplation, restore the drawer's forgotten gadgets, console the cowardly swan and the clever horn ensemble, keep any avaricious pests out of the bungalow, and let the streetcar approach without sparking jealous collisions or drifting errors.",
        "Assemble the caramel loop, introspective coat, the sling of crystals and breeze, plus the grudging yet splendid pillowy bundles, check the mild abrasion, tally the vast loot, pick up on any repulsion or lingering reconsideration, curate the crate's eccentric contents, placate the skittish turkey and the ingenious synth board, ward off all greedy creepers from the cottage entry, and wave the locomotive in without producing jealous cliques or branching confusion.",
        "Gather the toasted ringlet, literary vest, the parcel of magnets and haze, and the sulky but awe-struck fluff clusters, inspect the faint wound, total the massive treasure, sense any disgust or delayed reflection, remix the drawer's odd materials, reassure the nervous grouse and the intellectual amp setup, ensure no greedy forces seep into the manor, and have the tram roll in without sparking jealous crushes or divergent uproar."
    ]

    # Store as separate string components for GEPA compatibility
    seed_candidate = {f"question_{i}": q for i, q in enumerate(seed_questions)}

    # Create dummy training data
    trainset: List[QuestionDataInst] = [
        {"id": "eval1", "question": "placeholder"},
    ]

    # Create adapter
    adapter = QuestionBatchEvolutionAdapter(history_file="candidates_history.jsonl")

    # Reflection prompt (same as original, generates 1 question per call)
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
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,
        adapter=adapter,
        reflection_lm="openai/gpt-4.1",
        reflection_prompt_template=reflection_prompt,
        candidate_selection_strategy="current_best",
        reflection_minibatch_size=1,  # Mutate after every evaluation
        max_metric_calls=20,  # 20 rounds of evolution
        module_selector="all",
        display_progress_bar=True,
        run_dir=run_dir,
        seed=42,
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nBest batch of questions found:")
    for i in range(10):
        print(f"  {i+1}. {result.best_candidate[f'question_{i}']}")

    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest cluster separation score: {best_score:.4f}")
    print(f"(Higher = better separation between GPT-4.1 and Llama responses)")
    print(f"\nTotal candidates explored: {result.num_candidates}")
    print(f"Total metric calls: {result.total_metric_calls}")

    # Save final results
    final_output_file = "best_questions_batch.json"
    with open(final_output_file, "w") as f:
        json.dump({
            "best_candidate": result.best_candidate,
            "best_score": best_score,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
        }, f, indent=2)
    print(f"\nBest batch saved to: {final_output_file}")
    print(f"Full history saved to: candidates_history.jsonl")


if __name__ == "__main__":
    main()
