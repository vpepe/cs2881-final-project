"""
Massively parallel question evolution: Each GEPA instance evolves ONE question.

This version is designed for maximum parallelizability:
- Each GEPA instance evolves a single question (not a batch)
- Multiple instances can run concurrently across multiple processes/machines
- Supports both cosine and cluster objectives
- Each instance writes to its own log file to avoid conflicts

Usage:
  # Launch N parallel instances with single command
  python evolve_questions_beta.py --num-questions 20 --max-rounds 20

  # Or run a single instance
  python evolve_questions_beta.py --question-id 0 --objective cosine --max-rounds 20
"""

import os
import json
from datetime import datetime
from typing import Dict, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gepa import optimize, GEPAAdapter, EvaluationBatch
import subprocess
import sys


# Type definitions
class QuestionDataInst(TypedDict):
    """A data instance containing a question to be evolved."""
    id: str
    question: str


class DisagreementOutput(TypedDict):
    """Output containing the disagreement metrics for a single question."""
    question: str
    gpt_response: str
    llama_response: str
    cosine_similarity: float
    gpt_embedding: List[float]
    llama_embedding: List[float]
    objective: str


class DisagreementTrajectory(TypedDict):
    """Trajectory capturing execution details for a single question."""
    question: str
    gpt_response_preview: str
    llama_response_preview: str
    cosine_similarity: float
    score: float
    objective: str


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


class SingleQuestionAdapter(GEPAAdapter[QuestionDataInst, DisagreementTrajectory, DisagreementOutput]):
    """
    GEPA adapter for evolving a SINGLE question to maximize model disagreement.

    This adapter is designed for maximum parallelizability - each instance evolves
    one question independently. Multiple instances can run concurrently.

    Supports two objectives:
    - cosine: Minimize cosine similarity of response embeddings
    - cluster: Maximize embedding distance (used as proxy for cluster separation)
    """

    def __init__(self, objective: str = "cosine", history_file: str = "question_history.jsonl"):
        """
        Initialize adapter.

        Args:
            objective: Either "cosine" or "cluster"
            history_file: Path to JSONL file for logging candidates
        """
        self.objective = objective
        self.history_file = history_file
        self.eval_count = 0

    def evaluate(
        self,
        batch: List[QuestionDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DisagreementTrajectory, DisagreementOutput]:
        """
        Evaluate a single question candidate.

        Args:
            batch: List of data instances (we only use the first one)
            candidate: Dict with single key "question" containing the question text
            capture_traces: Whether to capture execution trajectories

        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories
        """
        outputs: List[DisagreementOutput] = []
        scores: List[float] = []
        trajectories: List[DisagreementTrajectory] | None = [] if capture_traces else None

        # Extract the single question
        question = candidate["question"]

        print(f"\n{'='*80}")
        print(f"Evaluating question (objective: {self.objective})")
        print(f"Question: {question}")
        print(f"{'='*80}")

        for data_inst in batch:
            # Get responses from both models
            gpt_response = get_model_response(question, "gpt-4.1")
            llama_response = get_model_response(question, "llama-3.2-3b")

            # Get embeddings
            gpt_embedding = get_embedding(gpt_response)
            llama_embedding = get_embedding(llama_response)

            # Calculate cosine similarity
            cosine_sim = cosine_similarity(
                gpt_embedding.reshape(1, -1),
                llama_embedding.reshape(1, -1)
            )[0, 0]

            print(f"\nGPT-4.1 response ({len(gpt_response)} chars):")
            print(f"  {gpt_response[:200]}...")
            print(f"\nLlama 3.2 response ({len(llama_response)} chars):")
            print(f"  {llama_response[:200]}...")
            print(f"\nCosine similarity: {cosine_sim:.4f}")

            # Calculate score based on objective
            if self.objective == "cosine":
                # Score: 1 - cosine_similarity (higher = more disagreement)
                score = float(1.0 - cosine_sim)
            else:  # cluster objective
                # For single questions, use Euclidean distance between embeddings
                # as proxy for cluster separation
                distance = float(np.linalg.norm(gpt_embedding - llama_embedding))
                score = distance

            output: DisagreementOutput = {
                "question": question,
                "gpt_response": gpt_response,
                "llama_response": llama_response,
                "cosine_similarity": float(cosine_sim),
                "gpt_embedding": gpt_embedding.tolist(),
                "llama_embedding": llama_embedding.tolist(),
                "objective": self.objective,
            }

            outputs.append(output)
            scores.append(score)

            # Create trajectory for reflection
            if trajectories is not None:
                trajectory: DisagreementTrajectory = {
                    "question": question,
                    "gpt_response_preview": gpt_response[:300],
                    "llama_response_preview": llama_response[:300],
                    "cosine_similarity": float(cosine_sim),
                    "score": score,
                    "objective": self.objective,
                }
                trajectories.append(trajectory)

            print(f"\n{'='*80}")
            if self.objective == "cosine":
                print(f"Cosine similarity: {cosine_sim:.4f}")
                print(f"Score (1 - similarity): {score:.4f}")
            else:
                print(f"Embedding distance (score): {score:.4f}")
            print(f"{'='*80}")

        # Log candidate to history file
        self._log_candidate(candidate, scores[0], outputs[0])

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def _log_candidate(self, candidate: Dict[str, str], score: float, output: DisagreementOutput):
        """Log candidate to JSONL history file."""
        self.eval_count += 1

        log_entry = {
            "eval_count": self.eval_count,
            "timestamp": datetime.now().isoformat(),
            "candidate": candidate,
            "score": score,
            "objective": self.objective,
            "cosine_similarity": output["cosine_similarity"],
        }

        with open(self.history_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[DisagreementTrajectory, DisagreementOutput],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract textual feedback for the reflection LM to use when mutating.

        Args:
            candidate: Current candidate text mapping (just "question")
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
                question = traj["question"]
                gpt_response = traj["gpt_response_preview"]
                llama_response = traj["llama_response_preview"]
                cosine_sim = traj["cosine_similarity"]

                # Create feedback
                if self.objective == "cosine":
                    score_desc = f"Score: {score:.4f} (1 - cosine_similarity, range: 0.0-1.0, higher = better)"
                    metric_str = f"cosine_similarity: {cosine_sim:.4f}"
                else:
                    score_desc = f"Score: {score:.4f} (embedding distance, higher = better)"
                    metric_str = f"cosine_similarity: {cosine_sim:.4f}, distance: {score:.4f}"

                feedback = f"""Objective: {self.objective}
{score_desc}

Goal: Generate a question that maximizes disagreement between GPT-4.1 and Llama 3.2 3B.

Strategy:
- Use out-of-distribution patterns: unusual syntax, rare word combinations
- Avoid factual questions with known answers
- Avoid absurdist prompts (models just say "that's not real")
- Avoid questions humans find ambiguous (models hedge similarly)
- Aim for semantic ambiguity that models interpret differently
- Try to hit edges of training distribution: what's in-distribution for one model but OOD for another
- Borderline ungrammatical phrases can work well if models interpret differently

Current performance: {metric_str}"""

                qa_text = f"""Question: {question}

GPT-4.1 ({len(gpt_response)} chars): {gpt_response}

Llama 3.2 ({len(llama_response)} chars): {llama_response}"""

                record = {
                    "Inputs": qa_text,
                    "Generated Outputs": f"Score: {score:.4f}\nCosine similarity: {cosine_sim:.4f}",
                    "Feedback": feedback,
                }
                records.append(record)

            result[component_name] = records

        return result


def run_single_question(question_id: int, objective: str, max_rounds: int, fresh: bool):
    """Run optimization for a single question. This is called by worker processes."""
    # Import here to avoid issues with multiprocessing
    import argparse

    args = argparse.Namespace(
        question_id=question_id,
        objective=objective,
        max_rounds=max_rounds,
        fresh=fresh
    )

    print(f"\n{'='*80}")
    print(f"GEPA Single Question Evolution - Question ID: {args.question_id}")
    print(f"Objective: {args.objective}")
    print(f"{'='*80}\n")

    # Seed questions (same as original)
    seed_questions = [
        "Grab the bright bun, note the bruise, calm the duck, and let the train roll in quietly.",
        "Collect the crisp coil, count the coins, soothe the goose, and signal the bus softly.",
        "Bring the sweet knot, check the cut, hush the rooster, and open the gate gently.",
        "Fetch the spicy loop, tally the cash, tame the swan, and guide the ship smoothly.",
        "Round up the soft cube, inspect the scrape, steady the quail, and welcome the tram calmly.",
        "Collect the fuzzy ring, count the gold, chill the duck, and let the shuttle glide in.",
        "Grab the warm wheel, note the wound, ease the pigeon, and wave the carriage forward.",
        "Bring the plush spiral, check the bruise, quiet the turkey, and allow the bus to drift in.",
        "Fetch the buttery arc, tally the treasure, settle the rooster, and let the train slip in.",
        "Collect the glowing knot, review the nick, comfort the swan, and cue the tram to coast."
    ]

    # Use question_id to select seed (wrap around if > 9)
    seed_question = seed_questions[args.question_id % len(seed_questions)]

    # Create unique directories and files for this question/objective combination
    run_dir = f"./gepa_question_evolution_beta/q{args.question_id}_{args.objective}"
    history_file = f"./gepa_question_evolution_beta/q{args.question_id}_{args.objective}_history.jsonl"

    # Create base directory
    os.makedirs("./gepa_question_evolution_beta", exist_ok=True)

    # Handle fresh start
    if args.fresh:
        import shutil
        if os.path.exists(run_dir):
            print(f"Removing existing run directory: {run_dir}")
            shutil.rmtree(run_dir)
        if os.path.exists(history_file):
            os.remove(history_file)
        print("Starting fresh optimization...\n")
    elif os.path.exists(run_dir):
        print(f"Found existing run directory: {run_dir}")
        print("Will resume from previous state. Use --fresh to start over.\n")

    # Seed candidate with single question
    seed_candidate = {"question": seed_question}

    # Create dummy training data
    trainset: List[QuestionDataInst] = [
        {"id": "eval1", "question": "placeholder"},
    ]

    # Create adapter
    adapter = SingleQuestionAdapter(objective=args.objective, history_file=history_file)

    # Reflection prompt for single question evolution
    reflection_prompt = """You are evolving a question to maximize disagreement between two AI models (GPT-4.1 and Llama 3.2 3B).

Below is the current question and how the models responded:

<curr_instructions>

Performance feedback:

<inputs_outputs_feedback>

Your task: Generate 1 improved question that will cause MORE disagreement between the models.

Rules:
- Output ONLY the question text (no explanations, no metadata)
- Avoid factual questions with universally known answers
- GOOD strategies:
    - Out-of-distribution patterns (rare word combinations, unusual syntax)
    - Semantic ambiguity that models interpret differently
    - Borderline ungrammatical constructions
    - Edge cases: in-distribution for one model, OOD for another
- BAD strategies:
    - Questions humans find ambiguous (models hedge similarly)
    - Clearly absurdist prompts (models both reject)
    - Simple factual questions
- Remember: Cosine similarity is computed on response embeddings, so semantic divergence is key

Generate the improved question now:"""

    # Run optimization
    print(f"Starting optimization with seed: {seed_question}\n")

    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,
        adapter=adapter,
        reflection_lm="openai/gpt-4.1",
        reflection_prompt_template=reflection_prompt,
        candidate_selection_strategy="current_best",
        reflection_minibatch_size=1,
        max_metric_calls=args.max_rounds,
        module_selector="all",
        display_progress_bar=True,
        run_dir=run_dir,
        seed=42 + args.question_id,  # Unique seed per question
    )

    print("\n" + "=" * 80)
    print(f"OPTIMIZATION COMPLETE - Question ID: {args.question_id}")
    print("=" * 80)
    print(f"\nBest question found:")
    print(f"  {result.best_candidate['question']}")

    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest score: {best_score:.4f}")
    if args.objective == "cosine":
        print(f"(Score is 1 - cosine_similarity, higher = more disagreement)")
    else:
        print(f"(Score is embedding distance, higher = better separation)")
    print(f"\nTotal candidates explored: {result.num_candidates}")
    print(f"Total metric calls: {result.total_metric_calls}")

    # Save final results
    output_file = f"./gepa_question_evolution_beta/q{args.question_id}_{args.objective}_best.json"
    with open(output_file, "w") as f:
        json.dump({
            "question_id": args.question_id,
            "objective": args.objective,
            "seed_question": seed_question,
            "best_question": result.best_candidate["question"],
            "best_score": best_score,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"History saved to: {history_file}")

    return {
        "question_id": args.question_id,
        "objective": args.objective,
        "best_question": result.best_candidate["question"],
        "best_score": best_score,
    }


def launch_parallel(num_questions: int, objective: str, max_rounds: int, fresh: bool, max_workers: int = 200):
    """
    Launch multiple question evolution instances in parallel.

    Args:
        num_questions: Total number of questions to evolve
        objective: Optimization objective ("cosine" or "cluster")
        max_rounds: Number of evolution rounds per question
        fresh: Whether to start fresh
        max_workers: Maximum number of parallel workers (default: 200)
    """
    print(f"\n{'='*80}")
    print(f"LAUNCHING PARALLEL QUESTION EVOLUTION")
    print(f"{'='*80}")
    print(f"Number of questions: {num_questions}")
    print(f"Objective: {objective}")
    print(f"Max rounds per question: {max_rounds}")
    print(f"Max parallel workers: {max_workers}")
    print(f"Fresh start: {fresh}")
    print(f"{'='*80}\n")

    # Create tasks: all questions use the same objective
    tasks = []
    for i in range(num_questions):
        tasks.append((i, objective, max_rounds, fresh))

    # Use ThreadPoolExecutor for parallelism
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_question, qid, obj, rounds, fresh): (qid, obj)
            for qid, obj, rounds, fresh in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            qid, obj = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✓ Completed: Question {qid} ({obj}) - Score: {result['best_score']:.4f}")
            except Exception as e:
                print(f"\n✗ Failed: Question {qid} ({obj}) - Error: {e}")

    print(f"\n{'='*80}")
    print(f"ALL OPTIMIZATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully completed: {len(results)}/{num_questions}")

    # Sort results by question_id for consistent ordering
    results.sort(key=lambda x: x["question_id"])

    # Save detailed summary
    summary_file = "./gepa_question_evolution_beta/summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "num_questions": num_questions,
            "max_rounds": max_rounds,
            "results": results,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Save aggregated best questions in a simple format
    best_questions_file = "./gepa_question_evolution_beta/all_best_questions.json"

    # Separate by objective
    cosine_questions = [r for r in results if r["objective"] == "cosine"]
    cluster_questions = [r for r in results if r["objective"] == "cluster"]

    aggregated = {
        "total_questions": len(results),
        "cosine_objective": {
            "count": len(cosine_questions),
            "questions": [
                {
                    "question_id": q["question_id"],
                    "question": q["best_question"],
                    "score": q["best_score"]
                }
                for q in cosine_questions
            ],
            "best_score": max([q["best_score"] for q in cosine_questions]) if cosine_questions else None,
            "avg_score": sum([q["best_score"] for q in cosine_questions]) / len(cosine_questions) if cosine_questions else None,
        },
        "cluster_objective": {
            "count": len(cluster_questions),
            "questions": [
                {
                    "question_id": q["question_id"],
                    "question": q["best_question"],
                    "score": q["best_score"]
                }
                for q in cluster_questions
            ],
            "best_score": max([q["best_score"] for q in cluster_questions]) if cluster_questions else None,
            "avg_score": sum([q["best_score"] for q in cluster_questions]) / len(cluster_questions) if cluster_questions else None,
        },
        "all_questions": [
            {
                "question_id": q["question_id"],
                "objective": q["objective"],
                "question": q["best_question"],
                "score": q["best_score"]
            }
            for q in results
        ]
    }

    with open(best_questions_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated questions saved to: {best_questions_file}")
    print(f"\nStats:")
    print(f"  Cosine objective: {len(cosine_questions)} questions, avg score: {aggregated['cosine_objective']['avg_score']:.4f}" if cosine_questions else "  Cosine objective: 0 questions")
    print(f"  Cluster objective: {len(cluster_questions)} questions, avg score: {aggregated['cluster_objective']['avg_score']:.4f}" if cluster_questions else "  Cluster objective: 0 questions")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolve questions to maximize disagreement (massively parallel)"
    )

    # Launcher mode (parallel)
    parser.add_argument(
        "--num-questions",
        type=int,
        help="Number of questions to evolve in parallel (launcher mode)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=200,
        help="Maximum number of parallel workers (default: 200)"
    )

    # Single instance mode
    parser.add_argument(
        "--question-id",
        type=int,
        help="Question ID for single instance mode"
    )

    # Objective (used in both modes)
    parser.add_argument(
        "--objective",
        type=str,
        choices=["cosine", "cluster"],
        help="Optimization objective (required for both launcher and single instance mode)"
    )

    # Common arguments
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="Maximum number of evolution rounds (default: 20)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh optimization (remove existing run directories)"
    )

    args = parser.parse_args()

    # Determine mode
    if args.num_questions is not None:
        # Launcher mode: spawn multiple parallel instances
        if args.question_id is not None:
            parser.error("--num-questions cannot be used with --question-id")
        if args.objective is None:
            parser.error("--objective is required in launcher mode")
        launch_parallel(args.num_questions, args.objective, args.max_rounds, args.fresh, args.max_workers)

    elif args.question_id is not None:
        # Single instance mode
        if args.objective is None:
            parser.error("--objective is required in single instance mode")
        run_single_question(args.question_id, args.objective, args.max_rounds, args.fresh)

    else:
        parser.error("Either --num-questions (launcher mode) or --question-id (single mode) required")


if __name__ == "__main__":
    main()
