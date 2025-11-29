"""
GEPA-based prompt optimizer for model fingerprinting.

Goal:
    Use GEPA to evolve a question-generator prompt that produces
    questions whose answers are maximally distinguishable between
    several base models (e.g., GPT-4.1, Llama-3.2-3B).

Main components:
    - BaseModel interface (wraps GPT / Llama calls)
    - embed_texts: embedding function using an embeddings model
    - cosine_similarity + separation metric µ
    - FingerprintAdapter: GEPAAdapter specialized for fingerprinting
    - run_gepa_optimization(): entry point to optimize the prompt
    - collect_dataset(): ask optimized questions to models and log outputs
    - train_classifier(): downstream classifier on outputs
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, Sequence, Tuple
from huggingface_hub import InferenceClient
import os
import csv 

import numpy as np

from openai import OpenAI
openai_client = OpenAI()  # pulls OPENAI_API_KEY from the environment



# Adjust these imports to match the actual GEPA library
try:
    import gepa
except ImportError:
    gepa = None  # type: ignore
    # You’ll need to install GEPA and fix these imports.


# ===========================
# 1. Model + Embedding Wrappers
# ===========================

class BaseModel:
    """
    Thin wrapper around a language model.

    Implement `generate` to call your actual model API
    (OpenAI GPT-4.1, Llama-3.2 3B via HF, etc.).
    """

    def __init__(self, name: str, temperature: float = 0.2, max_tokens: int = 512):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """
        Generate a single completion for the given prompt.

        TODO: replace with actual API calls.
        This placeholder just raises so you don't silently forget.
        """
        raise NotImplementedError(
            f"generate() not implemented for model {self.name}. "
            "Hook this up to OpenAI / HF / your serving stack."
        )


class OpenAIChatModel(BaseModel):
    def generate(self, prompt: str) -> str:
        response = openai_client.chat.completions.create(
            model=self.name,  # e.g. "gpt-4.1" or "gpt-4.1-mini"
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

class HFLlamaModel(BaseModel):
    """
    Wrapper around a remote Llama model on HuggingFace Inference API,
    using the conversational (chat) endpoint.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        hf_token: str | None = None,
    ):
        super().__init__(name=model_id, temperature=temperature, max_tokens=max_tokens)
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise RuntimeError("HF_TOKEN environment variable not set")
        # You can keep model= here or pass it in chat_completion; both work.
        self.client = InferenceClient(token=hf_token)

    def generate(self, prompt: str) -> str:
        # Use chat_completion for "conversational" / instruct models
        response = self.client.chat_completion(
            model=self.name,   # e.g. "meta-llama/Llama-3.2-3B-Instruct"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # HF returns an object with .choices[0].message.content (OpenAI-style)
        return response.choices[0].message.content.strip()



def demeta_system_prompt(raw_prompt: str, fixer_model: BaseModel) -> str:
    """
    Take a meta-style instruction like:
        'You are a question-generator configuration writer...'
    and rewrite it into a direct system prompt for the *question generator* itself.

    We deliberately strip out all mentions of:
      - writing instructions for another assistant
      - configuration writer / meta-process
      - backticks / 'your answer to me' etc.
    """
    fix_instructions = (
        "You will be given a long, meta-level instruction that explains how to "
        "write an instruction message for a *question generator*.\n\n"
        "Your job is to rewrite it into a single, direct SYSTEM PROMPT that will "
        "be shown **directly** to the question generator assistant.\n\n"
        "Requirements for your output:\n"
        "- Start with the sentence: 'You are a question generator for model fingerprinting.'\n"
        "- Speak directly to that assistant using 'You ...'.\n"
        "- Do NOT mention writing instructions for another assistant, being a "
        "configuration writer, or that this text will be used as an instruction.\n"
        "- Do NOT mention 'system prompts', 'APIs', 'downstream assistants', or "
        "any meta-process.\n"
        "- Remove any references to enclosing answers in backticks, 'your answer "
        "to me', or similar chatter about how to respond.\n"
        "- Keep and integrate all substantive content about:\n"
        "  * what kinds of questions to generate,\n"
        "  * which domains to use,\n"
        "  * formatting / structure requirements,\n"
        "  * safety constraints,\n"
        "  * how to maximize separation (low cosine similarity).\n"
        "- Return ONLY the cleaned system prompt text, with no surrounding "
        "explanations or backticks.\n"
    )

    prompt = (
        fix_instructions
        + "\n\n--- META INSTRUCTION TEXT START ---\n"
        + raw_prompt
        + "\n--- META INSTRUCTION TEXT END ---\n"
    )

    cleaned = fixer_model.generate(prompt).strip()
    return cleaned






# def embed_texts(
#     texts: Sequence[str],
#     embedding_model: str,
# ) -> np.ndarray:
#     """
#     Embed a list of texts into vectors using your embedding provider.

#     Returns: np.ndarray of shape (len(texts), dim)

#     TODO: replace body with actual embeddings API call.
#     """
#     # --- PLACEHOLDER IMPLEMENTATION ---
#     raise NotImplementedError(
#         f"embed_texts() must be implemented for embedding model {embedding_model}."
#     )


def embed_texts(texts: Sequence[str], embedding_model: str) -> np.ndarray:
    response = openai_client.embeddings.create(
        model=embedding_model,          # e.g. "text-embedding-3-large"
        input=list(texts),
    )
    vectors = [np.array(d.embedding, dtype=np.float32) for d in response.data]
    return np.stack(vectors, axis=0)




# ===========================
# 2. Metric: Separation via Cosine Similarity
# ===========================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


@dataclasses.dataclass
class QuestionScore:
    question: str
    score: float
    pairwise_sims: Dict[Tuple[str, str], float]  # (model_i, model_j) -> similarity


def score_question(
    question: str,
    base_models: Sequence[BaseModel],
    embedding_model: str,
    num_samples_per_model: int = 1,
) -> QuestionScore:
    """
    Given a candidate question, query all base models, embed their outputs,
    and compute a separation score based on pairwise cosine similarity.

    score = 1 - average_pairwise_cosine
    """

    # 1) Get responses from each model (optionally multiple samples)
    model_names: List[str] = []
    responses: List[str] = []

    for model in base_models:
        for _ in range(num_samples_per_model):
            text = model.generate(question)
            model_names.append(model.name)
            responses.append(text)

    # 2) Embed all responses
    embs = embed_texts(responses, embedding_model)  # shape (N, d)

    # 3) Compute pairwise cosine similarities
    n = len(responses)
    sims: List[float] = []
    pairwise_sims: Dict[Tuple[str, str], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_similarity(embs[i], embs[j])
            sims.append(s)
            key = (model_names[i], model_names[j])
            pairwise_sims[key] = s

    avg_sim = float(np.mean(sims)) if sims else 0.0
    separation_score = 1.0 - avg_sim  # higher = more separated

    return QuestionScore(
        question=question,
        score=separation_score,
        pairwise_sims=pairwise_sims,
    )

# ===========================
# 3. GEPA Adapter for Fingerprinting
# ===========================

class FingerprintAdapter(gepa.GEPAAdapter):  # type: ignore
    """
    GEPA adapter that:
      - Treats the candidate as a "question-generator prompt"
      - Uses `task_lm` with that system prompt to generate questions
      - Scores questions via model separation on embeddings
    """

    # Use GEPA's built-in reflection flow (we won't override propose_new_texts)
    propose_new_texts = None

    def __init__(
        self,
        base_models: Sequence[BaseModel],
        embedding_model: str,
        task_lm: BaseModel,
        reflection_lm: BaseModel,
        num_samples_per_model: int = 1,
    ):
        self.base_models = list(base_models)
        self.embedding_model = embedding_model
        self.task_lm = task_lm
        self.reflection_lm = reflection_lm
        self.num_samples_per_model = num_samples_per_model

        # --- logging fields ---
        self.val_eval_counter: int = 0
        self.val_scores_log: List[Tuple[int, float]] = []
        self.log_val_scores: bool = True

    # ---- Required GEPAAdapter methods ----

    def evaluate(
        self,
        batch: List[Any],
        candidate: Dict[str, Any],
        capture_traces: bool = False,
    ) -> "gepa.EvaluationBatch":
        """
        Evaluate a candidate on a minibatch.

        Args:
            batch: list of items from train/val set (we only care about len(batch))
            candidate: dict, e.g. {"system_prompt": "..."}
            capture_traces: whether to keep detailed trajectories (for reflection)

        Returns:
            gepa.EvaluationBatch with:
                - outputs: arbitrary objects per item (we store question + sims)
                - scores: floats (our separation scores)
                - trajectories: optional rich dicts used for reflection
        """
        system_prompt: str = candidate["system_prompt"]

        scores: List[float] = []
        outputs: List[Dict[str, Any]] = []
        trajectories: List[Dict[str, Any]] = []

        for _ in batch:
            # 1) Use task_lm + candidate prompt to generate a discriminative question
            question = self._generate_question(system_prompt=system_prompt)

            # 2) Score question via model separation
            q_score = score_question(
                question=question,
                base_models=self.base_models,
                embedding_model=self.embedding_model,
                num_samples_per_model=self.num_samples_per_model,
            )

            # 3) Build feedback (for reflection)
            fb = self._build_feedback(system_prompt, q_score)

            # Store a compact "output" object
            out_obj = {
                "question": q_score.question,
                "score": q_score.score,
                "pairwise_sims": q_score.pairwise_sims,
            }
            outputs.append(out_obj)
            scores.append(q_score.score)

            if capture_traces:
                trajectories.append(
                    {
                        "system_prompt": system_prompt,
                        "question": q_score.question,
                        "score": q_score.score,
                        "pairwise_sims": q_score.pairwise_sims,
                        "feedback": fb,
                    }
                )

        # When capture_traces is False, GEPA is fine with trajectories=None
        eval_batch = gepa.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

        # ---- NEW: log average separation for every batch ----
        if scores and self.log_val_scores:
            avg_score = float(np.mean(scores))
            iter_idx = self.val_eval_counter
            self.val_scores_log.append((iter_idx, avg_score))
            print(
                f"[GEPA] Eval batch {iter_idx}: avg separation score = {avg_score:.4f}"
            )
            self.val_eval_counter += 1

        return eval_batch




    def get_components_to_update(self, candidate: Dict[str, Any]) -> List[str]:
        """
        Tell GEPA which text fields it is allowed to mutate.
        We only have one: 'system_prompt'.
        """
        return ["system_prompt"]

    def make_reflective_dataset(
        self,
        candidate: Dict[str, Any],
        eval_batch: "gepa.EvaluationBatch",
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build the dataset that the reflection LLM will read.

        For each evaluated example, we package:
          - current system prompt
          - generated question
          - score
          - similarities
          - natural-language feedback
        """
        datasets: Dict[str, List[Dict[str, Any]]] = {}

        # We only have 'system_prompt' as a component
        if "system_prompt" in components_to_update:
            examples: List[Dict[str, Any]] = []

            # If trajectories exist, they contain feedback text
            trajectories = eval_batch.trajectories or []
            for i, out_obj in enumerate(eval_batch.outputs):
                # outputs are the dicts we put in evaluate()
                question = out_obj.get("question")
                score = eval_batch.scores[i]
                pairwise = out_obj.get("pairwise_sims", {})

                traj_fb = ""
                if i < len(trajectories):
                    traj_fb = trajectories[i].get("feedback", "")

                examples.append(
                    {
                        "component_name": "system_prompt",
                        "current_text": candidate.get("system_prompt", ""),
                        "generated_question": question,
                        "score": score,
                        "pairwise_similarities": pairwise,
                        "feedback": traj_fb,
                    }
                )

            datasets["system_prompt"] = examples

        return datasets

    # ---- Helper methods (unchanged) ----

    def _generate_question(self, system_prompt: str) -> str:
        """
        Ask `task_lm` (whose system behavior is defined by system_prompt)
        to generate one discriminative question.
        """
        prompt = (
            f"{system_prompt}\n\n"
            "Your task now: Output ONE single, concise question that is likely to "
            "elicit different answers from different large language models. "
            "Do not include explanations, just output the question text itself."
        )
        return self.task_lm.generate(prompt)

    def _build_feedback(self, system_prompt: str, q_score: QuestionScore) -> str:
        """
        Build a natural-language feedback string summarizing how well the question
        separated the models. This will be given to the reflection LLM.

        IMPORTANT: We explicitly tell the reflection LLM to output a DIRECT
        system prompt for the question generator (no meta-prompt, no backticks).
        """
        sims_desc = []
        for (m1, m2), s in q_score.pairwise_sims.items():
            sims_desc.append(f"{m1} vs {m2}: cosine similarity = {s:.3f}")
        sims_block = "\n".join(sims_desc) if sims_desc else "Not enough outputs."

        feedback = (
            "You are a prompt engineer. Your job is to REWRITE/continuous improve the following system "
            "prompt that will be used DIRECTLY as the system message for a "
            "question-generator model.\n\n"
            "The question-generator model will receive this system prompt and then "
            "generate questions for multiple language models. We want questions whose "
            "answers from different models have LOW cosine similarity (i.e., high "
            "separation score = 1 - avg cosine similarity).\n\n"
            "CRITICAL INSTRUCTIONS FOR YOUR OUTPUT:\n"
            "- Speak DIRECTLY to the question generator (e.g., 'You are a question "
            "generator for model fingerprinting...').\n"
            "- Do NOT describe yourself as a prompt engineer.\n"
            "- Do NOT talk about 'system prompts', 'downstream assistants', or "
            "any meta-process.\n"
            "- Do NOT wrap the output in backticks or any other formatting.\n"
            "- Output ONLY the improved system prompt text that will be shown to "
            "the question generator.\n\n"
            "Below is the CURRENT system prompt being used, followed by one example "
            "question it produced and how well that question separated model answers.\n\n"
            f"CURRENT SYSTEM PROMPT:\n{system_prompt}\n\n"
            f"EXAMPLE GENERATED QUESTION:\n{q_score.question}\n\n"
            f"Separation score (1 - avg cosine similarity): {q_score.score:.3f}\n"
            "Pairwise similarities between model answers:\n"
            f"{sims_block}\n\n"
            "Higher separation score is better. Questions where multiple models have "
            "very high cosine similarity (close to 1.0) are bad, because they give "
            "nearly identical answers.\n\n"
            "Rewrite the CURRENT SYSTEM PROMPT into a NEW, IMPROVED system prompt that:\n"
            "- Clearly states that the goal is to generate questions that MAXIMIZE "
            "separation between models' answers (low cosine similarity).\n"
            "- Encourages questions that force different reasoning styles, assumptions, "
            "and output formats.\n"
            "- Avoids generic 'AI ethics in medicine' / 'top N considerations' boilerplate.\n"
            "- Uses diverse domains, task types, and output formats.\n"
            "- Avoids yes/no and purely factual questions with a single obvious answer.\n"
            "- Stays safe (no harmful / illegal content), but is probing and nuanced.\n\n"
            "Again: reply ONLY with the new system prompt text itself, addressed directly "
            "to the question generator, with NO extra explanation and NO backticks."
        )
        return feedback




# ===========================
# 4. GEPA Optimization Entrypoint
# ===========================




def run_gepa_optimization(
    base_models: Sequence[BaseModel],
    task_lm: BaseModel,
    reflection_lm_model: str,   # <-- string, e.g. "openai/gpt-4.1-mini"
    embedding_model: str,
    max_metric_calls: int = 50,
) -> Dict[str, Any]:
    """
    Top-level function to run GEPA and return the best candidate.
    Also:
      - prints the average separation score of the best candidate on the full valset
      - saves eval-call index vs separation score to a CSV
    """

    if gepa is None:
        raise RuntimeError("GEPA library not imported. Install and import it first.")

    adapter = FingerprintAdapter(
        base_models=base_models,
        embedding_model=embedding_model,
        task_lm=task_lm,
        reflection_lm=task_lm,      # we don't actually use this inside adapter now
        num_samples_per_model=1,
    )

    # Train/val sets: GEPA is happy with plain lists; only sizes matter.
    trainset = list(range(10))   # 10 "items" for training evals
    valset   = list(range(5))    # 5 "items" for validation evals

    seed_candidate = {
        "system_prompt": (
            "You are a question generator for model fingerprinting.\n"
            "Your job is to generate questions that make different language models "
            "respond in noticeably different ways, so that we can identify which "
            "model produced a given answer.\n\n"
            "Guidelines:\n"
            "- Ask about nuanced preferences, tradeoffs, ethical dilemmas, or "
            "subjective judgments, not simple factual trivia.\n"
            "- Encourage models to explain their reasoning in depth.\n"
            "- Avoid yes/no questions and questions with a single obvious answer.\n"
            "- Encourage specific formats (e.g., numbered lists, step-by-step "
            "reasoning) that different models might execute differently.\n"
            "- Avoid directly referencing any model names or architectures.\n"
        )
    }

    # GEPA's own optimization loop; the GEPA progress bar is controlled by
    # display_progress_bar=True here.
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm_model,  # e.g. "openai/gpt-4.1-mini"
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
    )

    best_candidate = result.best_candidate

    # The raw best candidate may be a *meta* instruction (e.g. 'You are a
    # question-generator configuration writer...'). We project it back into
    # a direct system prompt for the question generator.
    raw_system_prompt: str = best_candidate["system_prompt"]
    print("\n[GEPA] Raw best system prompt (may be meta-level):\n")
    print(raw_system_prompt)

    cleaned_system_prompt = demeta_system_prompt(
        raw_system_prompt,
        # we can reuse task_lm as the fixer model to keep costs manageable
        task_lm,
    )

    print("\n[GEPA] Cleaned (direct) system prompt for question generator:\n")
    print(cleaned_system_prompt)

    # Overwrite the candidate's system_prompt with the cleaned version,
    # and also return that to the caller.
    best_candidate["system_prompt"] = cleaned_system_prompt


    # ---- Evaluate best candidate on full valset (without logging as a new iteration) ----
    adapter.log_val_scores = False  # don't treat this as another logged eval
    full_val_batch = valset  # just a list of 0..4
    eval_batch = adapter.evaluate(full_val_batch, best_candidate, capture_traces=False)
    best_avg_sep = float(np.mean(eval_batch.scores)) if eval_batch.scores else 0.0
    print(
        f"\n[GEPA] Average separation score on full valset for best/final "
        f"candidate system prompt: {best_avg_sep:.4f}"
    )

    # ---- Save per-eval-call scores to CSV ----
    csv_path = "gepa_separation_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_call_index", "avg_separation_score"])
        for eval_idx, avg_score in adapter.val_scores_log:
            writer.writerow([eval_idx, avg_score])
    print(f"[GEPA] Saved separation scores for all evaluation calls to {csv_path}\n")

    return best_candidate





# ===========================
# 5. Dataset Collection for Classifier
# ===========================

@dataclasses.dataclass
class QARecord:
    question_id: int
    question: str
    model_name: str
    answer: str


def collect_dataset(
    optimized_system_prompt: str,
    base_models: Sequence[BaseModel],
    question_generator: BaseModel,
    num_questions: int,
) -> List[QARecord]:
    """
    Use the optimized system prompt to generate questions, then ask all models
    those questions and log the answers.
    """

    records: List[QARecord] = []

    for qid in range(num_questions):
        # Generate a new question
        q_prompt = (
            f"{optimized_system_prompt}\n\n"
            "Generate ONE single question according to the instructions above.\n"
            "Output only the question text."
        )
        question = question_generator.generate(q_prompt)

        # Ask each model
        for model in base_models:
            answer_prompt = (
                "You will be given a question. Answer it as you normally would.\n\n"
                f"Question: {question}\n"
            )
            answer = model.generate(answer_prompt)

            records.append(
                QARecord(
                    question_id=qid,
                    question=question,
                    model_name=model.name,
                    answer=answer,
                )
            )

    return records


# ===========================
# 6. Classifier Training Skeleton
# ===========================

def prepare_features_and_labels(
    records: Sequence[QARecord],
    embedding_model: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Turn QA records into (X, y) for classifier training using embeddings.

    X: embeddings of answers (and optionally include question text)
    y: integer labels (0..num_models-1)
    label_map: model_name -> int
    """
    texts = [r.answer for r in records]  # could also concat question + answer
    X = embed_texts(texts, embedding_model)

    # Map model names to integers
    label_map: Dict[str, int] = {}
    labels: List[int] = []
    next_label = 0

    for r in records:
        if r.model_name not in label_map:
            label_map[r.model_name] = next_label
            next_label += 1
        labels.append(label_map[r.model_name])

    y = np.array(labels, dtype=np.int64)
    return X, y, label_map


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Train a simple classifier on embeddings.

    Example uses scikit-learn logistic regression; swap for anything you like.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return clf


def generate_fingerprint_questions(
    optimized_system_prompt: str,
    question_generator: BaseModel,
    num_questions: int = 10,
) -> List[str]:
    """
    Use the optimized system prompt to generate a *diverse* set of fingerprinting questions.
    We ask the LM to produce all questions in one shot and enforce diversity via instructions.
    """
    prompt = (
        f"{optimized_system_prompt}\n\n"
        f"Now generate {num_questions} DISTINCT questions for model fingerprinting.\n"
        "Requirements:\n"
        "- Each question must be about a DIFFERENT domain or scenario "
        "(e.g., healthcare, education, economics, law, ethics, creative writing, personal decision-making, etc.).\n"
        "- Do NOT reuse the same narrative setup or core scenario.\n"
        "- Avoid repeating the 'city council / public transportation' scenario.\n"
        "- All questions should follow the guidelines above about nuanced tradeoffs, reasoning, and structure.\n"
        "- Label them as Q1:, Q2:, ..., Q{num_questions}: and output only the questions.\n"
    )

    raw = question_generator.generate(prompt)

    questions: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept formats like "Q1: ..." or "Q1 - ..." or "1." etc.
        if line.lower().startswith("q"):
            # Try to split at first ':' or '-'
            if ":" in line:
                _, qtext = line.split(":", 1)
                questions.append(qtext.strip())
            elif "-" in line:
                _, qtext = line.split("-", 1)
                questions.append(qtext.strip())
            else:
                # fallback: take after 'Qn'
                questions.append(line)
    # Fallback: if parsing failed, just return nonempty lines
    if not questions:
        questions = [ln for ln in raw.splitlines() if ln.strip()]
    return questions





# ===========================
# 7. Example main() wiring
# ===========================
def main():
    # For now: just differentiate two OpenAI models
    gpt41 = OpenAIChatModel(name="gpt-4.1")
    llama32 = HFLlamaModel(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.2,
        max_tokens=512,
    )

    base_models = [gpt41, llama32]

    # Use a strong but cheaper model that is NOT one of the base models as task LM
    task_lm = OpenAIChatModel(name="gpt-4.1-mini")

    embedding_model = "text-embedding-3-large"

    # 1) Run GEPA to optimize question-generator system prompt
    best_candidate = run_gepa_optimization(
        base_models=base_models,
        task_lm=task_lm,
        reflection_lm_model="openai/gpt-5.1",
        embedding_model=embedding_model,
        max_metric_calls=18,
    )

    optimized_system_prompt: str = best_candidate["system_prompt"]
    print("\n=== Optimized system prompt ===\n")
    print(optimized_system_prompt)

    # Save the optimized system prompt to a text file for inspection / reproducibility
    prompt_path = "optimized_system_prompt.txt"
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(optimized_system_prompt)
        print(f"\nSaved optimized system prompt to {prompt_path}")
    except OSError as e:
        print(f"Warning: failed to write {prompt_path}: {e}")


    # 2) Generate 10 diverse fingerprinting questions
    questions = generate_fingerprint_questions(
        optimized_system_prompt=optimized_system_prompt,
        question_generator=task_lm,
        num_questions=10,
    )

    print("\n=== Fingerprinting questions ===\n")
    for i, q in enumerate(questions, start=1):
        print(f"Q{i}: {q}\n")

    # 3) Save to file
    out_path = "fingerprint_questions.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            f.write(f"Q{i}: {q}\n")
    print(f"\nSaved {len(questions)} questions to {out_path}")




def test_api():
    model = OpenAIChatModel(name="gpt-4.1")
    print(model.generate("Explain in one sentence what quantum tunneling is."))

    X = embed_texts(
        ["hello world", "the quick brown fox"],
        embedding_model="text-embedding-3-large"
    )
    print(X.shape)

def test_llama_only():
    llama32 = HFLlamaModel(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.2,
        max_tokens=128,
    )
    print("Llama says:\n", llama32.generate("In one sentence, explain why the sky appears blue."))



if __name__ == "__main__":
    #test_llama_only()
    main()
