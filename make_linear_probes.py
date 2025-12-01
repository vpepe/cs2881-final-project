"""
Linear probe experiment for model fingerprinting.

This script:
1. Loads fingerprinting questions from fingerprint_questions.txt
2. Generates outputs from gpt-4.1 and llama-3.2-3B-Instruct
3. Embeds the outputs using OpenAI embeddings or BERT embeddings
4. Trains a linear probe on the embedding differences
"""

import argparse
from typing import List, Sequence, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch


from fingerprint_gepa import (
    OpenAIChatModel,
    HFLlamaModel,
    embed_texts,
    QARecord,
)


def load_questions(filepath: str) -> List[str]:
    """Load questions from the fingerprint questions file."""
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse format "Q1: question text"
            if ":" in line:
                _, question = line.split(":", 1)
                questions.append(question.strip())
            else:
                questions.append(line)
    return questions


def generate_model_outputs(
    questions: List[str],
    model1: OpenAIChatModel,
    model2: HFLlamaModel,
    system_prompt: str
) -> List[QARecord]:
    """
    Generate outputs from both models for all questions.

    Returns a list of QARecord objects containing question-answer pairs.
    Save model outputs for later use.
    """
    records = []

    for qid, question in enumerate(questions):
        print(f"Processing question {qid + 1}/{len(questions)}: {question[:60]}...")

        # Generate answer from model 1 (GPT-4.1)
        try:
            answer1 = model1.generate(question, system_prompt)
            records.append(
                QARecord(
                    question_id=qid,
                    question=question,
                    model_name=model1.name,
                    answer=answer1,
                )
            )
            print(f"  ✓ {model1.name} answered")
        except Exception as e:
            print(f"  ✗ {model1.name} failed: {e}")

        # Generate answer from model 2 (Llama)
        try:
            answer2 = model2.generate(question, system_prompt)
            records.append(
                QARecord(
                    question_id=qid,
                    question=question,
                    model_name=model2.name,
                    answer=answer2,
                )
            )
            print(f"  ✓ {model2.name} answered")
        except Exception as e:
            print(f"  ✗ {model2.name} failed: {e}")

    # Save model outputs into a CSV for later use:
    with open("model_outputs.csv", "w", encoding="utf-8") as f:
        pd.DataFrame(
            {
                "question_id": [r.question_id for r in records],
                "question": [r.question for r in records],
                "model_name": [r.model_name for r in records],
                "answer": [r.answer for r in records],
            }
        ).to_csv(f, index=False, header=False)

    return records

def embed_with_bert(records: Sequence[str]) -> np.ndarray:
    # embed texts using BERT embeddings:
    model_1_name = records[0].model_name
    texts = [record.answer for record in records]
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    Y = [1 if record.model_name == model_1_name else 0 for record in records]

    return embeddings, Y
     

def create_embeddings(records: List[QARecord], embedding_model: str) -> np.ndarray:
    """Create embeddings for a list of texts using the specified embedding model."""

    model_1_name = records[0].model_name
    print([record.model_name for record in records])

    for record in records:
        print(
            f"Embedding response from {record.model_name} for question {record.question_id}"
        )
    texts = [record.answer for record in records]

    embeddings = embed_texts(texts, embedding_model=embedding_model)

    Y = [1 if record.model_name == model_1_name else 0 for record in records]
    print(Y)

    return np.array(embeddings), np.array(Y)


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a linear probe (logistic regression) on the embeddings.

    Returns:
        Trained classifier
    """
    print("\n" + "=" * 60)
    print("Training Linear Probe")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluate
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print("\nTest Set Classification Report:")
    print(
        classification_report(y_test, y_test_pred, target_names=["Model 1", "Model 2"])
    )

    # Show feature importance (coefficients)
    coef_norm = np.linalg.norm(clf.coef_)
    print(f"\nCoefficient norm: {coef_norm:.4f}")

    return clf


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Linear probe experiment for model fingerprinting"
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["bert", "openai"],
        default="bert",
        help="Type of embeddings to use: 'bert' or 'openai' (default: bert)",
    )
    parser.add_argument(
        "--generate-new",
        action="store_true",
        help="Generate new model outputs instead of reading from CSV",
    )
    parser.add_argument(
        "--inference_mode",
        action = "store_true",
        help = "Run in inference mode with pre-trained linear probe",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Linear Probe Experiment for Model Fingerprinting")
    print("=" * 60)
    print(f"Embedding type: {args.embedding_type}")
    print(f"Generate new data: {args.generate_new}")
    print("=" * 60)

    # Configuration
    questions_file = "fingerprint_questions.txt"
    openai_embedding_model = "text-embedding-3-large"

    # Load or generate data
    if args.generate_new:
        # Load questions
        print(f"\nLoading questions from {questions_file}...")
        questions = load_questions(questions_file)
        print(f"Loaded {len(questions)} questions")

        # load system prompt
        with open("optimized_system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        # Initialize models
        print("\nInitializing models...")
        gpt41 = OpenAIChatModel(name="gpt-4.1", temperature=0.7, max_tokens=512)
        llama32 = HFLlamaModel(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            temperature=0.7,
            max_tokens=512,
        )
        print(f"  - {gpt41.name}")
        print(f"  - {llama32.name}")

        # Generate outputs
        print("\n" + "=" * 60)
        print("Generating Model Outputs")
        print("=" * 60)
        records = generate_model_outputs(questions, gpt41, llama32, system_prompt)
        print(f"\nGenerated {len(records)} total responses")
    else:
        # Load from CSV
        print("\nLoading data from model_outputs.csv...")
        with open("model_outputs.csv", "r", encoding="utf-8") as f:
            df = pd.read_csv(
                f, header=None, names=["question_id", "question", "model_name", "answer"]
            )
        records = [
            QARecord(
                question_id=row["question_id"],
                question=row["question"],
                model_name=row["model_name"],
                answer=row["answer"],
            )
            for _, row in df.iterrows()
        ]
        print(f"Loaded {len(records)} records from model_outputs.csv")

    # Create embeddings
    print("\n" + "=" * 60)
    print(f"Creating {args.embedding_type.upper()} Embeddings for Linear Probe")
    print("=" * 60)

    if args.embedding_type == "bert":
        X, y = embed_with_bert(records)
    else:  # openai
        X, y = create_embeddings(records, openai_embedding_model)

    # Train linear probe
    clf = train_linear_probe(X, y)

    # Save the model
    output_filename = f"{args.embedding_type}_linear_probe_model.pkl"
    with open(output_filename, "wb") as f:
        import pickle
        pickle.dump(clf, f)
        print(f"\nSaved linear probe model to {output_filename}")

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)

    return clf, X, y, records



if __name__ == "__main__":
    clf, X, y, records = main()
