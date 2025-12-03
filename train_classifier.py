"""
Train a linear probe classifier to distinguish GPT-4.1 vs Llama 3.2 3B responses
based on response embeddings from evolved questions.

Usage:
    python train_classifier.py --questions best_questions_batch.json --save classifier_debate.pth --metrics classifier_metrics_debate.json
    python train_classifier.py --questions best_questions_batch.json --test "Your test question here"
    python train_classifier.py --questions train_questions.json --test-file test_questions.json
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import openai


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


def collect_training_data(questions: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Collect training data by querying both models and getting embeddings in parallel.

    Returns:
        X: Feature matrix of shape (2*n_questions, embedding_dim)
        y: Labels array of shape (2*n_questions,) where 0=GPT-4.1, 1=Llama
        responses: List of dicts containing questions and responses from both models
    """
    print(f"\nCollecting training data from {len(questions)} questions...")
    print("=" * 80)

    def process_question(i_question_tuple):
        """Process a single question: get responses and embeddings from both models."""
        i, question = i_question_tuple
        print(f"\nQuestion {i+1}/{len(questions)}: {question[:100]}...")

        # Get GPT-4.1 response and embedding
        print("  Querying GPT-4.1...")
        gpt_response = get_model_response(question, "gpt-4.1")
        print(f"  GPT-4.1 response: {gpt_response[:150]}...")
        gpt_embedding = get_embedding(gpt_response)

        # Get Llama response and embedding
        print("  Querying Llama 3.2 3B...")
        llama_response = get_model_response(question, "llama-3.2-3b")
        print(f"  Llama response: {llama_response[:150]}...")
        llama_embedding = get_embedding(llama_response)

        return gpt_embedding, llama_embedding, gpt_response, llama_response

    # Process all questions in parallel
    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(process_question, enumerate(questions)))

    # Unpack results
    embeddings = []
    labels = []
    responses = []
    for i, (gpt_embedding, llama_embedding, gpt_response, llama_response) in enumerate(results):
        embeddings.append(gpt_embedding)
        labels.append(0)  # GPT-4.1 = 0
        embeddings.append(llama_embedding)
        labels.append(1)  # Llama = 1

        # Store responses with their questions
        responses.append({
            "question": questions[i],
            "gpt_response": gpt_response,
            "llama_response": llama_response
        })

    X = np.array(embeddings)
    y = np.array(labels)

    print(f"\n{'=' * 80}")
    print(f"Collected {len(X)} samples ({len(questions)} from each model)")
    print(f"Embedding dimension: {X.shape[1]}")
    print(f"Label distribution: GPT-4.1={np.sum(y == 0)}, Llama={np.sum(y == 1)}")

    return X, y, responses


class LinearProbe(nn.Module):
    """Simple one-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.output(x)


def train_linear_probe(X: np.ndarray, y: np.ndarray, hidden_dim: int = 128,
                       test_size: float = 0.5, epochs: int = 50, 
                       X_test_custom: np.ndarray = None, y_test_custom: np.ndarray = None) -> Tuple[LinearProbe, Dict]:
    """
    Train a linear probe (one-layer NN) on the embeddings.

    Args:
        X: Feature matrix
        y: Labels
        hidden_dim: Hidden layer dimension
        test_size: Fraction of data for testing (ignored if X_test_custom is provided)
        epochs: Number of training epochs
        X_test_custom: Optional custom test set features
        y_test_custom: Optional custom test set labels

    Returns:
        classifier: Trained LinearProbe model
        metrics: Dictionary with evaluation metrics
    """
    print(f"\n{'=' * 80}")
    print("Training linear probe classifier...")
    print(f"Hidden dimension: {hidden_dim}")
    print("=" * 80)

    # Split data
    if X_test_custom is not None and y_test_custom is not None:
        # Use custom test set, train on all provided data
        X_train, y_train = X, y
        X_test, y_test = X_test_custom, y_test_custom
        print("\nUsing custom test set")
    else:
        # Use standard train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print("\nUsing random train-test split")

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")


    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # Initialize model
    input_dim = X.shape[1]
    classifier = LinearProbe(input_dim, hidden_dim)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Training loop
    print("\nTraining...")
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()

        outputs = classifier(X_train_t)
        loss = criterion(outputs, y_train_t)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            classifier.eval()
            with torch.no_grad():
                train_outputs = classifier(X_train_t)
                train_pred = torch.argmax(train_outputs, dim=1)
                train_acc = (train_pred == y_train_t).float().mean().item()

                test_outputs = classifier(X_test_t)
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = (test_pred == y_test_t).float().mean().item()

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Final evaluation
    classifier.eval()
    with torch.no_grad():
        train_outputs = classifier(X_train_t)
        train_pred = torch.argmax(train_outputs, dim=1).numpy()

        test_outputs = classifier(X_test_t)
        test_pred = torch.argmax(test_outputs, dim=1).numpy()

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n{'=' * 80}")
    print(f"Final Training accuracy: {train_acc:.4f}")
    print(f"Final Test accuracy: {test_acc:.4f}")

    print("\nTest set classification report:")
    print(classification_report(y_test, test_pred, target_names=["GPT-4.1", "Llama 3.2 3B"]))

    print("\nTest set confusion matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"                 Predicted GPT  Predicted Llama")
    print(f"Actual GPT       {cm[0][0]:14d}  {cm[0][1]:15d}")
    print(f"Actual Llama     {cm[1][0]:14d}  {cm[1][1]:15d}")

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
        "hidden_dim": hidden_dim,
        "epochs": epochs
    }

    return classifier, metrics


def predict_on_new_question(classifier: LinearProbe, question: str) -> Dict:
    """
    Run inference on a new question by getting responses from both models
    and predicting which model generated each response.

    Returns:
        Dictionary with predictions and probabilities
    """
    print(f"\n{'=' * 80}")
    print(f"Running inference on: {question}")
    print("=" * 80)

    results = {}

    # Get GPT-4.1 response
    print("\nQuerying GPT-4.1...")
    gpt_response = get_model_response(question, "gpt-4.1")
    print(f"Response: {gpt_response[:300]}...")
    gpt_embedding = get_embedding(gpt_response)

    # Predict with neural network
    classifier.eval()
    with torch.no_grad():
        gpt_embedding_t = torch.FloatTensor(gpt_embedding).unsqueeze(0)
        gpt_output = classifier(gpt_embedding_t)
        gpt_prob = torch.softmax(gpt_output, dim=1)[0]
        gpt_pred = torch.argmax(gpt_output, dim=1)[0]

    results["gpt"] = {
        "response": gpt_response,
        "predicted_label": int(gpt_pred.item()),
        "predicted_model": "GPT-4.1" if gpt_pred.item() == 0 else "Llama 3.2 3B",
        "correct": bool(gpt_pred.item() == 0),
        "confidence": {
            "gpt-4.1": float(gpt_prob[0].item()),
            "llama-3.2-3b": float(gpt_prob[1].item())
        }
    }

    # Get Llama response
    print("\nQuerying Llama 3.2 3B...")
    llama_response = get_model_response(question, "llama-3.2-3b")
    print(f"Response: {llama_response[:300]}...")
    llama_embedding = get_embedding(llama_response)

    # Predict with neural network
    with torch.no_grad():
        llama_embedding_t = torch.FloatTensor(llama_embedding).unsqueeze(0)
        llama_output = classifier(llama_embedding_t)
        llama_prob = torch.softmax(llama_output, dim=1)[0]
        llama_pred = torch.argmax(llama_output, dim=1)[0]

    results["llama"] = {
        "response": llama_response,
        "predicted_label": int(llama_pred.item()),
        "predicted_model": "GPT-4.1" if llama_pred.item() == 0 else "Llama 3.2 3B",
        "correct": bool(llama_pred.item() == 1),
        "confidence": {
            "gpt-4.1": float(llama_prob[0].item()),
            "llama-3.2-3b": float(llama_prob[1].item())
        }
    }

    print("\n" + "=" * 80)
    print("PREDICTIONS:")
    print("=" * 80)
    print(f"\nGPT-4.1 response predicted as: {results['gpt']['predicted_model']}")
    print(f"  Confidence: GPT-4.1={results['gpt']['confidence']['gpt-4.1']:.3f}, Llama={results['gpt']['confidence']['llama-3.2-3b']:.3f}")
    print(f"  Correct: {results['gpt']['correct']}")

    print(f"\nLlama response predicted as: {results['llama']['predicted_model']}")
    print(f"  Confidence: GPT-4.1={results['llama']['confidence']['gpt-4.1']:.3f}, Llama={results['llama']['confidence']['llama-3.2-3b']:.3f}")
    print(f"  Correct: {results['llama']['correct']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train classifier to distinguish GPT-4.1 vs Llama 3.2 3B")
    parser.add_argument("--questions", type=str, required=True, help="Path to JSON file with evolved questions")
    parser.add_argument("--test", type=str, help="Test question for inference")
    parser.add_argument("--test-file", type=str, help="Path to JSON file with test questions (for deterministic test set)")
    parser.add_argument("--save", type=str, default="classifier.pth", help="Path to save trained classifier")
    parser.add_argument("--load", type=str, help="Path to load pre-trained classifier (skips training)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension for linear probe")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--train-responses", type=str, default="train_responses.json", help="Path to save training responses")
    parser.add_argument("--metrics", type=str, default="classifier_metrics.json", help="Path to save classifier metrics")

    args = parser.parse_args()

    # Load questions
    print(f"Loading questions from {args.questions}...")
    with open(args.questions, "r") as f:
        data = json.load(f)

    # Extract questions using "question" key
    if isinstance(data, list):
        questions = [item["question"] for item in data if "question" in item]
    else:
        raise ValueError(f"Questions file must contain a list of questions with 'question' keys")

    print(f"Loaded {len(questions)} questions")

    # Load test questions if provided
    X_test_custom = None
    y_test_custom = None
    test_responses = None
    if args.test_file:
        print(f"\nLoading test questions from {args.test_file}...")
        with open(args.test_file, "r") as f:
            test_data = json.load(f)

        # Extract test questions
        if isinstance(test_data, list):
            test_questions = test_data
        else:
            raise ValueError(f"Test file must contain a list of questions")

        print(f"Loaded {len(test_questions)} test questions")
        print("Collecting test data...")
        X_test_custom, y_test_custom, test_responses = collect_training_data(test_questions)

    # Train or load classifier
    if args.load:
        print(f"\nLoading pre-trained classifier from {args.load}...")
        # Need to know input dimension to initialize model
        # Load a sample embedding to get dimension
        sample_response = get_model_response(questions[0], "gpt-4.1")
        sample_embedding = get_embedding(sample_response)
        input_dim = sample_embedding.shape[0]

        classifier = LinearProbe(input_dim, hidden_dim=args.hidden_dim)
        classifier.load_state_dict(torch.load(args.load))
        classifier.eval()
        print("Classifier loaded!")
    else:
        # Collect training data
        X, y, train_responses = collect_training_data(questions)

        # Save training responses
        train_responses_file = args.train_responses
        with open(train_responses_file, "w") as f:
            json.dump(train_responses, f, indent=2)
        print(f"\nTraining responses saved to {train_responses_file}")

        # Save test responses if they exist
        if test_responses is not None:
            test_responses_file = "test_responses.json"
            with open(test_responses_file, "w") as f:
                json.dump(test_responses, f, indent=2)
            print(f"Test responses saved to {test_responses_file}")

        # Train classifier with custom test set if provided
        classifier, metrics = train_linear_probe(
            X, y,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            X_test_custom=X_test_custom,
            y_test_custom=y_test_custom
        )

        # Save classifier
        print(f"\nSaving classifier to {args.save}...")
        torch.save(classifier.state_dict(), args.save)
        print("Classifier saved!")

        # Save metrics
        metrics_file = args.metrics
        metrics_to_save = {
            "train_accuracy": metrics["train_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
            "hidden_dim": metrics["hidden_dim"],
            "epochs": metrics["epochs"]
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Metrics saved to {metrics_file}")

    # Run inference if test question provided
    if args.test:
        results = predict_on_new_question(classifier, args.test)

        # Save inference results
        inference_file = "inference_results.json"
        with open(inference_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nInference results saved to {inference_file}")
    elif args.load and not args.test_file:
        # Interactive inference mode (only if loading a model and not batch testing)
        print("\n" + "=" * 80)
        print("INTERACTIVE INFERENCE MODE")
        print("=" * 80)
        print("\nYou can now test the classifier on custom questions.")
        print("Type 'quit' to exit.\n")

        while True:
            question = input("Enter a test question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break

            if question:
                results = predict_on_new_question(classifier, question)
                print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()