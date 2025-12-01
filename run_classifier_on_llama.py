"""Takes outputs from finetuned Llama models, embeds them with BERT and OpenAI,
and runs linear probe classifiers to distinguish between model outputs."""

import argparse
import pickle
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

from make_linear_probes import embed_with_bert, create_embeddings, classification_report

if __name__ == "__main__":
    # load answers from finetuned model
    with open("finetuned_model_outputs.csv", "r", encoding="utf-8") as f:
        df = pd.read_csv(f, names = ["question_id", "question", "model_name", "answer"])
    
    records = [
            QARecord(
                question_id=row["question_id"],
                question=row["question"],
                model_name=row["model_name"],
                answer=row["answer"],
            )
            for _, row in df.iterrows()
        ]
    
    # embed functions return (X, y) tuples — unpack both embeddings and labels
    X_bert, _ = embed_with_bert(records)
    X_openai, _ = create_embeddings(
        records,
        "text-embedding-3-large",
    )
    # Debugging info: shapes and label distribution
    print(f"BERT embeddings shape: {getattr(X_bert, 'shape', None)}")
    print(f"OpenAI embeddings shape: {getattr(X_openai, 'shape', None)}")

    with open("openai_linear_probe_model.pkl", "rb") as f:
        openai_linear_probe_model: LogisticRegression = pickle.load(f)

    with open("bert_linear_probe_model.pkl", "rb") as f:    
        bert_linear_probe_model: LogisticRegression = pickle.load(f)


    bert_test_pred = bert_linear_probe_model.predict(X_bert)
    openai_test_pred = openai_linear_probe_model.predict(X_openai)

    # Assume the ground truth is all from standard Llama. 
    y_test = np.array([1 for _ in range(8)])

    bert_test_acc = accuracy_score(y_test, bert_test_pred)
    
    openai_test_acc = accuracy_score(y_test, openai_test_pred)

    print(f"\nBERT accuracy: {bert_test_acc:.4f}")
    print(f"OPENAI accuracy: {openai_test_acc:.4f}")

    print("\nTest Set Classification Report:")
    print(
        classification_report(y_test, bert_test_pred, target_names=["Model 1", "Model 2"])
    )
    print(
        classification_report(y_test, openai_test_pred, target_names=["Model 1", "Model 2"])
    )