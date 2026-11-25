"""Part II demo: IMDB sentiment classification with averaged word2vec + a small MLP.

Adapted from the sample script provided with the assignment. Writes results_part2.txt.
"""
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from sentiment_classifier import prepare_classification_data, train_classifier, evaluate_classifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = prepare_classification_data("data/subset10000_IMDB_Dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
clf = train_classifier(X_train, y_train, hidden_dim=64, learning_rate=0.001, epochs=5, batch_size=32)
results = evaluate_classifier(clf, X_test, y_test)

with open("results_part2.txt", "w", encoding="utf-8") as f:
    print(f"X shape: {X.shape}  positives: {int((y == 1).sum())}  negatives: {int((y == 0).sum())}", file=f)
    print(f"Train / test: {len(y_train)} / {len(y_test)} (stratified, seed {SEED})", file=f)
    print("Test metrics (5 epochs, hidden_dim 64, lr 1e-3, batch 32):", file=f)
    for k, v in results.items():
        print(f"  {k:9s} {v:.4f}", file=f)
