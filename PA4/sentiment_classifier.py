import numpy as np
import pandas as pd
import gensim.downloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pretrained Google News word2vec (300-d). First use downloads ~1.6 GB via gensim.
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = 300


def document_vector(text):
    """
    Convert text into a single vector by averaging word embeddings.
    """
    # Tokenize the text based on whitespace
    tokens = text.split()
    
    # Ignore out-of-vocabulary (OOV) words
    vect = [model[x] for x in tokens if x in model]
    
    # If all words are OOV or text is empty, return a zero vector
    if not vect:
        return np.zeros(EMBEDDING_DIM)
        
    return np.mean(vect, axis=0)


def prepare_classification_data(file_path):
    """
    Load the IMDB dataset and convert it into feature and label arrays.
    """
    # Load CSV using pandas
    df = pd.read_csv(file_path)
    
    # Map labels to 1 / 0 (accepts both "pos"/"neg" and "positive"/"negative")
    label_map = {'pos': 1, 'positive': 1, 'neg': 0, 'negative': 0}
    df['sentiment'] = df['sentiment'].str.strip().str.lower().map(label_map)
    if df['sentiment'].isna().any():
        raise ValueError("Unknown sentiment labels in the dataset")
    
    # Apply document_vector to all reviews and stack them into an (N, 300) array
    X = np.stack(df['review'].apply(document_vector).values)
    y = df['sentiment'].values
    
    return X, y


def train_classifier(X_train, y_train,
                     hidden_dim=64,
                     learning_rate=0.001,
                     epochs=10,
                     batch_size=32):
    """
    Train a simple neural network classifier using document vectors.
    """
    # Convert NumPy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define a simple feed-forward neural network
    clf = nn.Sequential(
        nn.Linear(EMBEDDING_DIM, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid()
    )
    
    # Binary Cross Entropy Loss and Adam Optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)
    
    # Training Loop
    clf.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()          # Clear gradients
            outputs = clf(batch_X)         # Forward pass
            loss = criterion(outputs, batch_y) # Calculate loss
            loss.backward()                # Backward pass
            optimizer.step()               # Update weights
            
    return clf


def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluate the trained classifier.
    """
    # Set model to evaluation mode
    clf.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = clf(X_tensor).squeeze().numpy()
        
    # Convert probabilities to binary predictions (threshold 0.5)
    preds = (outputs >= 0.5).astype(int)
    
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    tn = np.sum((preds == 0) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    
    # Compute metrics (with safety checks for division by zero)
    acc = (tp + tn)
    
    accuracy = acc / len(y_test) if len(y_test) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * re) / (prec + re) if (prec + re) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(prec),
        "recall": float(re),
        "f1": float(f1)
    }