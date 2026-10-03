"""
Neural Network Sentiment Classifier

Name: Cayden Goddard
Date: 10/3/2026

Architecture:
Embedding -> Mean Pooling -> Linear -> ReLU -> Linear -> Output (logits)
"""

import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Sample dataset
# -----------------------------
data = [
    ("I love this movie", 1),
    ("This is amazing", 1),
    ("I hate this", 0),
    ("This is terrible", 0),
    ("I really enjoyed this", 1),
    ("This was awful", 0),
    # Expanded examples (the assignment allows this). None of the test
    # sentences below appear here, but their words do.
    ("I like this movie", 1),
    ("This movie is good", 1),
    ("What a great movie", 1),
    ("I really love this film", 1),
    ("This was wonderful", 1),
    ("I enjoyed this a lot", 1),
    ("That movie was bad", 0),
    ("This movie is not good", 0),
    ("I do not like this movie", 0),
    ("I really hated this film", 0),
    ("This was boring", 0),
    ("What a terrible movie", 0),
]

# 1 = positive
# 0 = negative


# -----------------------------
# Text preprocessing
# -----------------------------
def tokenize(text):
    """Convert text to lowercase and split into tokens."""
    return text.lower().split()


def build_vocabulary(dataset):
    """
    Build a vocabulary dictionary from the dataset.

    Reserve:
    0 = <PAD>
    1 = <UNK>
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for text, _ in dataset:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def text_to_indices(text, vocab):
    """Convert text into a list of token indices. Unknown words use <UNK>."""
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]


def pad_sequence(indices, max_length, pad_value=0):
    """Pad or truncate a sequence to max_length."""
    if len(indices) < max_length:
        return indices + [pad_value] * (max_length - len(indices))
    return indices[:max_length]


# -----------------------------
# Prepare dataset
# -----------------------------
def prepare_data(dataset, vocab, max_length):
    """Convert dataset into tensors for model training."""
    X = []
    y = []

    for text, label in dataset:
        indices = text_to_indices(text, vocab)
        padded = pad_sequence(indices, max_length, vocab["<PAD>"])

        X.append(padded)
        y.append(label)

    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor


# -----------------------------
# Model definition
# -----------------------------
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        # Embedding matrix E: [vocab_size x embedding_dim]
        # padding_idx=0 keeps the <PAD> vector at zero and stops it from learning
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Hidden layer: [embedding_dim -> hidden_dim]
        self.hidden = nn.Linear(embedding_dim, hidden_dim)

        # Non-linearity
        self.relu = nn.ReLU()

        # Output layer: [hidden_dim -> output_dim]
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """x shape: [batch_size, sequence_length]"""
        # [batch, seq_len] -> [batch, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # Mean pooling that ignores <PAD> positions -> [batch, embedding_dim]
        mask = (x != 0).unsqueeze(-1).float()  # [batch, seq_len, 1]
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # [batch, embedding_dim] -> [batch, hidden_dim]
        hidden = self.hidden(pooled)

        activated = self.relu(hidden)

        # [batch, hidden_dim] -> [batch, output_dim] (raw scores / logits)
        logits = self.output(activated)

        return logits


# -----------------------------
# Prediction helper
# -----------------------------
def predict_sentiment(text, model, vocab, max_length):
    """Predict sentiment for a single sentence."""
    model.eval()

    indices = text_to_indices(text, vocab)
    padded = pad_sequence(indices, max_length, vocab["<PAD>"])

    # Add a batch dimension: [1, max_length]
    input_tensor = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)

        # Softmax turns logits into probabilities; take the single row
        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_class = torch.argmax(probabilities).item()

        confidence = probabilities[predicted_class].item()

    label_name = "positive" if predicted_class == 1 else "negative"
    return label_name, confidence, probabilities


# -----------------------------
# Main program
# -----------------------------
def main():
    torch.manual_seed(42)  # reproducible results

    # Build vocabulary
    vocab = build_vocabulary(data)

    # Find max sequence length
    max_length = max(len(tokenize(text)) for text, _ in data)

    # Prepare tensors
    X_tensor, y_tensor = prepare_data(data, vocab, max_length)

    print("Vocabulary:")
    print(vocab)

    print("\nEncoded Inputs:")
    print(X_tensor)

    print("\nLabels:")
    print(y_tensor)

    # Hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 10
    hidden_dim = 8
    output_dim = 2
    learning_rate = 0.01
    epochs = 50

    # Create model
    model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

    print("\nModel Architecture:")
    print(model)

    # Loss function (applies softmax internally, so the model outputs logits)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nTraining...")
    for epoch in range(epochs):
        model.train()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(X_tensor)

        # Compute loss
        loss = criterion(logits, y_tensor)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_tensor).float().mean().item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")

    # Test predictions
    test_sentences = [
        "I love this",
        "This is bad",
        "I do not like this",
        "This was amazing",
        "This was terrible",
    ]

    print("\nPredictions:")
    for sentence in test_sentences:
        label, confidence, probabilities = predict_sentiment(sentence, model, vocab, max_length)
        print(f"Text: {sentence}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {probabilities.tolist()}")
        print("-" * 50)


if __name__ == "__main__":
    main()