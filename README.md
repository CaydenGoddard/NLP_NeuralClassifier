NLP_NeuralClassifier

A small neural network sentiment classifier built with PyTorch. It learns its own word representations with an embedding layer and a hidden layer instead of using hand-built features.

ARCHITECTURE

Embedding -> Mean Pooling -> Linear -> ReLU -> Linear -> Output (logits)

Embedding: nn.Embedding(vocab_size, 10, padding_idx=0) turns each word index into a learned 10-number vector.
Mean pooling: averages the word vectors (ignoring PAD) into one sentence vector.
Hidden layer: nn.Linear(10, 8) followed by nn.ReLU().
Output layer: nn.Linear(8, 2) gives logits for negative and positive.
Softmax turns the logits into probabilities at prediction time.

HOW IT WORKS

1. Preprocess: lowercase each sentence, split it into words, and build a vocabulary (0 = PAD, 1 = UNK).
2. Encode: convert each sentence into word indices and pad to the longest sentence.
3. Train: CrossEntropyLoss with the Adam optimizer (learning rate 0.01) for 50 epochs.
4. Predict: run a new sentence through the model, apply softmax, and report the label and confidence.

DATASET

The 6 sentences from the assignment plus 12 added examples (18 total, 9 positive and 9 negative). Labels: 1 = positive, 0 = negative. None of the test sentences appear in the training data.

SETUP AND RUN

pip install torch numpy
python neural_sentiment.py

RESULTS

Training output (loss decreases and accuracy reaches 100% on the training set):

Epoch 10/50 - Loss: 0.5895 - Accuracy: 0.7222
Epoch 20/50 - Loss: 0.4156 - Accuracy: 0.8889
Epoch 30/50 - Loss: 0.2042 - Accuracy: 1.0000
Epoch 40/50 - Loss: 0.0664 - Accuracy: 1.0000
Epoch 50/50 - Loss: 0.0183 - Accuracy: 1.0000

Test predictions:

I love this -> positive (confidence 0.9964)
This is bad -> positive (wrong, confidence 0.9958)
I do not like this -> negative (confidence 0.9912)
This was amazing -> positive (confidence 0.8925)
This was terrible -> negative (confidence 1.0000)

LIMITATIONS

- Tiny dataset: with only 18 sentences, the model can memorize them. "This is bad" was misclassified because "bad" appears in very few training examples, so the common words "this" and "is" outweighed it.
- No word order: mean pooling averages the word vectors, so the model can't directly tell that "not like" is a pair. It can only learn it from examples.
- Fixes to try: more training data, more varied contexts for key words like "bad" and "good", or a model that keeps word order (such as a transformer).
