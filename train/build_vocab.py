import json
import os
from collections import Counter

from config import MIN_FREQ, VOCAB_PATH
from preprocessing.preprocess_data import preprocess_file

SPECIAL_TOKENS = ["<pad>", "<unk>"]

def build_vocab_from_dir(train_dir):
    counter = Counter()

    for filename in os.listdir(train_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(train_dir, filename)
            inputs, _ = preprocess_file(filepath)
            for line in inputs:
                for token in line:
                    counter[token] += 1

    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for word, freq in counter.items():
        if freq >= MIN_FREQ and word not in vocab:
            vocab[word] = len(vocab)

    # Save vocab
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Saved vocab with {len(vocab)} tokens to {VOCAB_PATH}")
    return vocab
