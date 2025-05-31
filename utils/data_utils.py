import json
import os
from collections import Counter
from functools import partial
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.dataset import (PunctuationDataset, collate_fn,
                           load_samples_from_text_folder)
from utils.preprocess_data import preprocess_file


def load_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [label for _, labels in data for label in labels]


def compute_weights(label_list):
    counts = Counter(label_list)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = {label: total / (num_classes * count) for label, count in counts.items()}
    return weights, counts


def build_label2id(counts):
    labels = sorted(counts)
    return {label: idx for idx, label in enumerate(labels)}


def compute_class_weights_and_label2id(train_json_path):
    labels = load_labels(train_json_path)
    weights, counts = compute_weights(labels)
    label2id = build_label2id(counts)

    id2weight = [weights[label] for label in sorted(label2id.keys())]
    return label2id, np.array(id2weight, dtype=np.float32)


def build_vocab_from_dir(train_dir, min_freq):
    counter = Counter()
    for filename in os.listdir(train_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(train_dir, filename)
            inputs, _ = preprocess_file(filepath)
            for line in inputs:
                for token in line:
                    counter[token] += 1

    special_tokens = ["<pad>", "<unk>"]
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab



    samples = list(zip(all_inputs, all_targets))
    print(f"Loaded {len(samples)} samples.")
    return samples

def prepare_train_val_data(config):
    print("Processing training data...")
    # Load and aggregate all input/target pairs

    samples = load_samples_from_text_folder(config["TRAIN_DIR"])

    train_samples, val_samples = train_test_split(
        samples, test_size=config["VAL_RATIO"], random_state=config["SEED"]
    )

    # Save splits
    os.makedirs("data/processed", exist_ok=True)
    with open(config["TRAIN_JSON_PATH"], "w") as f:
        json.dump(train_samples, f)
    with open(config["VAL_JSON_PATH"], "w") as f:
        json.dump(val_samples, f)

    print("Saved train/val split.")

    label2id, weight_array = compute_class_weights_and_label2id(config["TRAIN_JSON_PATH"])

    # Save to disk
    with open(config["LABEL2ID_PATH"], "w") as f:
        json.dump(label2id, f)
    np.save(config["CLASS_WEIGHTS_PATH"], weight_array)
    print(f"Saved class weights to {config['CLASS_WEIGHTS_PATH']}")
    
    with open(config["LABEL2ID_PATH"], "w") as f:
        json.dump(label2id, f)

    print(f"Saved label2id to {config['LABEL2ID_PATH']}")

    #after the split, we can build the vocabulary
    vocab = build_vocab_from_dir(config["TRAIN_DIR"], config["MIN_FREQ"])
    # Save vocab
    with open(config["VOCAB_PATH"], "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab with {len(vocab)} tokens to {config['VOCAB_PATH']}")

    # Create Datasets and Loaders
    train_dataset = PunctuationDataset(config, val=False)
    val_dataset = PunctuationDataset(config, val=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        collate_fn=partial(collate_fn, vocab=vocab, config=config)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        collate_fn=partial(collate_fn, vocab=vocab, config=config)
    )

    return train_loader, val_loader, vocab


def prepare_test_loader(config):
    # Load vocab
    with open(config["VOCAB_PATH"], "r") as f:
        vocab = json.load(f)

    test_dataset = PunctuationDataset(config)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        collate_fn=partial(collate_fn, vocab=vocab)
    )

    return test_loader, vocab
