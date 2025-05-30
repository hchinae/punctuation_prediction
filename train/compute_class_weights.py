# train/compute_class_weights.py

import json
import os
from collections import Counter

import numpy as np

import config


def load_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [label for _, labels in data for label in labels]


def compute_weights(label_list):
    counts = Counter(label_list)
    total = sum(counts.values())
    num_classes = len(counts)
    return {label: total / (num_classes * count) for label, count in counts.items()}, counts


def build_label2id(counts):
    labels = sorted(counts)
    return {label: idx for idx, label in enumerate(labels)}


def main():
    os.makedirs("data", exist_ok=True)

    labels = load_labels(config.TRAIN_JSON_PATH)
    weights, counts = compute_weights(labels)
    label2id = build_label2id(counts)

    with open(config.LABEL2ID_PATH, "w") as f:
        json.dump(label2id, f)

    id2weight = [weights[label] for label in sorted(label2id.keys())]
    np.save(config.CLASS_WEIGHTS_PATH, np.array(id2weight, dtype=np.float32))

    print("Saved:")
    print(f"- Class weights to {config.CLASS_WEIGHTS_PATH}")
    print(f"- Label2ID to {config.LABEL2ID_PATH}")
    for label, idx in label2id.items():
        print(f"{repr(label):>5} -> {idx}, weight: {weights[label]:.4f}")


if __name__ == "__main__":
    main()
