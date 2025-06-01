import json
import os
from glob import glob

import yaml

from train import train_model  # From our previous step
from utils.dataset import \
    PunctuationDataset  # You'll create this file if not done
from utils.preprocess_data import preprocess_file
from utils.seed import set_seed  # Optional: for reproducibility


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        return json.load(f)

def load_label_map(label_path):
    with open(label_path, "r") as f:
        return json.load(f)

def main():
    config = load_config()
    set_seed(config["SEED"])  # Optional for reproducibility

    print("Loading vocab and labels...")
    vocab = load_vocab(config["VOCAB_PATH"])
    label2id = load_label_map(config["LABEL2ID_PATH"])

    print("Preprocessing training data...")
    train_files = glob(os.path.join(config["TRAIN_DIR"], "*.txt"))
    val_files = glob(os.path.join(config["VAL_DIR"], "*.txt"))

    train_inputs, train_targets = [], []
    for file in train_files:
        inp, tgt = preprocess_file(file)
        train_inputs.extend(inp)
        train_targets.extend(tgt)

    val_inputs, val_targets = [], []
    for file in val_files:
        inp, tgt = preprocess_file(file)
        val_inputs.extend(inp)
        val_targets.extend(tgt)

    print(f"Loaded {len(train_inputs)} training and {len(val_inputs)} validation sequences.")

    train_dataset = PunctuationDataset(train_inputs, train_targets, vocab, label2id, config)
    val_dataset = PunctuationDataset(val_inputs, val_targets, vocab, label2id, config)

    print("Starting training...")
    train_model(config, train_dataset, val_dataset, label2id)

    print(f"Training complete. Model saved to {config['MODEL_SAVE_PATH']}")

if __name__ == "__main__":
    main()
