import json
import os
import random
from glob import glob

from sklearn.model_selection import train_test_split

from config import SEED, TRAIN_DIR, VAL_RATIO
from preprocessing.preprocess_data import (
    get_punctuation_signs_for_prediction, preprocess_file)

random.seed(SEED)


def find_single_file_in_dir(directory):
    files = glob(os.path.join(directory, "*"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one .txt file in {directory}, found {len(files)}.")
    return files[0]


def main():
    input_file = find_single_file_in_dir(TRAIN_DIR)

    print(f"Loading and preprocessing: {input_file}")
    inputs, targets = preprocess_file(input_file)
    samples = list(zip(inputs, targets))
    print(f"Loaded {len(samples)} samples")

    # Shuffle and split
    train_samples, val_samples = train_test_split(
        samples,
        test_size=VAL_RATIO,
        random_state=SEED
    )

    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/train.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f)
    with open("data/processed/val.json", "w", encoding="utf-8") as f:
        json.dump(val_samples, f)

    print("✅ Saved train/val split to data/processed/")

if __name__ == "__main__":
    main()
