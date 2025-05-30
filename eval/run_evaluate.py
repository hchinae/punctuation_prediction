import argparse
import glob
import json
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, DEVICE, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN,
                    MODEL_SAVE_PATH, PADDING_IDX, VOCAB_PATH)
from eval.evaluate import evaluate
from model.baseline_bilstm import BiLSTMPunctuator
from preprocessing.preprocess_data import get_punctuation_signs_for_prediction
from train.dataset import PunctuationDataset, collate_fn


def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        return json.load(f)


def get_files_from_split(split):
    folder_map = {
        "test": "data/test",
        "proxy_test": "data/proxy_test",
        "train": "data/train"
    }
    if split not in folder_map:
        raise ValueError(f"Unsupported split: {split}. Must be one of: {list(folder_map)}")

    folder_path = folder_map[split]
    filepaths = glob.glob(os.path.join(folder_path, "*.txt"))
    if not filepaths:
        raise FileNotFoundError(f"No .txt files found in {folder_path}")
    return filepaths, f"report/{split}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True,
                        help="Which dataset split to evaluate: test, proxy_test, train")
    args = parser.parse_args()

    vocab = load_vocab()
    class_labels = get_punctuation_signs_for_prediction()

    model = BiLSTMPunctuator(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(class_labels),
        pad_idx=PADDING_IDX
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    filepaths, plot_dir = get_files_from_split(args.split)
    dataset = PunctuationDataset(filepaths=filepaths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, vocab=vocab))

    f1 = evaluate(model, loader, DEVICE, class_labels=class_labels, plot=True, plot_dir=plot_dir)
    print(f"Macro F1 for split '{args.split}': {f1:.4f}")


if __name__ == "__main__":
    main()
