import json
import os
from glob import glob

import torch
from torch.utils.data import Dataset

from utils.preprocess_data import get_punctuation_marker, preprocess_file


def load_samples_from_text_folder(data_folder):
    all_inputs, all_targets = [], []
    input_files = glob(os.path.join(data_folder, "*.txt"))
    print(f"Found {len(input_files)} text files in {data_folder}")
    for input_file in input_files:
        inputs, targets = preprocess_file(input_file)
        all_inputs.extend(inputs)
        all_targets.extend(targets)
    samples = list(zip(all_inputs, all_targets))
    print(f"Loaded {len(samples)} samples from {data_folder}.")
    return samples

class PunctuationDataset(Dataset):
    def __init__(self, config, val=None):
        self.samples = []
        self.marker = get_punctuation_marker()

        with open(config["VOCAB_PATH"], "r") as f:
            self.vocab = json.load(f)

        if config["mode"] == "train":
            if val:
                with open(config["VAL_JSON_PATH"], "r") as f:
                    self.samples = json.load(f)
            else:
                with open(config["TRAIN_JSON_PATH"], "r") as f:
                    self.samples = json.load(f)
        elif config["mode"] == "eval":
            #for all files in the test directory
            self.samples = load_samples_from_text_folder(config["TEST_DIR"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, puncts = self.samples[idx]
        return tokens, puncts


def collate_fn(batch, vocab, config):
    """
    Custom collate function for padding, input ID conversion, and <punctuation> masking.

    Args:
        batch: list of (tokens, labels)
        vocab: dict mapping token → index

    Returns:
        input_ids: [B, L] tensor
        target_labels: List[List[int]] — one list per sample, len = #<punctuation> markers
        mask: [B, L] tensor (bool) — True at <punctuation> tokens
    """
    marker = get_punctuation_marker()
    with open(config["LABEL2ID_PATH"], "r") as f:
        punct2idx = json.load(f)

    batch_input_ids = []
    batch_target = []
    batch_mask = []

    for tokens, labels in batch:
        input_ids = []
        mask = []
        target = []
        label_idx = 0

        for tok in tokens[:config["MAX_SEQ_LEN"]]:
            if tok == marker:
                input_ids.append(vocab.get(marker, config["UNK_IDX"]))
                mask.append(1)
                target.append(punct2idx[labels[label_idx]])
                label_idx += 1
            else:
                input_ids.append(vocab.get(tok, config["UNK_IDX"]))
                mask.append(0)

        # Padding if needed
        pad_len = config["MAX_SEQ_LEN"] - len(input_ids)
        input_ids += [config["PADDING_IDX"]] * pad_len
        mask += [0] * pad_len

        batch_input_ids.append(input_ids)
        batch_mask.append(mask)
        batch_target.append(target)

    input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(batch_mask, dtype=torch.bool)
    return input_ids_tensor, batch_target, mask_tensor
