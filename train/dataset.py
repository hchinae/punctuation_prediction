import json
import os

import torch
from torch.utils.data import Dataset

from config import MAX_SEQ_LEN, PADDING_IDX, UNK_IDX, VOCAB_PATH
from preprocessing.preprocess_data import (
    get_punctuation_marker, get_punctuation_signs_for_prediction,
    preprocess_file)


class PunctuationDataset(Dataset):
    """
    Dataset that loads tokenized input and punctuation labels from .txt files
    preprocessed using the provided `preprocess_file()` function.
    """
    def __init__(self, filepaths):
        self.samples = []
        self.marker = get_punctuation_marker()

        # Load vocab
        with open(VOCAB_PATH, "r") as f:
            self.vocab = json.load(f)
        self.word2idx = self.vocab
        self.idx2word = {v: k for k, v in self.vocab.items()}

        # Load all input-label pairs
        for path in filepaths:
            inputs, labels = preprocess_file(path)
            for x, y in zip(inputs, labels):
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, puncts = self.samples[idx]
        return tokens, puncts


def collate_fn(batch, vocab):
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
    punct2idx = {p: i for i, p in enumerate(get_punctuation_signs_for_prediction())}

    batch_input_ids = []
    batch_target = []
    batch_mask = []

    for tokens, labels in batch:
        input_ids = []
        mask = []
        target = []
        label_idx = 0

        for tok in tokens[:MAX_SEQ_LEN]:
            if tok == marker:
                input_ids.append(vocab.get(marker, UNK_IDX))
                mask.append(1)
                target.append(punct2idx[labels[label_idx]])
                label_idx += 1
            else:
                input_ids.append(vocab.get(tok, UNK_IDX))
                mask.append(0)

        # Padding if needed
        pad_len = MAX_SEQ_LEN - len(input_ids)
        input_ids += [PADDING_IDX] * pad_len
        mask += [0] * pad_len

        batch_input_ids.append(input_ids)
        batch_mask.append(mask)
        batch_target.append(target)

    input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(batch_mask, dtype=torch.bool)
    return input_ids_tensor, batch_target, mask_tensor
