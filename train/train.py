import json
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_GRAD_NORM,
                    MODEL_SAVE_PATH, PADDING_IDX, SEED, TRAIN_DIR, UNK_IDX,
                    VOCAB_PATH)
from model.baseline_bilstm import BiLSTMPunctuator
from preprocessing.preprocess_data import get_punctuation_signs_for_prediction
from train.dataset import PunctuationDataset, collate_fn


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_masked_loss(logits, targets, mask):
    """
    Args:
        logits: [B, L, C]
        targets: list of list[int] — variable # of punctuation labels per sample
        mask: [B, L] — bool, True at <punctuation> tokens

    Returns:
        CrossEntropyLoss at only <punctuation> positions
    """
    B, L, C = logits.shape
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []

    for i in range(B):
        sample_logits = logits[i][mask[i]]  # shape: [num_markers_i, C]
        if sample_logits.shape[0] == 0:
            continue  # skip samples with no markers
        sample_targets = torch.tensor(targets[i], dtype=torch.long).to(logits.device)
        loss = loss_fn(sample_logits, sample_targets)
        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load vocab
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    num_classes = len(get_punctuation_signs_for_prediction())

    # Load training data
    train_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith(".txt")]
    train_dataset = PunctuationDataset(train_files)
    data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, vocab=vocab)
    )

    # Init model
    model = BiLSTMPunctuator(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=num_classes,
        pad_idx=PADDING_IDX
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in data_loader:
            input_ids, targets, mask = batch
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            logits, _ = model(input_ids, mask)  # [B, L, C]
            loss = compute_masked_loss(logits, targets, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        # Save best model by loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved new best model to {MODEL_SAVE_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    train()
