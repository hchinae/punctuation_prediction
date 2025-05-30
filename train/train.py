import json
import os
import random
from functools import partial
from glob import glob

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_GRAD_NORM,
                    MODEL_SAVE_PATH, PADDING_IDX, SEED, TRAIN_DIR,
                    TRAIN_JSON_PATH, UNK_IDX, VAL_JSON_PATH, VOCAB_PATH)
from eval.evaluate import evaluate
from model.baseline_bilstm import BiLSTMPunctuator
from preprocessing.preprocess_data import get_punctuation_signs_for_prediction
from train.dataset import PunctuationDataset, collate_fn

class_labels = get_punctuation_signs_for_prediction()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_masked_loss(logits, targets, mask):
    B, L, C = logits.shape
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []

    for i in range(B):
        sample_logits = logits[i][mask[i]]
        if sample_logits.shape[0] == 0:
            continue
        sample_targets = torch.tensor(targets[i], dtype=torch.long).to(logits.device)
        loss = loss_fn(sample_logits, sample_targets)
        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True).to(logits.device)
    return torch.stack(losses).mean()


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    num_classes = len(class_labels)

    # all_files = glob(os.path.join(TRAIN_DIR, "*.txt"))
    # assert len(all_files) >= 2, "Need at least two files for train/validation split"

    # train_files, val_files = train_test_split(all_files, test_size=1, random_state=SEED)

    # print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    # #print the name of the val file
    # print(f"Validation files: {val_files}")

    # train_dataset = PunctuationDataset(train_files)
    # val_dataset = PunctuationDataset(val_files)
    train_dataset = PunctuationDataset(json_path=TRAIN_JSON_PATH)
    val_dataset = PunctuationDataset(json_path=VAL_JSON_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, vocab=vocab)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, vocab=vocab)
    )

    model = BiLSTMPunctuator(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=num_classes,
        pad_idx=PADDING_IDX
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for input_ids, targets, mask in train_loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            logits, _ = model(input_ids)
            loss = compute_masked_loss(logits, targets, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_f1 = evaluate(model, train_loader, device)
        val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        


        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model to {MODEL_SAVE_PATH}")
            evaluate(model, val_loader, device, class_labels=class_labels, plot=True, plot_dir="report")


    print(f"Training complete. Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()
