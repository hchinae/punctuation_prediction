
import random

import numpy as np
import torch

from model.baseline_bilstm import BiLSTMPunctuator
from utils.data_utils import prepare_train_val_data
from utils.eval_utils import evaluate
from utils.preprocess_data import get_punctuation_signs_for_prediction

class_labels = get_punctuation_signs_for_prediction()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_masked_loss(logits, targets, mask, loss_fn):
    B, L, C = logits.shape
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

def train_model(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, vocab = prepare_train_val_data(config)
    vocab_size = len(vocab)

    vocab_size = len(vocab)
    num_classes = len(class_labels)

    model = BiLSTMPunctuator(
        vocab_size=vocab_size,
        embedding_dim=config["EMBEDDING_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        output_dim=num_classes,
        pad_idx=config["PADDING_IDX"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["LEARNING_RATE"]))
    best_val_f1 = 0.0
    if config["USE_CLASS_WEIGHTS"]:
        class_weights = np.load(config["CLASS_WEIGHTS_PATH"])
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, config["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0

        for input_ids, targets, mask in train_loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            logits, _ = model(input_ids)
            loss = compute_masked_loss(logits, targets, mask, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["MAX_GRAD_NORM"])
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_macro_f1 = evaluate(model, train_loader, device)
        val_macro_f1  = evaluate(model, val_loader, device)


        print(f"\nEpoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Train macro F1: {train_macro_f1:.4f} | Val macro F1: {val_macro_f1:.4f}")


        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"Saved new best model to {config['MODEL_SAVE_PATH']}")
            evaluate(model, val_loader, device, class_labels=class_labels, plot=True, plot_dir=f"report/{config['mode']}")

    print(f"Training complete. Best validation F1: {best_val_f1:.4f}")
