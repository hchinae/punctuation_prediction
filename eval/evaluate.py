import torch
from sklearn.metrics import f1_score


def evaluate(model, data_loader, device):
    """
    Evaluate a model on a given DataLoader.
    
    Args:
        model: torch.nn.Module
        data_loader: torch.utils.data.DataLoader
        device: 'cuda' or 'cpu'

    Returns:
        Macro F1 score
    """
    model.eval()
    all_logits = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for input_ids, targets, mask in data_loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            logits, _ = model(input_ids)
            all_logits.append(logits.cpu())
            all_targets.extend(targets)
            all_masks.append(mask.cpu())

    return compute_f1(torch.cat(all_logits), all_targets, torch.cat(all_masks))


def compute_f1(logits, targets, mask):
    """
    Args:
        logits: [B, L, C] tensor
        targets: list[list[int]]
        mask: [B, L] tensor (bool)

    Returns:
        macro F1 score
    """
    preds = []
    labels = []

    for i in range(logits.shape[0]):
        sample_logits = logits[i][mask[i]]
        if sample_logits.shape[0] == 0:
            continue
        pred = sample_logits.argmax(dim=-1).tolist()
        label = targets[i]
        preds.extend(pred)
        labels.extend(label)

    if len(labels) == 0:
        return 0.0

    return f1_score(labels, preds, average="macro")

import numpy as np
from sklearn.metrics import classification_report


def compute_per_class_f1(logits, targets, mask, class_labels):
    preds = []
    labels = []

    for i in range(logits.shape[0]):
        sample_logits = logits[i][mask[i]]
        if sample_logits.shape[0] == 0:
            continue
        pred = sample_logits.argmax(dim=-1).tolist()
        label = targets[i]
        preds.extend(pred)
        labels.extend(label)

    report = classification_report(labels, preds, target_names=class_labels, digits=4, zero_division=0)
    print(report)
    return report
