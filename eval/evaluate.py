import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate(model, data_loader, device, class_labels=None, plot=False, plot_dir=None):
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

    logits = torch.cat(all_logits)
    mask = torch.cat(all_masks)
    targets = all_targets  # already a flat list of lists

    f1 = compute_f1(logits, targets, mask)

    if plot and class_labels:
        plot_classification_report(logits, targets, mask, class_labels, plot_dir)
        plot_confusion_matrix(logits, targets, mask, class_labels, plot_dir)

    return f1


def compute_f1(logits, targets, mask):
    from sklearn.metrics import f1_score
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


def plot_classification_report(logits, targets, mask, class_labels, save_dir=None):
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

    report = classification_report(labels, preds, target_names=class_labels, output_dict=True, zero_division=0)
    f1_scores = [report[cls]['f1-score'] for cls in class_labels]

    #print for each cls what is the f1 score
    for cls, f1 in zip(class_labels, f1_scores):
        print(f"F1 score for {cls}: {f1:.4f}")
        
    plt.figure(figsize=(10, 4))
    plt.bar(class_labels, f1_scores, color='skyblue')
    plt.title("Per-class F1 scores")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    if save_dir:
        path = f"{save_dir}/per_class_f1.png"
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved F1 plot to {path}")
    else:
        plt.show()


def plot_confusion_matrix(logits, targets, mask, class_labels, save_dir=None):
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

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_labels))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_dir:
        path = f"{save_dir}/confusion_matrix.png"
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved confusion matrix to {path}")
    else:
        plt.show()
