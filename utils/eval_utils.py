import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import matplotlib.pyplot as plt


def plot_classification_report(report_dict, label2id, save_path=None):
    class_labels = []
    f1_scores = []

    for label in sorted(label2id, key=label2id.get):
        if label in report_dict:
            class_labels.append(label)
            f1_scores.append(report_dict[label]['f1-score'])

    plt.figure(figsize=(10, 5))
    plt.bar(class_labels, f1_scores, color='skyblue')
    plt.ylim([0, 1])
    plt.xlabel("Punctuation Label")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Score")
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path+"/classification_report.png", bbox_inches='tight')
    else:
        plt.show()


def plot_confusion_matrix(labels, preds, class_labels, save_dir=None):
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
    plt.close()

    
