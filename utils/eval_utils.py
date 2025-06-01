import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import numpy

from utils.preprocess_data import get_punctuation_signs_for_prediction


def plot_classification_report(report, class_labels, save_dir=None):
    
    # Prepare full label list (all indices)
    all_class_indices = list(range(len(class_labels)))
    # Print macro and weighted average F1
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    print(f"\nMacro Avg F1: {macro_f1:.4f}")
    print(f"Weighted Avg F1: {weighted_f1:.4f}")
    
    # Build f1_scores with fallback for missing classes
    f1_scores = []
    for cls in class_labels:
        if cls in report:
            f1_scores.append(report[cls]['f1-score'])
        else:
            f1_scores.append(0.0)  # Class missing in both preds and labels

    # # Detect missing classes for logging
    # missing = set(all_class_indices) - set(labels) - set(preds)
    # if missing:
    #     print(f"Warning: These classes were missing in both predictions and labels: {missing}")

    # Print per-class F1
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

    
def evaluate_model(model, inputs, labels):

    # Obtain the model's confusion matrix
    classes = get_punctuation_signs_for_prediction()
    conf_matrix = numpy.zeros((len(classes), len(classes)), dtype="int32")

    all_predictions = []
    all_golds = []

    for input, label in zip(inputs, labels):
        
        # Get the model's prediction
        prediction = model.predict(input)
 
        # Ensure that the predictions are valid
        assert len(prediction) == len(label), f"Invalid number of predictions  pred: {len(prediction)}, gold: {len(label)} for input: {input}"
        assert all(isinstance(p, str) for p in prediction), "Model predicted non-string punctuation signs: {}".format(prediction)
        for p in prediction:
            assert p in classes, "Model predicted an invalid punctuation sign: {}".format(p)

        # Populate the confusion matrix
        for p, l in zip(prediction, label):
            conf_matrix[classes.index(l), classes.index(p)] += 1
            all_predictions.append(p)
            all_golds.append(l)


    print(f"Gold labels: {label[:5]}")
    # Compute the generalized F1 score from the confusion matrix
    class_f1_scores = []
    for i, punctuation in enumerate(classes):
        precision = (conf_matrix[i, i] + 1e-6) / (conf_matrix[:, i].sum() + 1e-6)
        recall = (conf_matrix[i, i] + 1e-6) / (conf_matrix[i, :].sum() + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        class_f1_scores.append(f1)

    f1_score = sum(class_f1_scores) / len(class_f1_scores)
    return f1_score,  all_golds, all_predictions, conf_matrix




