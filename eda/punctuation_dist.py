import os
from collections import Counter

import matplotlib.pyplot as plt

from utils.preprocess_data import (get_punctuation_signs_for_prediction,
                                   preprocess_file)


def analyze_punctuation_distribution(data_dir):
    all_labels = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(data_dir, filename)
        _, labels = preprocess_file(file_path)

        for label_seq in labels:
            all_labels.extend(label_seq)

    counter = Counter(all_labels)
    punctuation_set = get_punctuation_signs_for_prediction()

    # Ensure all punctuation classes are covered
    frequencies = [counter.get(p, 0) for p in punctuation_set]

    print("Punctuation Frequencies:")
    for p, freq in zip(punctuation_set, frequencies):
        print(f"{p:>2}: {freq}")

    plt.figure(figsize=(10, 5))
    plt.bar(punctuation_set, frequencies, color='skyblue')
    plt.xlabel("Punctuation")
    plt.ylabel("Frequency")
    plt.title("Punctuation Distribution in Test Set")
    plt.tight_layout()
    prefix = data_dir.split("/")[-1]
    file_path = f"report/{prefix}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    filename = f"{prefix}_eda_punctuation_distribution.png"
    print(f"Saving plot to {file_path}")
    plt.title(f"Punctuation Distribution in {prefix} Set")
    plt.savefig(f"{file_path}/{filename}")
    plt.close()

if __name__ == "__main__":
    analyze_punctuation_distribution("data/eval")
