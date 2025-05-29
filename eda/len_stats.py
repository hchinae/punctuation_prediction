import os

import matplotlib.pyplot as plt

from preprocessing.preprocess_data import preprocess_file


def analyze_sequence_lengths(data_dir):
    lengths = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(data_dir, filename)
        inputs, _ = preprocess_file(file_path)

        for sequence in inputs:
            lengths.append(len(sequence))

    print(f"Total sequences: {len(lengths)}")
    print(f"Average length: {sum(lengths) / len(lengths):.2f}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Mode length: {max(set(lengths), key=lengths.count)}")
    print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
    print(f"Standard deviation: {sum((x - (sum(lengths) / len(lengths))) ** 2 for x in lengths) ** 0.5 / len(lengths):.2f}")
    print(f"Length range: {min(lengths)} - {max(lengths)}")
    print(f"Length variance: {sum((x - (sum(lengths) / len(lengths))) ** 2 for x in lengths) / len(lengths):.2f}")
    print(f"Length percentiles: 25th: {sorted(lengths)[len(lengths) // 4]}, 50th: {sorted(lengths)[len(lengths) // 2]}, 75th: {sorted(lengths)[3 * len(lengths) // 4]}")

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=30, color='salmon', edgecolor='black')
    plt.xlabel("Token Sequence Length")
    plt.ylabel("Count")
    plt.title("Distribution of Tokenized Paragraph Lengths")
    plt.tight_layout()
    plt.savefig("eda/sequence_lengths.png")
    plt.close()

if __name__ == "__main__":
    analyze_sequence_lengths("data/test")
