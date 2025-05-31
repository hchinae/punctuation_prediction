# Punctuation Restoration with BiLSTM: A Baseline for Literary Texts

## Abstract

In the task of recovering missing punctuation from literary texts, we implemented a baseline model using a deep neural network architecture. Our model consists of a Bidirectional Long Short-Term Memory (BiLSTM) followed by a linear classifier that predicts 9 types of punctuation marks commonly removed from *Sherlock Holmes* stories. 

On a randomly selected validation set drawn from the training data, our model achieves a **macro F1 score of 49%** and a **weighted average F1 score of 80%**. On the test set (comprising three unseen Sherlock Holmes stories), the model scores a **macro F1 of 44%** and a **weighted F1 of 75%**. In future work, we plan to train on larger and more diverse text sources, and incorporate richer architectures such as Transformers to better capture long-range dependencies in the text.

---

## Introduction

The goal of this project is to restore missing punctuation marks from a given input sequence of words using deep learning. The challenge is framed as a sequence labeling problem, where each `<punctuation>` placeholder must be replaced with the correct punctuation mark.

The evaluation is performed on three specific stories by Arthur Conan Doyle:

- *A Scandal in Bohemia*
- *The Red-headed League*
- *A Case of Identity*

To avoid data leakage, we were instructed to train only on **legal public-domain data** distinct from these test stories. We curated our training data primarily from *The Adventures of Sherlock Holmes* and avoided overlapping chapters with the test set.

---

## Exploratory Data Analysis (EDA)

Before model development, we performed an extensive exploratory data analysis to better understand the structure of the data. This included computing:

- Sequence length distributions
- Per-class punctuation frequencies
- Distributional differences between train, test, and validation sets

### Punctuation Frequencies in Test Set and Test Set Statistics

<p float="left">
  <img src="test/test_eda_punctuation_distribution.png" width="500"/>
  <img src="test/test_eda_sequence_length_distribution.png" width="500"/>
</p>

---

## Data pre-processing

Initially, we included various public-domain books (e.g., *Dracula*, *Pride and Prejudice*), but performance degraded slightly, likely due to style mismatches. We ultimately restricted the training data to Sherlock Holmes stories *excluding the test chapters*. The goal is the training data be similar distribution to test data.
### Punctuation Frequencies in Train Set and Train Set Statistics

<p float="left">
  <img src="train/train_eda_punctuation_distribution.png" width="500"/>
  <img src="train/train_eda_sequence_length_distribution.png" width="500"/>
</p>

The training data is about three times of testing data which we need to augment it in the future iteration. We preprocessed each story by:

- Removing Gutenberg boilerplate headers and footers
- Replacing ASCII quotes with UTF-8 equivalents
- Tokenizing and inserting `<punctuation>` markers in place of real punctuation
- Building JSON-based datasets containing `[tokens, labels]` pairs

---

## Model Architecture

Given the sequential nature of the task, we implemented a **BiLSTM** model — a natural fit for capturing context around missing punctuation points.

### Components

- **Embedding layer**: maps tokens to 128-dimensional vectors
- **BiLSTM layer**: hidden size 192, bidirectional
- **Dropout**: 0.4 for regularization
- **Linear classifier**: outputs logits over 9 punctuation classes

Input → Embedding → BiLSTM → Dropout → Linear → Softmax

## Data Pipline

During training, a small validation set is randomly selected from the training set. The vocabulary is built only from the training data (excluding validation), as the validation set is intended to mimic the unseen testing data distribution.

To prepare training data, we convert each story into sequences of tokens (words) and their corresponding punctuation labels. Two special tokens are used:
- <punctuation>: a placeholder for where a punctuation mark was originally present
- <UNK>: used for out-of-vocabulary tokens during validation and testing

["yesterday", "<punctuation>", "i", "went", "to", "<UNK>", "park", "<punctuation>"]
Labels: [",", "."]

During training:
- Vocabulary (token-to-ID mapping) and punctuation label mappings are constructed.
- A custom collate function is used to pad sequences or truncate them to the maximum allowed length.

During evaluation (and inference), the data pipeline processes .txt files to:
- Tokenize sentences
- Detect punctuation positions
- Replace punctuation with <punctuation> placeholders
- Generate ground truth labels for comparison with model predictions

This unified pipeline ensures that training and evaluation data are consistently formatted and compatible with the model.

## Experiments
We began our experiments with standard parameters and applied light tuning to improve performance. Adjustments included increasing dropout to reduce overfitting and lowering the hidden layer size.

Final Hyperparameters:
- EMBEDDING_DIM: 128
- HIDDEN_DIM: 192
- DROPOUT: 0.4
- BATCH_SIZE: 32
- EPOCHS: 12
- LEARNING_RATE: 3e-4

We also implemented:
- A learning rate scheduler (starting at 3e-4, reducing on plateau)
- An early stopping mechanism with patience = 3

## Results

We evaluate our model on both validation (from train split), and the official test set. The validation set results are:
- Macro F1: 49%
- Weighted F1: 80%

The per class f1 on test set are shown here:

<img src="test/per_class_f1.png" width="700"/>

The test set f1 scores are:
Test Set
- Macro F1: 44%
- Weighted F1: 75%

These results indicate a solid baseline, but highlight the challenge of rare class prediction in imbalanced literary data.

## Error Analysis

To investigate the results further, we show the test set confusion matrix here:

<img src="test/confusion_matrix.png" width="600"/>

The model performs well on frequent punctuation marks such as ", ., and ,. However, performance drops significantly for rare punctuation classes like :, ;, (, and ). Common confusions include ":" and "." as well as ";" and ",".

For example, the colon : appears in three distinct contexts in the test set:

- In time expressions (e.g., 5:15)

- As an introducer of quoted speech (e.g., "The Coroner: Did your father say anything?")

- To elaborate or emphasize a preceding clause

However, these are rare cases both in training and test sets. To improve performance on such cases, we propose the following steps:

- Increasing Training Data size: Identify and incorporate training texts that exhibit similar narrative structures and punctuation usage to the test data. For example, classic literature, or dialog-heavy prose that mimics the stylistic tone of Conan Doyle’s writing.

- Transformer-based Architectures: While BiLSTM is effective for capturing short-range dependencies, attention-based architectures like transformers (e.g., BERT, Longformer) can model longer sequences and are likely to capture global context more effectively. This could improve the model’s ability to disambiguate punctuation that depends on broader sentence structure.

- Character-level Modeling: Rare punctuation can often be tied to subtle lexical or formatting cues. A character-level model could potentially learn finer-grained patterns — such as stylized names, abbreviations, or typography — that word-level models miss.


## Relevant Literature
A brief survey of the literature shows that deep neural networks have been widely explored for punctuation prediction tasks, often with strong results in domain-specific applications:

- Zelasko et al. (2018)
Punctuation Prediction Model for Conversational Speech
https://arxiv.org/abs/1807.00543
This paper explores two deep learning models — a BiLSTM and a CNN — for punctuation restoration in conversational speech, trained on the Fisher corpus. Their approach treats punctuation prediction as a sequence labeling problem.

- Xiaoyin Che et al. (2016)
Punctuation Prediction for Unsegmented Transcript Based on Word Vector (LREC 2016)
The authors propose a CNN or DNN model that uses only pre-trained word vectors as input. Their model predicts whether punctuation should follow a particular word in a 5-word sequence, and if so, which punctuation mark. The dataset used was TED Talks from the IWSLT corpus.

- Chin Char Juin et al. (2017)
Punctuation Prediction Using a Bidirectional Recurrent Neural Network with POS Tagging (TENCON 2017)
This work introduces a BiRNN model enhanced with attention mechanisms and part-of-speech (POS) tags. It handles 11 punctuation classes and is trained on the WikiText corpus.

These works highlight that punctuation recovery is a nuanced task requiring both syntactic and semantic understanding — and that neural models can effectively learn such patterns when provided with appropriate data and contextual features.
