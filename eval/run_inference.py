import json
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, DEVICE, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN,
                    MODEL_SAVE_PATH, PADDING_IDX, VOCAB_PATH)
from model.baseline_bilstm import BiLSTMPunctuator
from preprocessing.preprocess_data import (
    get_punctuation_marker, get_punctuation_signs_for_prediction)
from train.dataset import PunctuationDataset, collate_fn


def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    return vocab

def run_inference(model, loader, idx2punct, idx2word):
    model.eval()
    marker = get_punctuation_marker()

    with torch.no_grad():
        for input_ids, targets, mask in loader:
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)

            logits, _ = model(input_ids)  # [B, L, C]
            predictions = torch.argmax(logits, dim=-1)  # [B, L]

            for b in range(input_ids.size(0)):
                pred_seq = []
                input_seq = input_ids[b].cpu().tolist()
                pred_punct = predictions[b].cpu().tolist()
                mask_seq = mask[b].cpu().tolist()
                label_seq = targets[b]  # ground truth labels for <punctuation>

                label_idx = 0
                for i, token_id in enumerate(input_seq):
                    token_str = idx2word.get(token_id, "<unk>")
                    if mask_seq[i]:
                        pred = idx2punct[pred_punct[i]]
                        true = idx2punct[label_seq[label_idx]]
                        label_idx += 1
                        pred_seq.append(f"{token_str} [pred: {pred} | true: {true}]")
                    else:
                        pred_seq.append(token_str)
                print(" ".join(pred_seq))
                print("=" * 60)

def main():
    # Load vocab
    vocab = load_vocab()
    idx2word = {v: k for k, v in vocab.items()}
    punct_labels = get_punctuation_signs_for_prediction()
    idx2punct = {i: p for i, p in enumerate(punct_labels)}

    model = BiLSTMPunctuator(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,                  
    output_dim=len(punct_labels),             
    pad_idx=PADDING_IDX                       
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # Choose a validation file
    val_file = "data/train/the-picture-of-dorian-gray.txt"  # replace with your actual file
    dataset = PunctuationDataset([val_file])
    loader = DataLoader(dataset, batch_size=1, collate_fn=partial(collate_fn, vocab=vocab))


    run_inference(model, loader, idx2punct, idx2word)

if __name__ == "__main__":
    main()
