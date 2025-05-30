import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence

from config import (DEVICE, MAX_SEQ_LEN, MODEL_SAVE_PATH, PADDING_IDX, UNK_IDX,
                    VOCAB_PATH)
from model.baseline_bilstm import BiLSTMPunctuator
from preprocessing.preprocess_data import (
    get_punctuation_marker, get_punctuation_signs_for_prediction)


class LSTMWrapper:
    def __init__(self):
        # Load vocab
        with open(VOCAB_PATH, "r") as f:
            self.vocab = json.load(f)
        self.word2idx = self.vocab

        # Setup mappings
        self.idx2punct = {i: p for i, p in enumerate(get_punctuation_signs_for_prediction())}
        self.punct2idx = {p: i for i, p in self.idx2punct.items()}
        self.marker = get_punctuation_marker()

        # Load model
        self.model = BiLSTMPunctuator(
            vocab_size=len(self.vocab),
            embedding_dim=128,         # should match config
            hidden_dim=256,
            output_dim=len(self.idx2punct),
            pad_idx=PADDING_IDX
        )
        self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text):
        """
        Given a list of tokens (including <punctuation>), return predicted punctuation list.
        """
        input_ids = []
        marker_positions = []

        for i, token in enumerate(text[:MAX_SEQ_LEN]):
            if token == self.marker:
                marker_positions.append(i)
                input_ids.append(self.word2idx.get(self.marker, UNK_IDX))
            else:
                input_ids.append(self.word2idx.get(token, UNK_IDX))

        # Pad to MAX_SEQ_LEN
        pad_len = MAX_SEQ_LEN - len(input_ids)
        input_ids += [PADDING_IDX] * pad_len

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits, _ = self.model(input_tensor)  # [1, L, C]
            predictions = torch.argmax(logits, dim=-1).squeeze(0)  # [L]

        # Extract predictions at <punctuation> positions only
        pred_punct = [self.idx2punct[predictions[pos].item()] for pos in marker_positions]
        return pred_punct
