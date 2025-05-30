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
        input_ids = []
        marker_positions = []

        # Tokenize + collect marker positions
        for token in text:
            if token == self.marker:
                marker_positions.append(len(input_ids))
                input_ids.append(self.word2idx.get(self.marker, UNK_IDX))
            else:
                input_ids.append(self.word2idx.get(token, UNK_IDX))

        predictions = []
        marker_ptr = 0
        total_tokens = len(input_ids)

        # Process in chunks of MAX_SEQ_LEN
        for start in range(0, total_tokens, MAX_SEQ_LEN):
            end = start + MAX_SEQ_LEN
            chunk = input_ids[start:end]

            # Map markers within this chunk
            chunk_markers = []
            for i in range(len(chunk)):
                if (start + i) in marker_positions:
                    chunk_markers.append(i)

            if not chunk_markers:
                continue  # No markers in this chunk, skip

            # Pad if necessary
            pad_len = MAX_SEQ_LEN - len(chunk)
            chunk += [PADDING_IDX] * pad_len

            input_tensor = torch.tensor([chunk], dtype=torch.long).to(DEVICE)

            with torch.no_grad():
                logits, _ = self.model(input_tensor)
                pred = torch.argmax(logits, dim=-1).squeeze(0)

            for pos in chunk_markers:
                predictions.append(self.idx2punct[pred[pos].item()])

        assert len(predictions) == text.count(self.marker), (
            f"Expected {text.count(self.marker)} predictions, got {len(predictions)}"
        )
        return predictions
