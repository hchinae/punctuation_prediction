import json
import os

import torch

from model.baseline_bilstm import BiLSTMPunctuator
from utils.preprocess_data import (get_punctuation_marker,
                                   get_punctuation_signs_for_prediction)

# You can adjust these paths based on your config or hardcode for submission
MODEL_PATH = "checkpoints/best_model.pt"
VOCAB_PATH = "data/vocab.json"
PADDING_IDX = 0
UNK_IDX = 1
MAX_SEQ_LEN = 256

class MyModel:
    def __init__(self):
        with open(VOCAB_PATH, "r") as f:
            self.vocab = json.load(f)

        self.model = BiLSTMPunctuator(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=256,
            output_dim=len(get_punctuation_signs_for_prediction()),
            pad_idx=PADDING_IDX
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        self.punct_signs = get_punctuation_signs_for_prediction()
        self.marker = get_punctuation_marker()

    def predict(self, text):
        input_ids = []
        marker_positions = []

        for i, token in enumerate(text[:MAX_SEQ_LEN]):
            if token == self.marker:
                marker_positions.append(i)
                input_ids.append(self.vocab.get(token, UNK_IDX))
            else:
                input_ids.append(self.vocab.get(token, UNK_IDX))

        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            logits = logits[0]

        predictions = []
        for idx in marker_positions:
            pred_id = torch.argmax(logits[idx]).item()
            predictions.append(self.punct_signs[pred_id])

        return predictions
