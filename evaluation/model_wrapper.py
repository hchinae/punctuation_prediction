import json
import os

import torch
import yaml

from model.baseline_bilstm import BiLSTMPunctuator
from utils.preprocess_data import (get_punctuation_marker,
                                   get_punctuation_signs_for_prediction)

# Load config once globally
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


class MyModel:
    def __init__(self):
        self.config = config

        with open(config["VOCAB_PATH"], "r") as f:
            self.vocab = json.load(f)

        self.punct_signs = get_punctuation_signs_for_prediction()
        self.marker = get_punctuation_marker()

        self.model = BiLSTMPunctuator(
            vocab_size=len(self.vocab),
            embedding_dim=config["EMBEDDING_DIM"],
            hidden_dim=config["HIDDEN_DIM"],
            output_dim=len(self.punct_signs),
            pad_idx=config["PADDING_IDX"]
        )
        self.model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"], map_location="cpu"))
        self.model.eval()

    def predict(self, text):
        """
        Receives a list of preprocessed tokens including <punctuation> markers.
        Returns a list of predicted punctuation signs at those marker positions.
        """
        marker_positions = [i for i, tok in enumerate(text) if tok == self.marker]

        # Convert input to IDs
        input_ids = [self.vocab.get(tok, self.config["UNK_IDX"]) for tok in text]

        # Pad or truncate to MAX_SEQ_LEN
        max_len = self.config["MAX_SEQ_LEN"]
        if len(input_ids) < max_len:
            input_ids += [self.config["PADDING_IDX"]] * (max_len - len(input_ids))
        else:
            input_ids = input_ids[:max_len]
            marker_positions = [i for i in marker_positions if i < max_len]

        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            logits = logits[0]  # remove batch dimension

        predictions = []
        for idx in marker_positions:
            pred_id = torch.argmax(logits[idx]).item()
            predictions.append(self.punct_signs[pred_id])

        return predictions
