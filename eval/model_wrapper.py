import json

import torch
import yaml

from model.baseline_bilstm import BiLSTMPunctuator
from utils.preprocess_data import (get_punctuation_marker,
                                   get_punctuation_signs_for_prediction)

with open("../config.yaml", "r") as f: 
    CONFIG = yaml.safe_load(f)

class MyModel:
    def __init__(self):
        self.CONFIG = CONFIG
        with open(CONFIG["VOCAB_PATH"], "r") as f:
            self.vocab = json.load(f)

        self.model = BiLSTMPunctuator(
            vocab_size=len(self.vocab),
            embedding_dim=CONFIG["EMBEDDING_DIM"],
            hidden_dim=CONFIG["HIDDEN_DIM"],
            output_dim=len(get_punctuation_signs_for_prediction()),
            pad_idx=CONFIG["PADDING_IDX"]
        )
        self.model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location="cpu"))
        self.model.eval()

        self.punct_signs = get_punctuation_signs_for_prediction()
        self.marker = get_punctuation_marker()

    def predict(self, text):
        input_ids = []
        marker_positions = []

        for i, token in enumerate(text[:self.CONFIG["MAX_SEQ_LEN"]]):
            if token == self.marker:
                marker_positions.append(i)
                input_ids.append(self.vocab.get(token, self.CONFIG["UNK_IDX"]))
            else:
                input_ids.append(self.vocab.get(token, self.CONFIG["UNK_IDX"]))

        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            logits = logits[0]

        predictions = []
        for idx in marker_positions:
            pred_id = torch.argmax(logits[idx]).item()
            predictions.append(self.punct_signs[pred_id])

        return predictions
