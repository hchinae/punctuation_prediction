import json
import os

import torch
import yaml
from torch.nn.functional import softmax

from model import \
    PunctuationPredictor  # assuming your model is defined in model.py
from utils.preprocess_data import (get_punctuation_marker,
                                   get_punctuation_signs_for_prediction)


class MyModel():
    def __init__(self):
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load vocab and label mappings
        with open(self.config["VOCAB_PATH"]) as f:
            self.vocab = json.load(f)
        with open(self.config["LABEL2ID_PATH"]) as f:
            self.label2id = json.load(f)
        with open(self.config["ID2LABEL_PATH"]) as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.punct_token = get_punctuation_marker()
        self.valid_puncts = get_punctuation_signs_for_prediction()

        self.model = PunctuationPredictor(
            vocab_size=len(self.vocab),
            embed_dim=self.config["EMBEDDING_DIM"],
            hidden_dim=self.config["HIDDEN_DIM"],
            num_classes=self.config["NUM_CLASSES"],
            dropout=self.config["DROPOUT"],
            padding_idx=self.config["PADDING_IDX"]
        ).to(self.device)

        # Load trained weights
        self.model.load_state_dict(torch.load(self.config["MODEL_SAVE_PATH"], map_location=self.device))
        self.model.eval()

    def predict(self, text):
        input_ids = []
        punct_positions = []

        for i, token in enumerate(text):
            if token == self.punct_token:
                input_ids.append(self.vocab.get(token, self.config["UNK_IDX"]))
                punct_positions.append(i)
            else:
                input_ids.append(self.vocab.get(token, self.config["UNK_IDX"]))

        # Truncate or pad to max seq length
        input_ids = input_ids[:self.config["MAX_SEQ_LEN"]]
        padding_length = self.config["MAX_SEQ_LEN"] - len(input_ids)
        input_ids += [self.config["PADDING_IDX"]] * padding_length

        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)  # shape: (1, L, C)
            probs = softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1).squeeze(0)  # (L,)

        # Return predictions only at original punctuation positions
        output = []
        for pos in punct_positions:
            if pos < self.config["MAX_SEQ_LEN"]:  # Ensure within bounds
                label_id = predictions[pos].item()
                output.append(self.id2label[label_id])
        return output
