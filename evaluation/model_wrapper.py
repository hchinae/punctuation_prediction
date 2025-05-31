import json

import torch
import yaml

from model.baseline_bilstm import BiLSTMPunctuator
from utils.preprocess_data import (get_punctuation_marker,
                                   get_punctuation_signs_for_prediction)

with open("config.yaml", "r") as f: 
    CONFIG = yaml.safe_load(f)

def prepare_input(text, vocab, CONFIG):
    marker = get_punctuation_marker()
    punct2idx = {p: i for i, p in enumerate(get_punctuation_signs_for_prediction())}
    
    input_ids = []
    marker_positions = []

    for i, tok in enumerate(text[:CONFIG["MAX_SEQ_LEN"]]):
        input_ids.append(vocab.get(tok, CONFIG["UNK_IDX"]))
        if tok == marker:
            marker_positions.append(i)

    pad_len = CONFIG["MAX_SEQ_LEN"] - len(input_ids)
    input_ids += [CONFIG["PADDING_IDX"]] * pad_len

    return torch.tensor([input_ids]), marker_positions
    
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
        input_tensor, marker_positions = prepare_input(text, self.vocab, self.CONFIG)

        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            logits = logits[0]  # [seq_len, num_classes]

        predictions = []
        for idx in marker_positions:
            pred_id = torch.argmax(logits[idx]).item()
            predictions.append(self.punct_signs[pred_id])

        return predictions

    # def predict(self, text):
    #     input_ids = []
    #     marker_positions = []

    #     for i, token in enumerate(text[:self.CONFIG["MAX_SEQ_LEN"]]):
    #         if token == self.marker:
    #             marker_positions.append(i)
    #             input_ids.append(self.vocab.get(token, self.CONFIG["UNK_IDX"]))
    #         else:
    #             input_ids.append(self.vocab.get(token, self.CONFIG["UNK_IDX"]))

    #     input_tensor = torch.tensor([input_ids], dtype=torch.long)
    #     with torch.no_grad():
    #         logits, _ = self.model(input_tensor)
    #         logits = logits[0]

    #     predictions = []
    #     for idx in marker_positions:
    #         pred_id = torch.argmax(logits[idx]).item()
    #         predictions.append(self.punct_signs[pred_id])

    #     return predictions
