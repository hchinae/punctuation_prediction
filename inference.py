import json

import torch
import yaml

from model import PunctuationPredictor
from utils.preprocess_data import get_input_label_from_text, pad_punctuation


def load_inference_components(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(config["VOCAB_PATH"]) as f:
        vocab = json.load(f)

    with open(config["ID2LABEL_PATH"]) as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    model = PunctuationPredictor(
        vocab_size=len(vocab),
        embed_dim=config["EMBEDDING_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        num_classes=config["NUM_CLASSES"],
        dropout=config["DROPOUT"],
        padding_idx=config["PADDING_IDX"]
    )
    model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"], map_location="cpu"))
    model.eval()

    return model, vocab, id2label, config


def predict_punctuation(text, model, vocab, id2label, config):
    from utils.preprocess_data import get_punctuation_marker

    # Preprocess raw text
    padded = pad_punctuation(text)
    input_tokens, _ = get_input_label_from_text(padded)

    input_ids = [vocab.get(tok, config["UNK_IDX"]) for tok in input_tokens]
    punct_positions = [i for i, tok in enumerate(input_tokens) if tok == get_punctuation_marker()]

    # Pad
    input_ids = input_ids[:config["MAX_SEQ_LEN"]]
    input_ids += [config["PADDING_IDX"]] * (config["MAX_SEQ_LEN"] - len(input_ids))
    input_tensor = torch.tensor(input_ids).unsqueeze(0)  # (1, L)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1).squeeze(0)

    predicted_labels = [id2label[predictions[pos].item()] for pos in punct_positions if pos < config["MAX_SEQ_LEN"]]
    return predicted_labels

if __name__ == "__main__":
    text = "hello how are you i am fine and you"
    model, vocab, id2label, config = load_inference_components("config.yaml")
    preds = predict_punctuation(text, model, vocab, id2label, config)
    print("Predicted punctuations:", preds)