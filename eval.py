import glob
import json
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from model.baseline_bilstm import BiLSTMPunctuator
from utils.dataset import PunctuationDataset, collate_fn
from utils.eval_utils import evaluate
from utils.preprocess_data import get_punctuation_signs_for_prediction


def load_vocab(config):
    with open(config["VOCAB_PATH"], "r") as f:
        return json.load(f)


def get_eval_paths(test_dir):
    filepaths = glob.glob(os.path.join(test_dir, "*.txt"))
    if not filepaths:
        raise FileNotFoundError(f"No .txt files found in {test_dir}")
    return filepaths, f"report/{test_dir}"

def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab(config)
    class_labels = get_punctuation_signs_for_prediction()

    model = BiLSTMPunctuator(
        vocab_size=len(vocab),
        embedding_dim=config["EMBEDDING_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        output_dim=len(class_labels),
        pad_idx=config["PADDING_IDX"]
    ).to(device)
    model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"], map_location=device))
    model.eval()

    #filepaths, plot_dir = get_eval_paths(config["TEST_DIR"])
    dataset = PunctuationDataset(config)
    loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], collate_fn=partial(collate_fn, vocab=vocab, config=config))

    f1 = evaluate(model, loader, device, class_labels=class_labels, plot=True, plot_dir=f"report/{config['mode']}")
    print(f"Macro F1 for split '{config['TEST_DIR']}': {f1:.4f}")

