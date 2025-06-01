import glob
import json
import os

import numpy
import torch
from sklearn.metrics import classification_report

from evaluation.model_wrapper import BiLSTMPunctuatorWrapper
from model.baseline_bilstm import BiLSTMPunctuator
from utils.data_utils import load_samples_from_text_folder
from utils.dataset import PunctuationDataset, collate_fn
from utils.eval_utils import (evaluate_model, plot_classification_report,
                              plot_confusion_matrix)
from utils.preprocess_data import get_punctuation_signs_for_prediction

#from functools import partial

#from torch.utils.data import DataLoader



def load_vocab(config):
    with open(config["VOCAB_PATH"], "r") as f:
        return json.load(f)


def get_eval_paths(test_dir):
    filepaths = glob.glob(os.path.join(test_dir, "*.txt"))
    if not filepaths:
        raise FileNotFoundError(f"No .txt files found in {test_dir}")
    return filepaths, f"report/{test_dir}"

def evaluate_m(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab(config)
    class_labels = get_punctuation_signs_for_prediction()
    # Load label2id mapping
    with open(config["LABEL2ID_PATH"], "r") as f:
        label2id = json.load(f)
    all_class_indices = sorted(label2id.values())
    #labels = [label2id[label] for label in class_labels]
    
    conf_matrix = numpy.zeros((len(class_labels), len(class_labels)), dtype="int32")

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
    #loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], collate_fn=partial(collate_fn, vocab=vocab, config=config))

    test_inputs, test_labels = load_samples_from_text_folder(config["TEST_DIR"])
    #test_dataset = PunctuationDataset(test_inputs, test_labels, vocab, class_labels, config["MAX_SEQ_LEN"])
        
    myModelWrpped = BiLSTMPunctuatorWrapper(model, device, vocab, class_labels, config["MAX_SEQ_LEN"])
    calculated_f1, all_golds, all_preds, conf_matrix  = evaluate_model(myModelWrpped, test_inputs, test_labels)

    # Get classification report (handles missing labels safely)
    report = classification_report(
        all_golds,
        all_preds,
        labels=all_class_indices,
        target_names=class_labels,
        output_dict=True,
        zero_division=0
    )
    print(f"Our calculated F1 for split '{config['TEST_DIR']}': {calculated_f1:.4f}")
    plot_classification_report(
        report,
        label2id,
        save_dir=config["mode"],        
    )

    plot_confusion_matrix(all_golds, all_preds, class_labels, save_dir=config["mode"])


