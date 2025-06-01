import json

import torch
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from model import PunctuationPredictor
from utils.dataset import PunctuationDataset, collate_fn
from utils.preprocess_data import preprocess_file


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_model(model, dataloader, id2label, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(input_ids)
        preds = torch.argmax(logits, dim=-1)

        mask = target_ids != -100
        true = target_ids[mask].cpu().tolist()
        pred = preds[mask].cpu().tolist()

        all_labels.extend(true)
        all_preds.extend(pred)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\nMacro F1 score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        digits=4,
        zero_division=0
    ))

    return f1


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab and label maps
    with open(config["VOCAB_PATH"]) as f:
        vocab = json.load(f)
    with open(config["ID2LABEL_PATH"]) as f:
        id2label = json.load(f)

    # Preprocess evaluation data
    inputs, targets = preprocess_file(config["TEST_DIR"] + "sample_eval.txt")

    # Create dataset + dataloader
    dataset = PunctuationDataset(inputs, targets, vocab, label2id=None, max_len=config["MAX_SEQ_LEN"])
    dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], collate_fn=collate_fn)

    # Load model
    model = PunctuationPredictor(
        vocab_size=len(vocab),
        embed_dim=config["EMBEDDING_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        num_classes=config["NUM_CLASSES"],
        dropout=config["DROPOUT"],
        padding_idx=config["PADDING_IDX"]
    )
    model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"], map_location=device))
    model.to(device)

    evaluate_model(model, dataloader, id2label, device)


if __name__ == "__main__":
    main()
