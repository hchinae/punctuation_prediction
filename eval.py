import json

import torch
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from model import PunctuationPredictor
from utils.dataset import PunctuationDataset, collate_fn
from utils.eval_utils import plot_classification_report, plot_confusion_matrix
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
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)  

    print(f"\nMacro F1 score: {f1:.4f}")
    print(f"Weighted F1 score: {weighted_f1:.4f}")

    print("\nClassification Report:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[id2label[str(i)] for i in sorted(id2label, key=int)],
        digits=4,
        zero_division=0,
        output_dict=True 
    )
   

    return f1, report


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab and label maps
    with open(config["VOCAB_PATH"]) as f:
        vocab = json.load(f)
    with open(config["ID2LABEL_PATH"]) as f:
        id2label = json.load(f)

    # Preprocess evaluation data
    inputs, targets  = [], []    
    from glob import glob
    eval_files = glob(config["VAL_DIR"] + "*.txt")
    for file in eval_files:
        inp, tgt = preprocess_file(file)
        inputs.extend(inp)
        targets.extend(tgt)

    #load label2id mapping
    with open(config["LABEL2ID_PATH"]) as f:
        label2id = json.load(f)
    # Create dataset + dataloader
    dataset = PunctuationDataset(inputs, targets, vocab, label2id, config)
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

    f1, report = evaluate_model(model, dataloader, id2label, device)
    #load label2id mapping
    with open(config["LABEL2ID_PATH"]) as f:
        label2id = json.load(f)
    class_labels = [id2label[str(i)] for i in sorted(id2label, key=int)]
    plot_classification_report(report, label2id, "report/val/")
    plot_confusion_matrix(report, class_labels, "report/val/")

if __name__ == "__main__":
    main()
