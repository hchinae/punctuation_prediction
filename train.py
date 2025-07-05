import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval import evaluate
from utils.early_stopping import EarlyStopping


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(config, train_dataset, val_dataset, id2label):
    from model import PunctuationPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PunctuationPredictor(
        vocab_size=len(train_dataset.vocab),
        embed_dim=config["EMBEDDING_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        num_classes=config["NUM_CLASSES"],
        dropout=config["DROPOUT"],
        padding_idx=config["PADDING_IDX"]
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"])

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["LEARNING_RATE"]), weight_decay=float(config.get("WEIGHT_DECAY", 0.0)))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.get("LR_REDUCTION_FACTOR", 0.5),
        patience=config.get("SCHEDULER_PATIENCE", 2),
        threshold=config.get("SCHEDULER_THRESHOLD", 1e-4)
        )

    early_stopper = EarlyStopping(
        patience=config.get("EARLY_STOPPING_PATIENCE", 5),
        mode="max",
        delta=0.001,
        save_path=config["MODEL_SAVE_PATH"],
        verbose=True
    )
    #best_val_loss = float("inf")
    best_val_f1 = 0.0
    for epoch in range(config["EPOCHS"]):
        print(f"\nEpoch {epoch + 1}/{config['EPOCHS']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_f1 = evaluate(model, val_loader,  device, id2label, loss_fn)
        train_loss, train_f1 = evaluate(model, train_loader,  device, id2label, loss_fn)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} Train F1: {train_f1} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"Saved best model to {config['MODEL_SAVE_PATH']}")

        scheduler.step(val_f1)
        early_stopper.step(val_f1, model)

        if early_stopper.early_stop:
            break
    return model
