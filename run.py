import yaml

from train import train_model  # From our previous step
from utils.dataset import \
    PunctuationDataset  # You'll create this file if not done
from utils.dataset import load_data  # From our previous step
from utils.seed import set_seed  # Optional: for reproducibility


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    set_seed(config["SEED"])  # Optional for reproducibility

    train_inputs, train_targets, val_inputs, val_targets, vocab, label2id = load_data(config)

    print(f"Loaded {len(train_inputs)} training and {len(val_inputs)} validation sequences.")

    train_dataset = PunctuationDataset(train_inputs, train_targets, vocab, label2id, config)
    val_dataset = PunctuationDataset(val_inputs, val_targets, vocab, label2id, config)

    print("Starting training...")
    train_model(config, train_dataset, val_dataset, label2id)

    print(f"Training complete. Model saved to {config['MODEL_SAVE_PATH']}")

if __name__ == "__main__":
    main()
