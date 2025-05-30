# Model params
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.3
NUM_CLASSES = 10  # will update after loading punctuation list

# Training
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0

# Data
MAX_SEQ_LEN = 128
MIN_FREQ = 2  # min frequency for vocab
PADDING_IDX = 0
UNK_IDX = 1

# Device
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Files
TRAIN_DIR = "data/train/"
VOCAB_PATH = "data/vocab.json"
TRAIN_JSON_PATH = "data/processed/train.json"
VAL_JSON_PATH = "data/processed/val.json"
MODEL_SAVE_PATH = "checkpoints/best_model.pt"
VAL_RATIO = 0.1

# Others
SEED = 42