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

# Files
TRAIN_DIR = "data/train/"
VOCAB_PATH = "data/vocab.json"
MODEL_SAVE_PATH = "checkpoints/best_model.pt"

# Others
SEED = 42