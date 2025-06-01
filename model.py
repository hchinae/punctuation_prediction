import torch.nn as nn


class PunctuationPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)         # (B, L, D)
        x, _ = self.lstm(x)                   # (B, L, 2H)
        x = self.dropout(x)
        logits = self.classifier(x)           # (B, L, C)
        return logits
