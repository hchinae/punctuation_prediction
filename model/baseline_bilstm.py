import torch.nn as nn


class BiLSTMPunctuator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, mask=None):
        """
        input_ids: [B, L]
        mask: optional, not used in forward pass (used in loss computation)

        Returns:
            logits: [B, L, output_dim]
            mask: passed through for convenience
        """
        x = self.embedding(input_ids)        # [B, L, E]
        x, _ = self.bilstm(x)                # [B, L, 2H]
        x = self.dropout(x)
        logits = self.classifier(x)          # [B, L, C]
        return logits, mask
