import torch
from torch.utils.data import Dataset


class PunctuationDataset(Dataset):
    def __init__(self, input_sequences, target_labels, vocab, label2id, config):
        self.inputs = input_sequences
        self.targets = target_labels
        self.vocab = vocab
        self.label2id = label2id
        self.config = config

        self.pad_idx = config["PADDING_IDX"]
        self.unk_idx = config["UNK_IDX"]
        self.max_len = config["MAX_SEQ_LEN"]
        self.punct_token = "<punctuation>"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        tokens = self.inputs[idx]
        labels = self.targets[idx]

        input_ids = []
        target_ids = []

        label_idx = 0

        for tok in tokens:
            if tok == self.punct_token:
                input_ids.append(self.vocab.get(tok, self.unk_idx))
                # Replace with label at that <punctuation> position
                target_ids.append(self.label2id.get(labels[label_idx], -100))
                label_idx += 1
            else:
                input_ids.append(self.vocab.get(tok, self.unk_idx))
                target_ids.append(-100)  # We do not compute loss here

        # Pad or truncate input & target
        input_ids = input_ids[:self.max_len]
        target_ids = target_ids[:self.max_len]

        padding_length = self.max_len - len(input_ids)
        input_ids += [self.pad_idx] * padding_length
        target_ids += [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    input_ids = torch.stack(input_ids)
    target_ids = torch.stack(target_ids)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids
    }
