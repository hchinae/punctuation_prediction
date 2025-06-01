import torch


class EarlyStopping:
    def __init__(self, patience=5, mode="max", delta=0.0, save_path=None, verbose=True):
        """
        Args:
            patience (int): How long to wait after last improvement.
            mode (str): "min" or "max" — whether lower or higher is better.
            delta (float): Minimum change to count as improvement.
            save_path (str): Where to save best model.
            verbose (bool): Print messages.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.is_better = self._get_comparator(mode)

    def _get_comparator(self, mode):
        if mode == "min":
            return lambda current, best: current < best - self.delta
        elif mode == "max":
            return lambda current, best: current > best + self.delta
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def step(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(model)
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self._save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                self.early_stop = True

    def _save_checkpoint(self, model):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Best model saved to {self.save_path}")
