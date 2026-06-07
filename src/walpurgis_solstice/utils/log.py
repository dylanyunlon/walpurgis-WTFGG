import os
import sys
import time
import json
import numpy as np

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    print(f"[SOL:log:{tag}] {val}", file=sys.stderr)


class TrainLogger:
    def __init__(self, model_name, dataset_name, log_dir='logs'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.start_time = time.time()
        _adbg("init", f"model={model_name} dataset={dataset_name}")

    def log_epoch(self, epoch, train_loss, val_loss, val_mape=0, val_rmse=0, lr=0):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        elapsed = time.time() - self.start_time
        improved = ''
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            improved = ' *BEST*'
        msg = (f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
               f"MAPE: {val_mape:.4f} RMSE: {val_rmse:.4f} | "
               f"LR: {lr:.6f} | Time: {elapsed:.0f}s{improved}")
        print(msg)
        _adbg("epoch", msg)

    def save_log(self):
        log_data = {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': self.best_epoch,
            'total_time': time.time() - self.start_time
        }
        path = os.path.join(self.log_dir, f'{self.model_name}_{self.dataset_name}.json')
        with open(path, 'w') as f:
            json.dump(log_data, f, indent=2)
        _adbg("saved", path)

    def summary(self):
        print(f"\n{'='*60}")
        print(f"Training Summary: {self.model_name} on {self.dataset_name}")
        print(f"Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print(f"Total time: {time.time()-self.start_time:.0f}s")
        print(f"{'='*60}\n")
