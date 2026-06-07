import torch
import os
import json
import yaml
import sys
import numpy as np

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    print(f"[EQX:utils:{tag}] {val}", file=sys.stderr)


class EarlyStopping:
    """upstream: 无
    aurora: 新增patience-based早停, 避免过拟合"""
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            _edbg("early_stop", f"no improve: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def set_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    _edbg("config", json.dumps(cfg, indent=2, default=str))
    return cfg


def data_reshaper(data, device):
    data = torch.Tensor(data).to(device)
    return data


def save_model(model, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(model.state_dict(), path)
    _edbg("save_model", path)


def get_num_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total
