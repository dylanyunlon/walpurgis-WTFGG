"""log — Zenith变体"""
import time
import os
import shutil
import json


class TrainLogger():
    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime())
        self.log_dir = os.path.join(path, cur_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.events_file = os.path.join(
            self.log_dir, 'events.jsonl')
        self.metrics_file = os.path.join(
            self.log_dir, 'metrics.csv')
        with open(self.metrics_file, 'w') as f:
            f.write("epoch,train_mae,train_mape,train_rmse,"
                    "val_mae,val_mape,val_rmse,lr\n")

    def _write_event(self, event_type, data):
        record = {
            "ts": time.time(),
            "type": event_type,
            **data
        }
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(record) + "\n")

    def __print(self, dic, note=None, ban=[]):
        print("=" * 20 + f" {note} " + "=" * 20)
        for key, value in dic.items():
            if key in ban:
                continue
            print(f'|{key:>20s}:|{str(value):>20s}|')
        print("-" * 44)

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)
        self._write_event("model_args", {
            k: str(v) for k, v in model_args.items()
            if k not in ban})

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)
        self._write_event("optim_args", {
            k: str(v) for k, v in optim_args.items()
            if k not in ban})

    def log_epoch(self, epoch, metrics):
        self._write_event("epoch", {
            "epoch": epoch, **metrics})
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{metrics.get('train_mae', 0):.4f},"
                    f"{metrics.get('train_mape', 0):.4f},"
                    f"{metrics.get('train_rmse', 0):.4f},"
                    f"{metrics.get('val_mae', 0):.4f},"
                    f"{metrics.get('val_mape', 0):.4f},"
                    f"{metrics.get('val_rmse', 0):.4f},"
                    f"{metrics.get('lr', 0):.6f}\n")
