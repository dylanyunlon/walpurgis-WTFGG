"""Rift log — JSONL + CSV dual dump pattern for training metrics"""
import time
import os
import json
import csv


class TrainLogger():
    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.dataset = dataset
        self.log_dir = os.path.join('log', f'rift_{model_name}_{dataset}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.log_dir, exist_ok=True)
        # JSONL log
        self._jsonl_path = os.path.join(self.log_dir, 'metrics.jsonl')
        # CSV log
        self._csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self._csv_initialized = False

    def _init_csv(self, fieldnames):
        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        self._csv_initialized = True
        self._csv_fields = fieldnames

    def log_metrics(self, epoch, **kwargs):
        record = {'epoch': epoch, 'ts': time.time()}
        record.update(kwargs)
        # JSONL
        with open(self._jsonl_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
        # CSV
        if not self._csv_initialized:
            self._init_csv(list(record.keys()))
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction='ignore')
            writer.writerow(record)

    def print_model_args(self, model_args, ban=[]):
        print("=============== model args =================")
        for key, value in model_args.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")

    def print_optim_args(self, optim_args, ban=[]):
        print("=============== optim args =================")
        for key, value in optim_args.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")
