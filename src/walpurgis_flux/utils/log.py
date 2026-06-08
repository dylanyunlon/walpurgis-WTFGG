"""Flux log: JSONL + CSV双写日志 + 流式推理延迟追踪.
与upstream(基础print)和vortex(JSONL+rolling stats)不同,
Flux额外追踪每个epoch的流式推理统计:
窗口命中率, 渐进式解码各阶段精度, 推理延迟分布."""
import time
import os
import json
import csv
import sys
import numpy as np


class TrainLogger:
    def __init__(self, model_name, dataset):
        ts = time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime())
        self.log_dir = os.path.join('log', ts)
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(
            self.log_dir, 'events.jsonl')
        self.csv_path = os.path.join(
            self.log_dir, 'metrics.csv')
        self._csv_init = False
        self._rolling = []
        # Flux: 流式推理统计日志
        self.stream_log_path = os.path.join(
            self.log_dir, 'stream_stats.jsonl')
        for src in [f'output/{model_name}_{dataset}.pt',
                    f'output/{model_name}_{dataset}_resume.pt']:
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, os.path.join(
                    self.log_dir, os.path.basename(src)))
        print(f"[FX:log] dir={self.log_dir}",
              file=sys.stderr)

    def log_metrics(self, epoch, **metrics):
        record = {
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics}
        # Rolling statistics
        self._rolling.append(metrics)
        if len(self._rolling) > 10:
            self._rolling = self._rolling[-10:]
        if len(self._rolling) >= 3:
            for key in metrics:
                if isinstance(metrics[key], (int, float)):
                    vals = [
                        r[key] for r in self._rolling
                        if key in r and isinstance(
                            r[key], (int, float))]
                    if vals:
                        record[f"{key}_rolling_mean"] = \
                            float(np.mean(vals))
                        record[f"{key}_rolling_std"] = \
                            float(np.std(vals))
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        if not self._csv_init:
            with open(self.csv_path, 'w',
                      newline='') as f:
                w = csv.DictWriter(
                    f, fieldnames=record.keys())
                w.writeheader()
                w.writerow(record)
            self._csv_init = True
        else:
            with open(self.csv_path, 'a',
                      newline='') as f:
                csv.DictWriter(
                    f,
                    fieldnames=record.keys()
                ).writerow(record)

    def log_stream_stats(self, epoch, **stream_stats):
        """Flux: 记录流式推理统计"""
        record = {
            "epoch": epoch,
            "timestamp": time.time(),
            **stream_stats}
        with open(self.stream_log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def _print(self, dic, note=None, ban=[]):
        print(f"=============== {note} =================")
        for k, v in dic.items():
            if k in ban:
                continue
            print(f'|{k:>20s}:|{str(v):>20s}|')
        print("--------------------------------------------")

    def print_model_args(self, model_args, ban=[]):
        self._print(
            model_args, note='model args', ban=ban)

    def print_optim_args(self, optim_args, ban=[]):
        self._print(
            optim_args, note='optim args', ban=ban)
