"""Nebula log: JSONL + CSV dual dump."""
import time, os, json, csv, sys

class TrainLogger:
    def __init__(self, model_name, dataset):
        ts = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        self.log_dir = os.path.join('log', ts)
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, 'events.jsonl')
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self._csv_init = False
        for src in [f'output/{model_name}_{dataset}.pt', f'output/{model_name}_{dataset}_resume.pt']:
            if os.path.exists(src):
                import shutil; shutil.copy2(src, os.path.join(self.log_dir, os.path.basename(src)))
        print(f"[NEB:log] dir={self.log_dir}", file=sys.stderr)

    def log_metrics(self, epoch, **metrics):
        record = {"epoch": epoch, "timestamp": time.time(), **metrics}
        with open(self.jsonl_path, 'a') as f: f.write(json.dumps(record) + '\n')
        if not self._csv_init:
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=record.keys()); w.writeheader(); w.writerow(record)
            self._csv_init = True
        else:
            with open(self.csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=record.keys()).writerow(record)

    def _print(self, dic, note=None, ban=[]):
        print(f"=============== {note} =================")
        for k, v in dic.items():
            if k in ban: continue
            print(f'|{k:>20s}:|{str(v):>20s}|')
        print("--------------------------------------------")
    def print_model_args(self, model_args, ban=[]): self._print(model_args, note='model args', ban=ban)
    def print_optim_args(self, optim_args, ban=[]): self._print(optim_args, note='optim args', ban=ban)
