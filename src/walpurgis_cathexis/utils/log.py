import time, os, shutil, json, csv

class TrainLogger:
    def __init__(self, model_name, dataset):
        self.log_dir = os.path.join('log', time.strftime("%Y-%m-%d-%H:%M:%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self._events = os.path.join(self.log_dir, 'events.jsonl')
        self._metrics = os.path.join(self.log_dir, 'metrics.csv')
        self._csv_header_written = False

    def _print(self, dic, note=None, ban=[]):
        print(f"=============== {note} =================")
        for k, v in dic.items():
            if k in ban: continue
            print(f"|{k:>20s}:|{str(v):>20s}|")
        print("-" * 44)

    def print_model_args(self, model_args, ban=[]): self._print(model_args, note='model args', ban=ban)
    def print_optim_args(self, optim_args, ban=[]): self._print(optim_args, note='optim args', ban=ban)

    def log_epoch(self, epoch, metrics_dict):
        with open(self._events, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'ts': time.time(), **metrics_dict}) + '\n')
        with open(self._metrics, 'a') as f:
            writer = csv.writer(f)
            if not self._csv_header_written:
                writer.writerow(['epoch'] + list(metrics_dict.keys()))
                self._csv_header_written = True
            writer.writerow([epoch] + [metrics_dict[k] for k in metrics_dict])
