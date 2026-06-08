import time
import os
import json


class TrainLogger():
    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        self.log_dir = path + cur_time
        os.makedirs(self.log_dir, exist_ok=True)
        self._metrics_file = os.path.join(self.log_dir, "metrics.csv")
        self._events_file = os.path.join(self.log_dir, "events.jsonl")
        with open(self._metrics_file, "w") as f:
            f.write("epoch,train_mae,train_mape,train_rmse,val_mae,val_mape,val_rmse,lr\n")

    def print_model_args(self, model_args, ban=[]):
        print("=" * 50 + " model args " + "=" * 50)
        for key, value in model_args.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))

    def print_optim_args(self, optim_args, ban=[]):
        print("=" * 50 + " optim args " + "=" * 50)
        for key, value in optim_args.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))

    def log_epoch(self, epoch, metrics):
        with open(self._metrics_file, "a") as f:
            f.write("{},{},{},{},{},{},{},{}\n".format(
                epoch,
                metrics.get('train_mae', 0), metrics.get('train_mape', 0),
                metrics.get('train_rmse', 0), metrics.get('val_mae', 0),
                metrics.get('val_mape', 0), metrics.get('val_rmse', 0),
                metrics.get('lr', 0)))
        with open(self._events_file, "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
