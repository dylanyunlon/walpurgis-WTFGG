import time
import os
import json


class TrainLogger():
    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.dataset = dataset
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        self.log_dir = os.path.join(path, cur_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.epoch_logs = []

    def __print(self, dic, note=None, ban=[]):
        print("=============== " + note + " =================")
        for key, value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)

    def log_epoch(self, epoch, metrics):
        record = {"epoch": epoch, **metrics}
        self.epoch_logs.append(record)
        log_file = os.path.join(
            self.log_dir, f"{self.model_name}_{self.dataset}.json")
        with open(log_file, 'w') as f:
            json.dump(self.epoch_logs, f, indent=2)
