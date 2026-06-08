"""
log — Parallax变体 (M054)
训练日志记录器
"""
import time
import os


class TrainLogger():
    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime())
        self.log_dir = path + cur_time
        os.makedirs(self.log_dir, exist_ok=True)

    def __print(self, dic, note=None, ban=[]):
        print("=" * 16 + " " + note + " " + "=" * 17)
        for key, value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("-" * 44)

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)
