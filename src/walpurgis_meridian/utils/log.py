"""Meridian utils/log.py — training logger."""
import time
import os
import shutil


def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print('[%0.8fs] %s' % (elapsed, name))
        return result
    return clocked


class TrainLogger():
    def __init__(self, model_name, dataset):
        path = 'log/'
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time = cur_time.replace(" ", "-")
        os.makedirs(path + cur_time, exist_ok=True)
        try:
            shutil.copytree('models', path + cur_time + "/models")
            shutil.copytree('configs', path + cur_time + "/configs")
            shutil.copyfile('main.py', path + cur_time + "/main.py")
        except Exception:
            pass
        try:
            shutil.copyfile(
                'output/' + model_name + "_" + dataset + ".pt",
                path + cur_time + "/" + model_name + "_" + dataset + ".pt")
        except Exception:
            pass

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
