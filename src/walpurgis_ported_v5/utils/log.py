import time
import os
import shutil

# Delta vs upstream:
#   1. clock decorator: adds peak memory tracking via torch.cuda if available
#   2. TrainLogger: computes cumulative file hash for snapshot integrity check


def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        # ── delta 1: memory footprint alongside timing ──
        mem_str = ""
        try:
            import torch
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                mem_str = f" | peak_gpu={mem_mb:.1f}MB"
        except Exception:
            pass
        print(f'[{elapsed:0.8f}s{mem_str}] {name}')
        return result
    return clocked


class TrainLogger():
    def __init__(self, model_name, dataset):
        path        = 'log/'
        cur_time    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time    = cur_time.replace(" ", "-")
        os.makedirs(path + cur_time)
        shutil.copytree('models',  path + cur_time + "/models")
        shutil.copytree('configs', path + cur_time + "/configs")
        shutil.copyfile('main.py', path + cur_time + "/main.py")
        # ── delta 2: snapshot hash for reproducibility audit ──
        self._snapshot_dir = path + cur_time
        self._write_manifest()
        try:
            shutil.copyfile(
                'output/' + model_name + "_" + dataset + ".pt",
                path + cur_time + "/" + model_name + "_" + dataset + ".pt")
            shutil.copyfile(
                'output/' + model_name + "_" + dataset + "_resume" + ".pt",
                path + cur_time + "/" + model_name + "_" + dataset + "_resume.pt")
        except:
            pass

    def _write_manifest(self):
        """Write file listing with sizes for integrity checking."""
        import hashlib
        manifest = []
        for root, dirs, files in os.walk(self._snapshot_dir):
            for f in sorted(files):
                fp = os.path.join(root, f)
                sz = os.path.getsize(fp)
                # fast hash: first 4KB
                h = hashlib.md5()
                with open(fp, 'rb') as fh:
                    h.update(fh.read(4096))
                manifest.append(f"{fp}\t{sz}\t{h.hexdigest()[:12]}")
        with open(os.path.join(self._snapshot_dir, "MANIFEST.txt"), 'w') as mf:
            mf.write("\n".join(manifest))

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
