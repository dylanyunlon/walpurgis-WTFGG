import os as _os
import json as _json

# upstream 无 datasets/__init__.py; 新增.
# 提供数据集自动发现 + 元信息打印.

_DATASET_DIR = _os.path.join(_os.path.dirname(__file__), "raw_data")

_KNOWN_DATASETS = {
    "METR-LA":   {"type": "speed", "nodes": 207, "interval": "5min"},
    "PEMS-BAY":  {"type": "speed", "nodes": 325, "interval": "5min"},
    "PEMS04":    {"type": "flow",  "nodes": 307, "interval": "5min"},
    "PEMS08":    {"type": "flow",  "nodes": 170, "interval": "5min"},
}


def list_datasets():
    """列出 raw_data 下所有可用数据集目录, 标注已知/未知."""
    found = []
    if _os.path.isdir(_DATASET_DIR):
        for d in sorted(_os.listdir(_DATASET_DIR)):
            full = _os.path.join(_DATASET_DIR, d)
            if _os.path.isdir(full) and not d.startswith("_"):
                meta = _KNOWN_DATASETS.get(d, {})
                found.append({"name": d, "path": full, **meta})
    return found


def print_dataset_info():
    """打印所有数据集的元信息表."""
    ds_list = list_datasets()
    print(f"[walpurgis:datasets] {len(ds_list)} datasets found under {_DATASET_DIR}")
    for ds in ds_list:
        extras = ", ".join(f"{k}={v}" for k, v in ds.items()
                           if k not in ("name", "path"))
        print(f"  {ds['name']:12s}  {extras or '(unknown)'}")


__all__ = ["list_datasets", "print_dataset_info"]
