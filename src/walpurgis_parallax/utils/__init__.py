import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch

def check_nan_inf(tensor, raise_ex=True):
    nan = torch.any(torch.isnan(tensor))
    inf = torch.any(torch.isinf(tensor))
    if raise_ex and (nan or inf):
        raise Exception({"nan": nan, "inf": inf})
    return {"nan": nan, "inf": inf}, nan or inf

def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor
