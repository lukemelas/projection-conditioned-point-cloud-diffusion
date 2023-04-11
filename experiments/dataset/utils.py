from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


def show_item(item: Dict):
    for key in item.keys():
        value = item[key]
        if torch.is_tensor(value) and value.numel() < 5:
            value_str = value
        elif torch.is_tensor(value):
            value_str = value.shape
        elif isinstance(value, str):
            value_str = ('...' + value[-52:]) if len(value) > 50 else value
        elif isinstance(value, dict):
            value_str = str({k: type(v) for k, v in value.items()})
        else:
            value_str = type(value)
        print(f"{key:<30} {value_str}")


def normalize_to_zero_one(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())
    

def default(x, d):
    return d if x is None else x


@dataclass
class DatasetMap:
    train: Optional[Iterable] = None
    val: Optional[Iterable] = None
    test: Optional[Iterable] = None