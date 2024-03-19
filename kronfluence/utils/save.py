import json
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors import safe_open

FACTOR_SAVE_PREFIX = "factors_"
SCORE_SAVE_PREFIX = "scores_"

FACTOR_ARGUMENTS_NAME = "factor"
SCORE_ARGUMENTS_NAME = "score"


def load_file(path: Path) -> Dict[str, torch.Tensor]:
    """Loads a dictionary of tensors from the path."""
    load_dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            load_dict[key] = f.get_tensor(name=key)
    return load_dict


def save_json(obj: Any, path: Path) -> None:
    """Saves the object to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def load_json(path: Path) -> Dict[str, Any]:
    """Loads an object from the JSON file."""
    with open(path, "rb") as f:
        obj = json.load(f)
    return obj


@torch.no_grad()
def verify_models_equivalence(state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> bool:
    """Checks if two models are equivalent given their `state_dict`."""
    if len(state_dict1) != len(state_dict2):
        return False

    if state_dict1.keys() != state_dict2.keys():
        return False

    for name in state_dict1:
        tensor1 = state_dict1[name].to(dtype=torch.float32).cpu()
        tensor2 = state_dict2[name].to(dtype=torch.float32).cpu()
        if not torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5):
            return False

    return True
