import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file as _safetensors_save_file


def load_file(path: Path) -> Dict[str, torch.Tensor]:
    """Loads a dictionary of tensors from a file using `safetensors`.

    Args:
        path (Path):
            The path to the file containing tensor data.

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary where keys are tensor names and values are the corresponding tensors.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}.")
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as e:
        raise RuntimeError(f"Error loading file {path}: {str(e)}") from e


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Atomically saves tensors to `filename` via `safetensors`.

    Writes to a sibling `<filename>.tmp` and then `os.replace`s into place, so a
    process killed mid-write never leaves a half-written file at `filename`. The
    "exists" checks downstream therefore never observe a torn save.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp_filename = filename.with_name(filename.name + ".tmp")
    try:
        _safetensors_save_file(tensors=tensors, filename=str(tmp_filename), metadata=metadata)
        os.replace(tmp_filename, filename)
    except Exception:
        if tmp_filename.exists():
            try:
                tmp_filename.unlink()
            except OSError:
                pass
        raise


def save_json(obj: Any, path: Path) -> None:
    """Saves an object to a JSON file.

    This function serializes the given object to JSON format and writes it to a file.

    Args:
        obj (Any):
            The object to be saved. Must be JSON-serializable.
        path (Path):
            The path where the JSON file will be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)
    except TypeError as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise TypeError(f"Object is not JSON-serializable: {str(e)}") from e
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise IOError(f"Error saving JSON file {path}: {str(e)}") from e


def load_json(path: Path) -> Dict[str, Any]:
    """Loads an object from a JSON file.

    Args:
        path (Path):
            The path to the JSON file to be loaded.

    Returns:
        Dict[str, Any]:
            The object loaded from the JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_models_equivalence(state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> bool:
    """Check if two models are equivalent given their state dictionaries.

    This function compares two model state dictionaries to determine if they represent
    equivalent models. It checks for equality in the number of parameters, parameter names,
    and parameter values (within a small tolerance).

    Args:
        state_dict1 (Dict[str, torch.Tensor]):
            The state dictionary of the first model.
        state_dict2 (Dict[str, torch.Tensor]):
            The state dictionary of the second model.

    Returns:
        bool:
            `True` if the models are equivalent, `False` otherwise.

    Notes:
        - The function uses a relative tolerance of 1.3e-6 and an absolute tolerance of 1e-5
          when comparing tensor values.
        - Tensors are compared in float32 precision on the CPU to ensure consistency.
    """
    if len(state_dict1) != len(state_dict2):
        return False

    if state_dict1.keys() != state_dict2.keys():
        return False

    for name in state_dict1:
        tensor1 = state_dict1[name].to(dtype=torch.float32).cpu()
        tensor2 = state_dict2[name].to(dtype=torch.float32).cpu()
        if not torch.allclose(tensor1, tensor2, rtol=1.3e-6, atol=1e-5):
            return False

    return True
