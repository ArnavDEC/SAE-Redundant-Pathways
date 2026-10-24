"""Load agent-supplied tensors in the grader.

Use safetensors; torch.load on untrusted tensors can run arbitrary code.
Returns None on a missing or invalid file (-> score 0)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from safetensors import SafetensorError

from safe_read import read_submission_bytes

if TYPE_CHECKING:
    import numpy as np
    import torch

T = TypeVar("T")


def read_tensors_torch(path: str, max_bytes: int) -> dict[str, torch.Tensor] | None:
    from safetensors.torch import load

    return _read_tensors(path, max_bytes, load)


def read_tensors_numpy(path: str, max_bytes: int) -> dict[str, np.ndarray] | None:
    from safetensors.numpy import load

    return _read_tensors(path, max_bytes, load)


def _read_tensors(path: str, max_bytes: int, load: Callable[[bytes], T]) -> T | None:
    data = read_submission_bytes(path, max_bytes)
    if data is None:
        return None
    try:
        return load(data)
    # KeyError/TypeError: valid file, dtype this backend can't represent (e.g. BF16 under numpy)
    except (SafetensorError, KeyError, TypeError) as exc:
        print(f"safe_read_tensors: {path} is not loadable safetensors: {exc}")
        return None
