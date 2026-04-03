from __future__ import annotations

from typing import Sequence

import torch


def label_even_parity_cpu(bits: Sequence[int]) -> int:
    return int(sum(bits) % 2 == 0)


def label_even_parity_gpu(bits: torch.Tensor) -> torch.Tensor:
    return (torch.sum(bits, dim=1) % 2 == 0).long()
