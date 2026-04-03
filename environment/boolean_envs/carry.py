from __future__ import annotations

from typing import Sequence

import torch


def label_carry_cpu(bits: Sequence[int], left_bits: int, right_bits: int) -> int:
    left = int("".join(str(bit) for bit in bits[:left_bits]), 2)
    right = int("".join(str(bit) for bit in bits[left_bits : left_bits + right_bits]), 2)
    return int(left + right >= (1 << max(left_bits, right_bits)))


def label_carry_gpu(bits: torch.Tensor, left_bits: int, right_bits: int) -> torch.Tensor:
    left = _bits_to_int(bits[:, :left_bits])
    right = _bits_to_int(bits[:, left_bits : left_bits + right_bits])
    return (left + right >= (1 << max(left_bits, right_bits))).long()


def _bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    weights = (2 ** torch.arange(bits.shape[1] - 1, -1, -1, device=bits.device)).long()
    return torch.sum(bits.long() * weights, dim=1)
