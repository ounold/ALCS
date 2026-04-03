from __future__ import annotations

from typing import Sequence

import torch


def label_multiplexer_cpu(bits: Sequence[int], address_bits: int) -> int:
    address = int("".join(str(bit) for bit in bits[:address_bits]), 2)
    return int(bits[address_bits + address])


def label_multiplexer_gpu(bits: torch.Tensor, address_bits: int) -> torch.Tensor:
    address = _bits_to_int(bits[:, :address_bits])
    return bits[:, address_bits:].gather(1, address.unsqueeze(1)).squeeze(1).long()


def _bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    weights = (2 ** torch.arange(bits.shape[1] - 1, -1, -1, device=bits.device)).long()
    return torch.sum(bits.long() * weights, dim=1)
