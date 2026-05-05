from __future__ import annotations

from itertools import count
from typing import List, Optional, Set, Tuple

from .confCPU3 import ACS2ConfigurationCPU3


classifier_id_counterCPU3 = count()
SYMBOL_BITS_CPU3 = 6
SYMBOL_MASK_CPU3 = (1 << SYMBOL_BITS_CPU3) - 1


class ClassifierCPU3:
    def __init__(
        self,
        condition: List[str],
        action: int,
        effect: List[str],
        cfg: ACS2ConfigurationCPU3,
        time_created: int = 0,
        origin_source: str = "unknown",
        creation_episode: int = 0,
    ):
        self.id: int = next(classifier_id_counterCPU3)
        self.parents: Optional[Tuple[int, int]] = None
        self.origin_source = origin_source
        self.creation_episode = creation_episode
        self.cfg = cfg
        self.A = action
        self._C = list(condition)
        self._E = list(effect)
        self.q = 0.5
        self.r = 0.5
        self.ir = 0.0
        self.M: List[Set[str]] = [set() for _ in range(cfg.l_len)]
        self.t_ga = time_created
        self.t_alp = time_created
        self.aav = 0.0
        self.exp = 0
        self.num = 1
        self.condition_bits = (0, 0)
        self.wildcard_mask = (0, 0)
        self.effect_bits = (0, 0)
        self.effect_wildcard_mask = (0, 0)
        self._update_bitmasks()

    @property
    def C(self) -> List[str]:
        return self._C

    @C.setter
    def C(self, value: List[str]) -> None:
        self._C = list(value)
        self._update_bitmasks()

    @property
    def E(self) -> List[str]:
        return self._E

    @E.setter
    def E(self, value: List[str]) -> None:
        self._E = list(value)
        self._update_bitmasks()

    def _update_bitmasks(self) -> None:
        self.condition_bits = (0, 0)
        self.wildcard_mask = (0, 0)
        self.effect_bits = (0, 0)
        self.effect_wildcard_mask = (0, 0)
        for i, sym in enumerate(self._C):
            shift = i * SYMBOL_BITS_CPU3
            if sym != "#":
                if shift < 64:
                    self.condition_bits = (self.condition_bits[0] | (int(sym) << shift), self.condition_bits[1])
                    self.wildcard_mask = (self.wildcard_mask[0] | (SYMBOL_MASK_CPU3 << shift), self.wildcard_mask[1])
                else:
                    self.condition_bits = (self.condition_bits[0], self.condition_bits[1] | (int(sym) << (shift - 64)))
                    self.wildcard_mask = (self.wildcard_mask[0], self.wildcard_mask[1] | (SYMBOL_MASK_CPU3 << (shift - 64)))
        for i, sym in enumerate(self._E):
            shift = i * SYMBOL_BITS_CPU3
            if sym != "#":
                if shift < 64:
                    self.effect_bits = (self.effect_bits[0] | (int(sym) << shift), self.effect_bits[1])
                    self.effect_wildcard_mask = (self.effect_wildcard_mask[0] | (SYMBOL_MASK_CPU3 << shift), self.effect_wildcard_mask[1])
                else:
                    self.effect_bits = (self.effect_bits[0], self.effect_bits[1] | (int(sym) << (shift - 64)))
                    self.effect_wildcard_mask = (self.effect_wildcard_mask[0], self.effect_wildcard_mask[1] | (SYMBOL_MASK_CPU3 << (shift - 64)))

    def sync_from_bits(self) -> None:
        new_c = ["#" for _ in range(self.cfg.l_len)]
        new_e = ["#" for _ in range(self.cfg.l_len)]
        for i in range(self.cfg.l_len):
            shift = i * SYMBOL_BITS_CPU3
            if shift < 64:
                if (self.wildcard_mask[0] >> shift) & SYMBOL_MASK_CPU3:
                    new_c[i] = str((self.condition_bits[0] >> shift) & SYMBOL_MASK_CPU3)
                if (self.effect_wildcard_mask[0] >> shift) & SYMBOL_MASK_CPU3:
                    new_e[i] = str((self.effect_bits[0] >> shift) & SYMBOL_MASK_CPU3)
            else:
                if (self.wildcard_mask[1] >> (shift - 64)) & SYMBOL_MASK_CPU3:
                    new_c[i] = str((self.condition_bits[1] >> (shift - 64)) & SYMBOL_MASK_CPU3)
                if (self.effect_wildcard_mask[1] >> (shift - 64)) & SYMBOL_MASK_CPU3:
                    new_e[i] = str((self.effect_bits[1] >> (shift - 64)) & SYMBOL_MASK_CPU3)
        self._C = new_c
        self._E = new_e

    def specified_attribute_count(self) -> int:
        return (self.wildcard_mask[0].bit_count() + self.wildcard_mask[1].bit_count()) // SYMBOL_BITS_CPU3

    def matches(self, state: List[str]) -> bool:
        return all(sym == "#" or sym == state[i] for i, sym in enumerate(self._C))

    def matches_bits(self, state_bits: Tuple[int, int]) -> bool:
        return (state_bits[0] & self.wildcard_mask[0]) == self.condition_bits[0] and \
               (state_bits[1] & self.wildcard_mask[1]) == self.condition_bits[1]

    def predict_bits(self, state_bits: Tuple[int, int]) -> Tuple[int, int]:
        return ((state_bits[0] & (~self.effect_wildcard_mask[0])) | self.effect_bits[0],
                (state_bits[1] & (~self.effect_wildcard_mask[1])) | self.effect_bits[1])

    def get_anticipation(self, state: List[str]) -> List[str]:
        predicted = list(state)
        for i, sym in enumerate(self._E):
            if sym != "#":
                predicted[i] = sym
        return predicted

    @property
    def fitness(self) -> float:
        return self.q * self.r

    @property
    def key(self) -> Tuple[Tuple[int, int], Tuple[int, int], int, Tuple[int, int], Tuple[int, int]]:
        return (
            self.condition_bits,
            self.wildcard_mask,
            self.A,
            self.effect_bits,
            self.effect_wildcard_mask,
        )

    def copy(self) -> "ClassifierCPU3":
        new_cl = ClassifierCPU3(
            condition=list(self._C),
            action=self.A,
            effect=list(self._E),
            cfg=self.cfg,
            time_created=self.t_ga,
            origin_source=self.origin_source,
            creation_episode=self.creation_episode,
        )
        new_cl.q = self.q
        new_cl.r = self.r
        new_cl.ir = self.ir
        new_cl.exp = self.exp
        new_cl.num = self.num
        new_cl.aav = self.aav
        new_cl.t_alp = self.t_alp
        new_cl.parents = self.parents
        new_cl.M = [set(mark) for mark in self.M]
        new_cl.condition_bits = self.condition_bits
        new_cl.wildcard_mask = self.wildcard_mask
        new_cl.effect_bits = self.effect_bits
        new_cl.effect_wildcard_mask = self.effect_wildcard_mask
        new_cl.sync_from_bits()
        return new_cl

    def __repr__(self) -> str:
        self.sync_from_bits()
        return f"Cl(C={''.join(self._C)}, A={self.A}, E={''.join(self._E)}, q={self.q:.2f}, r={self.r:.2f}, exp={self.exp})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassifierCPU3):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)
