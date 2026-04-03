from __future__ import annotations
from typing import List, Set, Optional, Tuple
from .confGPU2 import ACS2Configuration
from itertools import count
import cupy as cp

# Module-level counter for unique classifier IDs
classifier_id_counter = count()


class Classifier:
    """
    Represents a single rule in ACS2 (GPU version).
    """
    def __init__(self, condition: List[str], action: int, effect: List[str], cfg: ACS2Configuration, time_created: int = 0, origin_source: str = "unknown", creation_episode: int = 0):
        self.id: int = next(classifier_id_counter)
        self.parents: Optional[Tuple[int, int]] = None
        self.origin_source: str = origin_source
        self.creation_episode: int = creation_episode
        
        self.C: List[str] = condition
        self.A: int = action
        self.E: List[str] = effect
        self.cfg: ACS2Configuration = cfg

        self.q: float = 0.5  # Quality
        self.r: float = 0.5  # Reward
        self.ir: float = 0.0  # Immediate reward

        self.M: List[Set[str]] = [set() for _ in range(cfg.l_len)]

        self.t_ga: int = time_created
        self.t_alp: int = time_created
        self.aav: float = 0.0
        self.exp: int = 0
        self.num: int = 1

    def matches(self, state: List[str]) -> bool:
        for i, sym in enumerate(self.C):
            if sym != '#' and sym != state[i]:
                return False
        return True

    def get_anticipation(self, state: List[str]) -> List[str]:
        predicted_state = list(state)
        for i, sym in enumerate(self.E):
            if sym != '#':
                predicted_state[i] = sym
        return predicted_state

    @property
    def fitness(self) -> float:
        return self.q * self.r

    def __repr__(self) -> str:
        return f"Cl(C={''.join(self.C)}, A={self.A}, E={''.join(self.E)}, q={self.q:.2f}, r={self.r:.2f}, exp={self.exp})"