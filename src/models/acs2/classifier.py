from __future__ import annotations
from typing import List, Set, Optional, Tuple
from .conf import ACS2Configuration
from itertools import count

# Module-level counter for unique classifier IDs
classifier_id_counter = count()


class Classifier:
    """
    Represents a single rule in ACS2.
    """
    def __init__(self, condition: List[str], action: int, effect: List[str], cfg: ACS2Configuration, time_created: int = 0, origin_source: str = "unknown", creation_episode: int = 0):
        self.id: int = next(classifier_id_counter)
        self.parents: Optional[Tuple[int, int]] = None
        self.origin_source: str = origin_source # New attribute
        self.creation_episode: int = creation_episode # New attribute
        
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
        """
        Checks if the classifier's condition matches the given state.
        """
        for i, sym in enumerate(self.C):
            if sym != '#' and sym != state[i]:
                return False
        return True

    def get_anticipation(self, state: List[str]) -> List[str]:
        """
        Predicts the next state based on the current state and the classifier's effect.
        """
        predicted_state = list(state)
        for i, sym in enumerate(self.E):
            if sym != '#':
                predicted_state[i] = sym
        return predicted_state

    @property
    def fitness(self) -> float:
        """
        Calculates the fitness of the classifier.
        """
        return self.q * self.r

    def __repr__(self) -> str:
        return f"Cl(C={''.join(self.C)}, A={self.A}, E={''.join(self.E)}, q={self.q:.2f}, r={self.r:.2f}, exp={self.exp})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Classifier):
            return NotImplemented
        return self.C == other.C and self.A == other.A and self.E == other.E

    def __hash__(self) -> int:
        return hash((tuple(self.C), self.A, tuple(self.E)))
