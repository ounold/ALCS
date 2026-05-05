from __future__ import annotations
from typing import List, Set, Optional, Tuple
from .confCPU2 import ACS2Configuration
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

        # --- Bitmask representation ---
        # Each symbol is encoded into 8 bits. Total bits = 8 * l_len.
        self.condition_bits: int = 0
        self.wildcard_mask: int = 0
        self.effect_bits: int = 0
        self.effect_wildcard_mask: int = 0
        self._update_bitmasks()

    def _update_bitmasks(self):
        """ Pre-calculates bitmasks for fast matching and prediction. """
        self.condition_bits = 0
        self.wildcard_mask = 0
        self.effect_bits = 0
        self.effect_wildcard_mask = 0
        for i, sym in enumerate(self.C):
            shift = i * 8
            if sym != '#':
                val = int(sym)
                self.condition_bits |= (val << shift)
                self.wildcard_mask |= (0xFF << shift)
        
        for i, sym in enumerate(self.E):
            shift = i * 8
            if sym != '#':
                val = int(sym)
                self.effect_bits |= (val << shift)
                self.effect_wildcard_mask |= (0xFF << shift)

    def matches(self, state: List[str]) -> bool:
        """
        Checks if the classifier's condition matches the given state.
        Deprecated: use matches_bits for performance.
        """
        for i, sym in enumerate(self.C):
            if sym != '#' and sym != state[i]:
                return False
        return True

    def matches_bits(self, state_bits: int) -> bool:
        """
        Checks if the classifier's condition matches the given state (in bit format).
        """
        return (state_bits & self.wildcard_mask) == self.condition_bits

    def predict_bits(self, state_bits: int) -> int:
        """
        Predicts the next state bits.
        """
        return (state_bits & (~self.effect_wildcard_mask)) | self.effect_bits

    def get_anticipation(self, state: List[str]) -> List[str]:
        """
        Predicts the next state based on the current state and the classifier's effect.
        """
        predicted_state = list(state)
        for i, sym in enumerate(self.E):
            if sym != '#':
                predicted_state[i] = sym
        return predicted_state

    def copy(self) -> Classifier:
        """
        Creates a lightweight copy of the classifier.
        """
        # Create a new instance without calling __init__ to avoid ID increment if needed,
        # but here we actually want a new ID for the offspring in GA, so we call __init__.
        # We use current attributes to initialize.
        new_cl = Classifier(
            condition=list(self.C),
            action=self.A,
            effect=list(self.E),
            cfg=self.cfg,
            time_created=self.t_ga,
            origin_source=self.origin_source,
            creation_episode=self.creation_episode
        )
        # Copy learning parameters
        new_cl.q = self.q
        new_cl.r = self.r
        new_cl.ir = self.ir
        new_cl.exp = self.exp
        new_cl.num = self.num
        new_cl.aav = self.aav
        new_cl.t_alp = self.t_alp
        
        # Deep copy the Mark set list
        new_cl.M = [set(m) for m in self.M]
        
        # Explicitly copy bitmasks to ensure they are immediate
        new_cl.condition_bits = self.condition_bits
        new_cl.wildcard_mask = self.wildcard_mask
        new_cl.effect_bits = self.effect_bits
        new_cl.effect_wildcard_mask = self.effect_wildcard_mask
        
        return new_cl

    @property
    def fitness(self) -> float:
        """
        Calculates the fitness of the classifier.
        """
        return self.q * self.r

    @property
    def key(self) -> Tuple:
        """ Returns a unique key for the classifier based on C, A, and E. """
        return (tuple(self.C), self.A, tuple(self.E))

    def __repr__(self) -> str:
        return f"Cl(C={''.join(self.C)}, A={self.A}, E={''.join(self.E)}, q={self.q:.2f}, r={self.r:.2f}, exp={self.exp})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Classifier):
            return NotImplemented
        return self.C == other.C and self.A == other.A and self.E == other.E

    def __hash__(self) -> int:
        return hash((tuple(self.C), self.A, tuple(self.E)))
