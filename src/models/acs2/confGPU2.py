from dataclasses import dataclass, field
from typing import List

@dataclass
class ACS2Configuration:
    """
    Data class for holding ACS2 algorithm parameters.
    This version is for the GPU implementation.
    """
    l_len: int
    num_actions: int
    total_episodes: int = 0

    # --- Control Flags ---
    do_simple_mode: bool = False
    do_ga: bool = False
    do_subsumption: bool = True
    do_alp: bool = True

    # --- Epsilon Decay ---
    do_decay_epsilon: bool = False
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Learning Parameters
    beta: float = 0.05
    gamma: float = 0.95
    theta_i: float = 0.1
    theta_r: float = 0.9
    epsilon: float = 0.1
    u_max: int = 1

    # Genetic Algorithm (GA)
    theta_ga: int = 50
    mu: float = 0.3
    chi: float = 0.8
    theta_as: int = 20  # Max size of Action Set
    theta_exp: int = 20  # Experience threshold for subsumption

    def __post_init__(self):
        if not (0 <= self.beta <= 1):
            raise ValueError("beta must be in [0, 1]")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")