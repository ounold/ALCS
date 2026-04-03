from dataclasses import dataclass


@dataclass
class ACS2ConfigurationCPU3:
    l_len: int
    num_actions: int
    symbol_capacity: int = 8
    total_episodes: int = 0
    do_simple_mode: bool = False
    do_ga: bool = False
    do_subsumption: bool = True
    do_alp: bool = True
    alp_mark_only_incorrect: bool = True
    do_decay_epsilon: bool = False
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    beta: float = 0.05
    gamma: float = 0.95
    theta_i: float = 0.1
    theta_r: float = 0.9
    epsilon: float = 0.1
    u_max: int = 1
    theta_ga: int = 50
    mu: float = 0.3
    chi: float = 0.8
    theta_as: int = 20
    theta_exp: int = 20
    metric_calculation_frequency: int = 1

    def __post_init__(self) -> None:
        for name in ("beta", "gamma", "theta_i", "theta_r", "epsilon"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.l_len <= 0:
            raise ValueError("l_len must be positive")
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if self.symbol_capacity <= 0:
            raise ValueError("symbol_capacity must be positive")
        if self.metric_calculation_frequency < 1:
            raise ValueError("metric_calculation_frequency must be >= 1")
