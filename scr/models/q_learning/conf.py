from dataclasses import dataclass

@dataclass
class QLearningConfiguration:
    """
    Configuration for the Q-learning model.
    """
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    num_actions: int = 8 # This will be provided by the environment
