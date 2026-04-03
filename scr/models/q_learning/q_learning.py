from collections import defaultdict
import random
from typing import List, Tuple

from .conf import QLearningConfiguration

class QLearning:
    """
    A simple Q-learning implementation.
    """
    def __init__(self, cfg: QLearningConfiguration):
        self.cfg = cfg
        self.q_table = defaultdict(lambda: [0.0] * cfg.num_actions)

    def run_step(self, env_state: List[str], explore: bool = True) -> Tuple[int, None]:
        """
        Selects an action using an epsilon-greedy policy.
        Returns the action and None (to match the ACS2 signature).
        """
        state_key = tuple(env_state)
        
        if explore and random.random() < self.cfg.epsilon:
            action = random.randint(0, self.cfg.num_actions - 1)
        else:
            q_values = self.q_table[state_key]
            action = q_values.index(max(q_values))
            
        return action, None # Return None for action_set

    def apply_learning(self, state: List[str], action: int, reward: float, next_state: List[str], done: bool):
        """
        Applies the Q-learning update rule.
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        old_value = self.q_table[state_key][action]
        
        if done:
            next_max = 0.0
        else:
            next_max = max(self.q_table[next_state_key])
        
        new_value = (1 - self.cfg.learning_rate) * old_value + self.cfg.learning_rate * (reward + self.cfg.discount_factor * next_max)
        self.q_table[state_key][action] = new_value