from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.runtime_cpu3 import EnvironmentCPU3


class GymnasiumEnvironmentCPU3:
    def __init__(self, name: str, env_id: str, observation_encoding: str = "discrete", bins: Optional[Sequence[int]] = None, is_slippery: Optional[bool] = None):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ValueError("Gymnasium support requires the 'gymnasium' package in the active environment") from exc
        kwargs = {"is_slippery": bool(is_slippery)} if is_slippery is not None else {}
        self.env = gym.make(env_id, **kwargs)
        self.type = "gymnasium"
        self.name = name
        self.env_id = env_id
        self.observation_encoding = observation_encoding
        self.bins = list(bins or [])
        self.num_actions = int(self.env.action_space.n)
        self.state_len = self._infer_state_len()
        self.supports_metric_evaluation = False
        self.supports_policy_map = False

    def reset(self) -> List[str]:
        observation, _ = self.env.reset()
        return self._encode(observation)

    def step(self, action: int):
        observation, reward, terminated, truncated, _ = self.env.step(int(action))
        return self._encode(observation), float(reward), bool(terminated or truncated)

    def peek_step(self, state: Sequence[str], action: int) -> List[str]:
        raise NotImplementedError("peek_step is not available for Gymnasium environments")

    def metric_states(self) -> List[List[str]]:
        return []

    def optimal_avg_steps(self) -> float:
        return 0.0

    def to_metadata(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "parameters": {
                "env_id": self.env_id,
                "observation_encoding": self.observation_encoding,
                "bins": self.bins,
                "num_actions": self.num_actions,
                "encoded_state_length": self.state_len,
            },
            "reset_to_random_start": False,
        }

    def _infer_state_len(self) -> int:
        space = self.env.observation_space
        if getattr(space, "n", None) is not None:
            return 1
        if hasattr(space, "spaces"):
            return len(space.spaces)
        if self.observation_encoding == "binned":
            total = 1
            for size in getattr(space, "shape", ()) or ():
                total *= int(size)
            return total
        raise ValueError(f"Unsupported observation space: {space}")

    def _encode(self, observation: Any) -> List[str]:
        space = self.env.observation_space
        if getattr(space, "n", None) is not None:
            return [str(int(observation))]
        if hasattr(space, "spaces"):
            return [str(int(item)) for item in observation]
        if self.observation_encoding == "binned":
            flat = list(_flatten_numeric(observation))
            bins = self.bins or [8] * len(flat)
            lower = list(_flatten_numeric(space.low))
            upper = list(_flatten_numeric(space.high))
            encoded = []
            for value, low, high, n_bins in zip(flat, lower, upper, bins):
                if not math.isfinite(low):
                    low = -4.0
                if not math.isfinite(high):
                    high = 4.0
                clipped = min(max(float(value), low), high)
                bucket = min(int(((clipped - low) / max(high - low, 1e-9)) * n_bins), n_bins - 1)
                encoded.append(str(bucket))
            return encoded
        raise ValueError(f"Unsupported observation encoding: {self.observation_encoding}")


def _flatten_numeric(value: Any) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_numeric(item)
    elif hasattr(value, "tolist"):
        yield from _flatten_numeric(value.tolist())
    else:
        yield float(value)
