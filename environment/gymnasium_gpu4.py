from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence

import torch


class GymnasiumEnvironmentGPU4:
    def __init__(self, n_exp: int, device: str, name: str, env_id: str, observation_encoding: str = "discrete", bins: Optional[Sequence[int]] = None, is_slippery: Optional[bool] = None):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ValueError("Gymnasium support requires the 'gymnasium' package in the active environment") from exc
        kwargs = {"is_slippery": bool(is_slippery)} if is_slippery is not None else {}
        self.envs = [gym.make(env_id, **kwargs) for _ in range(int(n_exp))]
        self.device = torch.device(device)
        self.type = "gymnasium"
        self.name = name
        self.env_id = env_id
        self.observation_encoding = observation_encoding
        self.bins = list(bins or [])
        self.num_actions = int(self.envs[0].action_space.n)
        self.state_len = self._infer_state_len()
        self.supports_metric_evaluation = False
        self.supports_policy_map = False

    def reset(self) -> torch.Tensor:
        states = []
        for env in self.envs:
            observation, _ = env.reset()
            states.append(self._encode(env, observation))
        return torch.tensor(states, device=self.device, dtype=torch.long)

    def step(self, actions: torch.Tensor):
        next_states = []
        rewards = []
        dones = []
        for env, action in zip(self.envs, actions.tolist()):
            observation, reward, terminated, truncated, _ = env.step(int(action))
            next_states.append(self._encode(env, observation))
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))
        return (
            torch.tensor(next_states, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.bool),
        )

    def peek_step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("peek_step is not available for Gymnasium environments")

    def metric_states(self) -> torch.Tensor:
        return torch.empty((0, self.state_len), device=self.device, dtype=torch.long)

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
        space = self.envs[0].observation_space
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

    def _encode(self, env: Any, observation: Any) -> List[int]:
        space = env.observation_space
        if getattr(space, "n", None) is not None:
            return [int(observation)]
        if hasattr(space, "spaces"):
            return [int(item) for item in observation]
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
                encoded.append(bucket)
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
