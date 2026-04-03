from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from environment.boolean_envs import label_carry_gpu, label_even_parity_gpu, label_multiplexer_gpu
from environment.gymnasium_gpu4 import GymnasiumEnvironmentGPU4
from environment.registry import EnvironmentSpec


class EnvironmentGPU4:
    type: str
    name: str
    num_actions: int
    state_len: int
    supports_policy_map: bool = False
    supports_metric_evaluation: bool = False

    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def peek_step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metric_states(self) -> torch.Tensor:
        return torch.empty((0, self.state_len), device=self.device, dtype=torch.long)

    def optimal_avg_steps(self) -> float:
        return 0.0

    def to_metadata(self) -> dict:
        raise NotImplementedError


class GridEnvironmentGPU4(EnvironmentGPU4):
    supports_policy_map = True
    supports_metric_evaluation = True

    def __init__(self, n_exp: int, rows: int, cols: int, start_pos: Tuple[int, int], goal_pos: Tuple[int, int], obstacles: Optional[Iterable[Tuple[int, int]]] = None, device: str = "cpu", reset_to_random_start: bool = True, name: str = "grid_maze", maze_source: Optional[str] = None):
        self.type = "grid_maze"
        self.name = name
        self.n_exp = int(n_exp)
        self.rows = int(rows)
        self.cols = int(cols)
        self.device = torch.device(device)
        self.start_pos = torch.tensor(start_pos, device=self.device, dtype=torch.long)
        self.goal_pos = torch.tensor(goal_pos, device=self.device, dtype=torch.long)
        self.reset_to_random_start = bool(reset_to_random_start)
        self.state_len = 2
        self.num_actions = 8
        self.maze_source = maze_source
        self.obs_mask = torch.zeros((self.rows, self.cols), device=self.device, dtype=torch.bool)
        for row, col in obstacles or ():
            self.obs_mask[row, col] = True
        self.current_pos = torch.zeros((self.n_exp, 2), device=self.device, dtype=torch.long)
        self.action_deltas = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]], device=self.device, dtype=torch.long)
        grid_indices = torch.arange(self.rows * self.cols, device=self.device).view(self.rows, self.cols)
        goal_index = self.goal_pos[0] * self.cols + self.goal_pos[1]
        valid_indices = torch.where(~self.obs_mask & (grid_indices != goal_index))
        self.valid_coords = torch.stack(valid_indices, dim=1)

    def reset(self) -> torch.Tensor:
        if not self.reset_to_random_start:
            self.current_pos[:] = self.start_pos
        else:
            idx = torch.randint(0, max(len(self.valid_coords), 1), (self.n_exp,), device=self.device)
            self.current_pos = self.valid_coords[idx]
        return self.current_pos.clone()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deltas = self.action_deltas[actions]
        new_pos = self.current_pos + deltas
        in_bounds = (new_pos[:, 0] >= 0) & (new_pos[:, 0] < self.rows) & (new_pos[:, 1] >= 0) & (new_pos[:, 1] < self.cols)
        candidate = torch.where(in_bounds.unsqueeze(1), new_pos, self.current_pos)
        can_move = in_bounds & (~self.obs_mask[candidate[:, 0], candidate[:, 1]])
        self.current_pos = torch.where(can_move.unsqueeze(1), new_pos, self.current_pos)
        done = torch.all(self.current_pos == self.goal_pos, dim=1)
        reward = torch.where(done, torch.full((self.n_exp,), 1000.0, device=self.device), torch.full((self.n_exp,), -1.0, device=self.device))
        return self.current_pos.clone(), reward, done

    def peek_step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        deltas = self.action_deltas[actions]
        new_pos = states + deltas
        in_bounds = (new_pos[:, 0] >= 0) & (new_pos[:, 0] < self.rows) & (new_pos[:, 1] >= 0) & (new_pos[:, 1] < self.cols)
        candidate = torch.where(in_bounds.unsqueeze(1), new_pos, states)
        can_move = in_bounds & (~self.obs_mask[candidate[:, 0], candidate[:, 1]])
        return torch.where(can_move.unsqueeze(1), new_pos, states)

    def metric_states(self) -> torch.Tensor:
        return self.valid_coords

    def optimal_avg_steps(self) -> float:
        obstacle_set = {tuple(item) for item in self.to_metadata()["parameters"]["obstacles"]}
        moves = [tuple(item) for item in self.action_deltas.cpu().tolist()]
        goal = tuple(self.goal_pos.cpu().tolist())
        starts = [(row, col) for row in range(self.rows) for col in range(self.cols) if (row, col) != goal and (row, col) not in obstacle_set]
        if not starts:
            return 0.0
        from collections import deque

        total = 0
        for start in starts:
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                (curr_row, curr_col), dist = queue.popleft()
                if (curr_row, curr_col) == goal:
                    total += dist
                    break
                for d_row, d_col in moves:
                    nxt = (curr_row + d_row, curr_col + d_col)
                    if 0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols and nxt not in visited and nxt not in obstacle_set:
                        visited.add(nxt)
                        queue.append((nxt, dist + 1))
        return total / len(starts)

    def to_metadata(self) -> dict:
        obstacle_rows, obstacle_cols = torch.where(self.obs_mask)
        return {
            "type": self.type,
            "name": self.name,
            "parameters": {
                "rows": self.rows,
                "cols": self.cols,
                "start_pos": self.start_pos.tolist(),
                "goal_pos": self.goal_pos.tolist(),
                "obstacles": [[int(row), int(col)] for row, col in zip(obstacle_rows.tolist(), obstacle_cols.tolist())],
                "maze_source": self.maze_source,
            },
            "reset_to_random_start": self.reset_to_random_start,
        }


class BinaryClassificationEnvironmentGPU4(EnvironmentGPU4):
    supports_metric_evaluation = True

    def __init__(self, n_exp: int, device: str, name: str, problem_kind: str, input_bits: int, sampling: str = "random", left_bits: Optional[int] = None, right_bits: Optional[int] = None, address_bits: Optional[int] = None):
        self.type = "multiplexer" if problem_kind == "multiplexer" else "binary_classification"
        self.name = name
        self.n_exp = int(n_exp)
        self.device = torch.device(device)
        self.problem_kind = problem_kind
        self.input_bits = int(input_bits)
        self.state_len = self.input_bits + 1
        self.num_actions = 2
        self.sampling = sampling
        self.left_bits = left_bits
        self.right_bits = right_bits
        self.address_bits = address_bits
        self.current_bits = torch.zeros((self.n_exp, self.input_bits), device=self.device, dtype=torch.long)
        self._cursor = 0
        self._metric_cache = self._build_metric_states(limit=8192)

    def reset(self) -> torch.Tensor:
        if self.sampling == "exhaustive":
            indices = (torch.arange(self.n_exp, device=self.device) + self._cursor) % (1 << self.input_bits)
            self._cursor = int((self._cursor + self.n_exp) % (1 << self.input_bits))
            self.current_bits = ((indices.unsqueeze(1) >> torch.arange(self.input_bits - 1, -1, -1, device=self.device)) & 1).long()
        else:
            self.current_bits = torch.randint(0, 2, (self.n_exp, self.input_bits), device=self.device)
        return torch.cat([self.current_bits, torch.zeros((self.n_exp, 1), device=self.device, dtype=torch.long)], dim=1)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = self._label(self.current_bits)
        success = (actions.long() == labels).long()
        next_states = torch.cat([self.current_bits, success.unsqueeze(1)], dim=1)
        rewards = torch.where(success.bool(), torch.full((self.n_exp,), 1000.0, device=self.device), torch.full((self.n_exp,), -1.0, device=self.device))
        dones = torch.ones(self.n_exp, device=self.device, dtype=torch.bool)
        return next_states, rewards, dones

    def peek_step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        bits = states[:, : self.input_bits].long()
        success = (actions.long() == self._label(bits)).long()
        return torch.cat([bits, success.unsqueeze(1)], dim=1)

    def metric_states(self) -> torch.Tensor:
        return self._metric_cache

    def optimal_avg_steps(self) -> float:
        return 1.0

    def to_metadata(self) -> dict:
        parameters = {"problem_kind": self.problem_kind, "input_bits": self.input_bits, "sampling": self.sampling}
        if self.left_bits is not None:
            parameters["left_bits"] = self.left_bits
        if self.right_bits is not None:
            parameters["right_bits"] = self.right_bits
        if self.address_bits is not None:
            parameters["address_bits"] = self.address_bits
        return {"type": self.type, "name": self.name, "parameters": parameters, "reset_to_random_start": False}

    def _label(self, bits: torch.Tensor) -> torch.Tensor:
        if self.problem_kind == "even_parity":
            return label_even_parity_gpu(bits)
        if self.problem_kind == "carry":
            left_bits = self.left_bits or self.input_bits // 2
            right_bits = self.right_bits or (self.input_bits - left_bits)
            return label_carry_gpu(bits, left_bits, right_bits)
        if self.problem_kind == "multiplexer":
            address_bits = self.address_bits or max(1, int(math.log2(max(self.input_bits, 2))))
            return label_multiplexer_gpu(bits, address_bits)
        raise ValueError(f"Unsupported problem_kind: {self.problem_kind}")

    def _build_metric_states(self, limit: int) -> torch.Tensor:
        total = 1 << self.input_bits
        step = max(1, total // limit) if total > limit else 1
        indices = torch.arange(0, total, step, device=self.device)
        bits = ((indices.unsqueeze(1) >> torch.arange(self.input_bits - 1, -1, -1, device=self.device)) & 1).long()
        return torch.cat([bits, torch.zeros((bits.shape[0], 1), device=self.device, dtype=torch.long)], dim=1)


def create_environmentGPU4(spec: EnvironmentSpec, n_exp: int, device: str) -> EnvironmentGPU4:
    if spec.type == "grid_maze":
        params = spec.parameters
        return GridEnvironmentGPU4(n_exp, int(params["rows"]), int(params["cols"]), tuple(params["start_pos"]), tuple(params["goal_pos"]), [tuple(item) for item in params.get("obstacles", [])], device, spec.reset_to_random_start, spec.name, params.get("maze_source"))
    if spec.type in {"multiplexer", "binary_classification"}:
        params = spec.parameters
        return BinaryClassificationEnvironmentGPU4(n_exp, device, spec.name, params["problem_kind"], int(params["input_bits"]), str(params.get("sampling", "random")), params.get("left_bits"), params.get("right_bits"), params.get("address_bits"))
    if spec.type == "gymnasium":
        params = spec.parameters
        return GymnasiumEnvironmentGPU4(n_exp, device, spec.name, str(params["env_id"]), str(params.get("observation_encoding", "discrete")), params.get("bins", []), params.get("is_slippery"))
    raise ValueError(f"Unsupported environment.type: {spec.type}")


def environment_from_metadataGPU4(metadata: dict, n_exp: int, device: str = "cpu") -> EnvironmentGPU4:
    spec = EnvironmentSpec(
        type=metadata["type"],
        name=metadata.get("name", metadata["type"]),
        parameters=dict(metadata.get("parameters", {})),
        reset_to_random_start=metadata.get("reset_to_random_start", True),
        observation_mode=metadata.get("observation_mode", "symbolic"),
        action_mode=metadata.get("action_mode", "discrete"),
        reward_mode=metadata.get("reward_mode", "default"),
    ).validate()
    return create_environmentGPU4(spec, n_exp, device)


def calculate_optimal_metricsGPU4(environment: EnvironmentGPU4) -> float:
    return environment.optimal_avg_steps()
