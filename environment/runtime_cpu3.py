from __future__ import annotations

import math
import random
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple

from environment.boolean_envs import label_carry_cpu, label_even_parity_cpu, label_multiplexer_cpu
from environment.gymnasium_cpu3 import GymnasiumEnvironmentCPU3
from environment.registry import EnvironmentSpec


class EnvironmentCPU3:
    type: str
    name: str
    num_actions: int
    state_len: int
    supports_policy_map: bool = False
    supports_metric_evaluation: bool = False

    def reset(self) -> List[str]:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[List[str], float, bool]:
        raise NotImplementedError

    def peek_step(self, state: Sequence[str], action: int) -> List[str]:
        raise NotImplementedError

    def metric_states(self) -> List[List[str]]:
        return []

    def optimal_avg_steps(self) -> float:
        return 0.0

    def to_metadata(self) -> dict:
        raise NotImplementedError


class GridEnvironmentCPU3(EnvironmentCPU3):
    supports_policy_map = True
    supports_metric_evaluation = True

    def __init__(self, rows: int, cols: int, start_pos: Tuple[int, int], goal_pos: Tuple[int, int], obstacles: Optional[Iterable[Tuple[int, int]]] = None, reset_to_random_start: bool = True, name: str = "grid_maze", maze_source: Optional[str] = None):
        self.type = "grid_maze"
        self.name = name
        self.rows = int(rows)
        self.cols = int(cols)
        self.start_pos = tuple(start_pos)
        self.goal_pos = tuple(goal_pos)
        self.obstacles = {tuple(obstacle) for obstacle in (obstacles or ())}
        self.reset_to_random_start = bool(reset_to_random_start)
        self.current_pos = list(self.start_pos)
        self.state_len = 2
        self.num_actions = 8
        self.maze_source = maze_source
        self.action_map = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
            4: (-1, -1),
            5: (-1, 1),
            6: (1, -1),
            7: (1, 1),
        }
        self._valid_starts = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in self.obstacles and (row, col) != self.goal_pos
        ]

    def reset(self) -> List[str]:
        if not self.reset_to_random_start or not self._valid_starts:
            self.current_pos = [self.start_pos[0], self.start_pos[1]]
        else:
            self.current_pos = list(random.choice(self._valid_starts))
        return [str(self.current_pos[0]), str(self.current_pos[1])]

    def step(self, action: int) -> Tuple[List[str], float, bool]:
        next_state, reward, done = self._compute_step(self.current_pos, action)
        self.current_pos = [int(next_state[0]), int(next_state[1])]
        return next_state, reward, done

    def peek_step(self, state: Sequence[str], action: int) -> List[str]:
        next_state, _, _ = self._compute_step([int(state[0]), int(state[1])], action)
        return next_state

    def metric_states(self) -> List[List[str]]:
        return [[str(row), str(col)] for row, col in self._valid_starts]

    def optimal_avg_steps(self) -> float:
        if not self._valid_starts:
            return 0.0
        total = 0
        moves = list(self.action_map.values())
        for start in self._valid_starts:
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                (curr_row, curr_col), dist = queue.popleft()
                if (curr_row, curr_col) == self.goal_pos:
                    total += dist
                    break
                for d_row, d_col in moves:
                    nxt = (curr_row + d_row, curr_col + d_col)
                    if 0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols and nxt not in visited and nxt not in self.obstacles:
                        visited.add(nxt)
                        queue.append((nxt, dist + 1))
        return total / len(self._valid_starts)

    def to_metadata(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "parameters": {
                "rows": self.rows,
                "cols": self.cols,
                "start_pos": list(self.start_pos),
                "goal_pos": list(self.goal_pos),
                "obstacles": [list(obstacle) for obstacle in sorted(self.obstacles)],
                "maze_source": self.maze_source,
            },
            "reset_to_random_start": self.reset_to_random_start,
        }

    def _compute_step(self, pos: List[int], action: int) -> Tuple[List[str], float, bool]:
        delta = self.action_map.get(action)
        if delta is None:
            return [str(pos[0]), str(pos[1])], -1.0, False
        next_row, next_col = pos[0] + delta[0], pos[1] + delta[1]
        if 0 <= next_row < self.rows and 0 <= next_col < self.cols and (next_row, next_col) not in self.obstacles:
            final_pos = [next_row, next_col]
        else:
            final_pos = list(pos)
        done = tuple(final_pos) == self.goal_pos
        return [str(final_pos[0]), str(final_pos[1])], (1000.0 if done else -1.0), done


class BinaryClassificationEnvironmentCPU3(EnvironmentCPU3):
    supports_metric_evaluation = True

    def __init__(self, name: str, problem_kind: str, input_bits: int, sampling: str = "random", left_bits: Optional[int] = None, right_bits: Optional[int] = None, address_bits: Optional[int] = None):
        self.type = "multiplexer" if problem_kind == "multiplexer" else "binary_classification"
        self.name = name
        self.problem_kind = problem_kind
        self.input_bits = int(input_bits)
        self.state_len = self.input_bits + 1
        self.num_actions = 2
        self.sampling = sampling
        self.left_bits = left_bits
        self.right_bits = right_bits
        self.address_bits = address_bits
        self.current_bits = [0] * self.input_bits
        self._cursor = 0
        self._metric_cache = self._build_metric_states(limit=8192)

    def reset(self) -> List[str]:
        if self.sampling == "exhaustive":
            index = self._cursor % (1 << self.input_bits)
            self._cursor += 1
            self.current_bits = [int(bit) for bit in format(index, f"0{self.input_bits}b")]
        else:
            self.current_bits = [random.randint(0, 1) for _ in range(self.input_bits)]
        return self._state(self.current_bits, 0)

    def step(self, action: int) -> Tuple[List[str], float, bool]:
        label = self._label(self.current_bits)
        success = int(int(action) == label)
        return self._state(self.current_bits, success), (1000.0 if success else -1.0), True

    def peek_step(self, state: Sequence[str], action: int) -> List[str]:
        bits = [int(symbol) for symbol in state[: self.input_bits]]
        return self._state(bits, int(int(action) == self._label(bits)))

    def metric_states(self) -> List[List[str]]:
        return list(self._metric_cache)

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

    def _state(self, bits: Sequence[int], outcome: int) -> List[str]:
        return [str(int(bit)) for bit in bits] + [str(int(outcome))]

    def _label(self, bits: Sequence[int]) -> int:
        if self.problem_kind == "even_parity":
            return label_even_parity_cpu(bits)
        if self.problem_kind == "carry":
            left_bits = self.left_bits or self.input_bits // 2
            right_bits = self.right_bits or (self.input_bits - left_bits)
            return label_carry_cpu(bits, left_bits, right_bits)
        if self.problem_kind == "multiplexer":
            address_bits = self.address_bits or max(1, int(math.log2(max(self.input_bits, 2))))
            return label_multiplexer_cpu(bits, address_bits)
        raise ValueError(f"Unsupported problem_kind: {self.problem_kind}")

    def _build_metric_states(self, limit: int) -> List[List[str]]:
        total = 1 << self.input_bits
        step = max(1, total // limit) if total > limit else 1
        states = []
        for index in range(0, total, step):
            bits = [int(bit) for bit in format(index, f"0{self.input_bits}b")]
            states.append(self._state(bits, 0))
        return states


def create_environmentCPU3(spec: EnvironmentSpec) -> EnvironmentCPU3:
    if spec.type == "grid_maze":
        params = spec.parameters
        return GridEnvironmentCPU3(int(params["rows"]), int(params["cols"]), tuple(params["start_pos"]), tuple(params["goal_pos"]), [tuple(item) for item in params.get("obstacles", [])], spec.reset_to_random_start, spec.name, params.get("maze_source"))
    if spec.type in {"multiplexer", "binary_classification"}:
        params = spec.parameters
        return BinaryClassificationEnvironmentCPU3(spec.name, params["problem_kind"], int(params["input_bits"]), str(params.get("sampling", "random")), params.get("left_bits"), params.get("right_bits"), params.get("address_bits"))
    if spec.type == "gymnasium":
        params = spec.parameters
        return GymnasiumEnvironmentCPU3(spec.name, str(params["env_id"]), str(params.get("observation_encoding", "discrete")), params.get("bins", []), params.get("is_slippery"))
    raise ValueError(f"Unsupported environment.type: {spec.type}")


def environment_from_metadataCPU3(metadata: dict) -> EnvironmentCPU3:
    spec = EnvironmentSpec(
        type=metadata["type"],
        name=metadata.get("name", metadata["type"]),
        parameters=dict(metadata.get("parameters", {})),
        reset_to_random_start=metadata.get("reset_to_random_start", True),
        observation_mode=metadata.get("observation_mode", "symbolic"),
        action_mode=metadata.get("action_mode", "discrete"),
        reward_mode=metadata.get("reward_mode", "default"),
    ).validate()
    return create_environmentCPU3(spec)


def calculate_optimal_metricsCPU3(env: EnvironmentCPU3) -> float:
    return env.optimal_avg_steps()
