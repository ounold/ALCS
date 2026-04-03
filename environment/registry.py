from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from environment.maze_loader import load_acs2_maze_catalog


MAZE_LIBRARY = load_acs2_maze_catalog()


def _as_tuple2(value: Any, field_name: str) -> Tuple[int, int]:
    if value is None:
        raise ValueError(f"{field_name} is required")
    point = tuple(int(item) for item in value)
    if len(point) != 2:
        raise ValueError(f"{field_name} must contain exactly 2 integers")
    return point[0], point[1]


def _normalize_obstacles(value: Any) -> Tuple[Tuple[int, int], ...]:
    if value in (None, (), []):
        return ()
    return tuple(_as_tuple2(item, "obstacle") for item in value)


@dataclass(frozen=True)
class EnvironmentSpec:
    type: str
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reset_to_random_start: bool = True
    observation_mode: str = "symbolic"
    action_mode: str = "discrete"
    reward_mode: str = "default"

    def validate(self) -> "EnvironmentSpec":
        env_type = self.type.strip().lower()
        if env_type not in {"grid_maze", "multiplexer", "binary_classification", "gymnasium"}:
            raise ValueError(f"Unsupported environment.type: {self.type}")
        name = self.name.strip() or env_type
        params = dict(self.parameters)

        if env_type == "grid_maze":
            params = _resolve_grid_maze_parameters(name, params)
        elif env_type == "multiplexer":
            address_bits = int(params.get("address_bits", 2))
            data_bits = 1 << address_bits
            params = {
                **params,
                "problem_kind": "multiplexer",
                "address_bits": address_bits,
                "input_bits": address_bits + data_bits,
                "sampling": params.get("sampling", "random"),
            }
        elif env_type == "binary_classification":
            params = _resolve_binary_parameters(name, params)
        else:
            params = _resolve_gym_parameters(name, params)

        return EnvironmentSpec(
            type=env_type,
            name=name,
            parameters=params,
            reset_to_random_start=bool(self.reset_to_random_start),
            observation_mode=self.observation_mode,
            action_mode=self.action_mode,
            reward_mode=self.reward_mode,
        )

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "parameters": self.parameters,
            "reset_to_random_start": self.reset_to_random_start,
            "observation_mode": self.observation_mode,
            "action_mode": self.action_mode,
            "reward_mode": self.reward_mode,
        }

    @property
    def state_length(self) -> int:
        if self.type == "grid_maze":
            return 2
        if self.type in {"multiplexer", "binary_classification"}:
            return int(self.parameters["input_bits"]) + 1
        if self.type == "gymnasium":
            return int(self.parameters["encoded_state_length"])
        raise ValueError(f"Unsupported environment.type: {self.type}")

    @property
    def num_actions(self) -> int:
        if self.type == "grid_maze":
            return 8
        if self.type in {"multiplexer", "binary_classification"}:
            return 2
        if self.type == "gymnasium":
            return int(self.parameters["num_actions"])
        raise ValueError(f"Unsupported environment.type: {self.type}")

    @property
    def symbol_capacity(self) -> int:
        if self.type == "grid_maze":
            return max(int(self.parameters["rows"]), int(self.parameters["cols"]))
        if self.type in {"multiplexer", "binary_classification"}:
            return 2
        if self.type == "gymnasium":
            return int(self.parameters["symbol_capacity"])
        raise ValueError(f"Unsupported environment.type: {self.type}")

    @property
    def supports_policy_map(self) -> bool:
        return self.type == "grid_maze"


def environment_spec_from_mapping(mapping: Mapping[str, Any]) -> EnvironmentSpec:
    raw_env = dict(mapping)
    if "type" not in raw_env:
        raw_env = {
            "type": "grid_maze",
            "name": raw_env.get("name", "grid_maze"),
            "parameters": {
                "rows": raw_env.get("rows", 7),
                "cols": raw_env.get("cols", 7),
                "start_pos": raw_env.get("start_pos", [0, 0]),
                "goal_pos": raw_env.get("goal_pos", [0, 6]),
                "obstacles": raw_env.get("obstacles", []),
            },
            "reset_to_random_start": raw_env.get("reset_to_random_start", True),
        }
    return EnvironmentSpec(
        type=raw_env.get("type", "grid_maze"),
        name=raw_env.get("name", raw_env.get("type", "grid_maze")),
        parameters=dict(raw_env.get("parameters", {})),
        reset_to_random_start=raw_env.get("reset_to_random_start", True),
        observation_mode=raw_env.get("observation_mode", "symbolic"),
        action_mode=raw_env.get("action_mode", "discrete"),
        reward_mode=raw_env.get("reward_mode", "default"),
    ).validate()


def environment_spec_from_args(raw_env: MutableMapping[str, Any], fallback_name: Optional[str] = None) -> EnvironmentSpec:
    env_type = str(raw_env.get("type", "grid_maze"))
    env_name = str(raw_env.get("name", fallback_name or env_type))
    if env_type == "grid_maze":
        parameters = {}
        if "rows" in raw_env and raw_env.get("rows") is not None:
            parameters["rows"] = int(raw_env.get("rows"))
        if "cols" in raw_env and raw_env.get("cols") is not None:
            parameters["cols"] = int(raw_env.get("cols"))
        if "start_pos" in raw_env and raw_env.get("start_pos") is not None:
            parameters["start_pos"] = list(raw_env.get("start_pos"))
        if "goal_pos" in raw_env and raw_env.get("goal_pos") is not None:
            parameters["goal_pos"] = list(raw_env.get("goal_pos"))
        if "obstacles" in raw_env and raw_env.get("obstacles") is not None:
            parameters["obstacles"] = list(raw_env.get("obstacles"))
    elif env_type == "multiplexer":
        parameters = {
            "address_bits": int(raw_env.get("address_bits", 2)),
            "sampling": raw_env.get("sampling", "random"),
        }
    elif env_type == "binary_classification":
        parameters = {
            "problem_kind": raw_env.get("problem_kind", "even_parity"),
            "input_bits": int(raw_env.get("input_bits", 4)),
            "left_bits": raw_env.get("left_bits"),
            "right_bits": raw_env.get("right_bits"),
            "sampling": raw_env.get("sampling", "random"),
        }
    else:
        parameters = {
            "env_id": raw_env.get("env_id", env_name),
            "observation_encoding": raw_env.get("observation_encoding", "auto"),
            "bins": list(raw_env.get("bins", [])),
        }
        if raw_env.get("is_slippery") is not None:
            parameters["is_slippery"] = raw_env.get("is_slippery")
    return EnvironmentSpec(
        type=env_type,
        name=env_name,
        parameters=parameters,
        reset_to_random_start=bool(raw_env.get("reset_to_random_start", env_type == "grid_maze")),
    ).validate()


def _resolve_grid_maze_parameters(name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    if name in MAZE_LIBRARY:
        definition = MAZE_LIBRARY[name]
        merged = {**definition.to_parameters(), **dict(params)}
        merged["start_pos"] = list(_as_tuple2(merged.get("start_pos"), "start_pos"))
        merged["goal_pos"] = list(_as_tuple2(merged.get("goal_pos"), "goal_pos"))
        merged["obstacles"] = [list(item) for item in _normalize_obstacles(merged.get("obstacles"))]
        return merged
    rows = int(params.get("rows", 7))
    cols = int(params.get("cols", 7))
    return {
        **dict(params),
        "rows": rows,
        "cols": cols,
        "start_pos": list(_as_tuple2(params.get("start_pos", [0, 0]), "start_pos")),
        "goal_pos": list(_as_tuple2(params.get("goal_pos", [0, cols - 1]), "goal_pos")),
        "obstacles": [list(item) for item in _normalize_obstacles(params.get("obstacles"))],
    }


def _resolve_binary_parameters(name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    problem_kind = str(params.get("problem_kind", name.split("_", 1)[0])).strip().lower()
    resolved = dict(params)
    resolved["problem_kind"] = problem_kind
    resolved["sampling"] = str(resolved.get("sampling", "random")).lower()
    if problem_kind == "carry":
        left_bits = int(resolved.get("left_bits") or resolved.get("input_bits", 8) // 2)
        right_bits = int(resolved.get("right_bits") or resolved.get("input_bits", left_bits * 2) - left_bits)
        resolved["left_bits"] = left_bits
        resolved["right_bits"] = right_bits
        resolved["input_bits"] = left_bits + right_bits
    elif problem_kind == "multiplexer":
        address_bits = int(resolved.get("address_bits", 2))
        resolved["address_bits"] = address_bits
        resolved["input_bits"] = address_bits + (1 << address_bits)
    else:
        resolved["input_bits"] = int(resolved.get("input_bits", _infer_width_from_name(name, default=4)))
    return resolved


def _resolve_gym_parameters(name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = dict(params)
    resolved["env_id"] = resolved.get("env_id", name)
    encoding = str(resolved.get("observation_encoding", "auto")).lower()
    if encoding == "auto":
        encoding = "discrete"
    resolved["observation_encoding"] = encoding
    resolved["bins"] = [int(item) for item in resolved.get("bins", [])]
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ValueError("Gymnasium support requires the 'gymnasium' package in the active environment") from exc
    env = gym.make(resolved["env_id"], **_gym_make_kwargs(resolved))
    try:
        action_space = env.action_space
        if getattr(action_space, "n", None) is None:
            raise ValueError(f"Unsupported gymnasium action space for {resolved['env_id']}: {action_space}")
        encoded_state_length, symbol_capacity = _infer_gym_observation_shape(env, encoding, resolved["bins"])
        resolved["num_actions"] = int(action_space.n)
        resolved["encoded_state_length"] = encoded_state_length
        resolved["symbol_capacity"] = symbol_capacity
    finally:
        env.close()
    return resolved


def _gym_make_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    kwargs = {}
    if "is_slippery" in params:
        kwargs["is_slippery"] = bool(params["is_slippery"])
    return kwargs


def _infer_gym_observation_shape(env: Any, encoding: str, bins: list[int]) -> Tuple[int, int]:
    space = env.observation_space
    if getattr(space, "n", None) is not None:
        return 1, int(space.n)
    if hasattr(space, "spaces"):
        widths = [getattr(subspace, "n", None) for subspace in space.spaces]
        if any(width is None for width in widths):
            raise ValueError(f"Unsupported tuple observation space for symbolic encoding: {space}")
        return len(space.spaces), max(int(width) for width in widths)
    if encoding == "binned":
        shape = getattr(space, "shape", None)
        if not shape:
            raise ValueError(f"Unsupported binned observation space: {space}")
        flat_size = 1
        for size in shape:
            flat_size *= int(size)
        if bins and len(bins) != flat_size:
            raise ValueError("bins length must match the flattened observation size")
        bin_sizes = bins or [8] * flat_size
        return flat_size, max(int(size) for size in bin_sizes)
    raise ValueError(f"Unsupported gymnasium observation space for symbolic encoding: {space}")


def _infer_width_from_name(name: str, default: int) -> int:
    for chunk in reversed(name.split("_")):
        if chunk.isdigit():
            return int(chunk)
    return default
