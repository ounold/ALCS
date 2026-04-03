from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from environment.registry import EnvironmentSpec, environment_spec_from_args, environment_spec_from_mapping
from src.defaults_cpu3_gpu4 import DEFAULT_ENVIRONMENT, DEFAULT_EXPERIMENT_VALUES


def parse_boolGPU4(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class PhaseConfigGPU4:
    episodes: int
    epsilon: float
    beta: float
    alp: bool
    ga: bool
    decay: bool

    def validate(self, phase_name: str) -> "PhaseConfigGPU4":
        if self.episodes < 0:
            raise ValueError(f"{phase_name}.episodes must be >= 0")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"{phase_name}.epsilon must be in [0, 1]")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"{phase_name}.beta must be in [0, 1]")
        return self

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "beta": self.beta,
            "alp": self.alp,
            "ga": self.ga,
            "decay": self.decay,
        }


@dataclass(frozen=True)
class ExperimentConfigGPU4:
    n_exp: int
    seed: int
    n_steps: int
    beta: float
    gamma: float
    theta_i: float
    theta_r: float
    epsilon: float
    u_max: int
    theta_ga: int
    mu: float
    chi: float
    theta_as: int
    theta_exp: int
    alp_mark_only_incorrect: bool
    no_subsumption: bool
    metric_calculation_frequency: int
    max_population: int
    environment: EnvironmentSpec
    phases: Dict[str, PhaseConfigGPU4]
    device: str = "auto"
    steps_to_goal_threshold: Optional[int] = None

    def validate(self) -> "ExperimentConfigGPU4":
        if self.n_exp <= 0 or self.n_steps <= 0:
            raise ValueError("n_exp and n_steps must be positive")
        if self.metric_calculation_frequency < 1:
            raise ValueError("metric_calculation_frequency must be >= 1")
        if self.max_population <= 0:
            raise ValueError("max_population must be positive")
        self.environment.validate()
        total = 0
        for phase_name in ("explore", "exploit1", "exploit2"):
            if phase_name not in self.phases:
                raise ValueError(f"missing phase configuration: {phase_name}")
            total += self.phases[phase_name].validate(phase_name).episodes
        if total <= 0:
            raise ValueError("at least one phase must contain episodes")
        return self

    @property
    def total_episodes(self) -> int:
        return sum(phase.episodes for phase in self.phases.values())

    @property
    def params_phases(self) -> Dict[str, Dict[str, Any]]:
        return {name: phase.to_metadata() for name, phase in self.phases.items()}


def load_yaml_defaultsGPU4(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_arg_parserGPU4(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACS2 Experiment Runner (GPU4)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=defaults.get("device", "auto"))
    parser.add_argument("--max_population", type=int, default=defaults.get("max_population", 1024))
    parser.add_argument("--save_dashboard_data", action="store_true")
    parser.add_argument("--load_dashboard_data", type=str)
    env_defaults_raw = defaults.get("environment", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
    parser.add_argument("--environment_type", type=str, default=env_defaults_raw.get("type", DEFAULT_ENVIRONMENT["type"]))
    parser.add_argument("--environment_name", type=str, default=env_defaults_raw.get("name", DEFAULT_ENVIRONMENT["name"]))

    general = parser.add_argument_group("General Parameters")
    general.add_argument("--n_exp", type=int, default=defaults.get("n_exp", DEFAULT_EXPERIMENT_VALUES["n_exp"]))
    general.add_argument("--seed", type=int, default=defaults.get("seed", DEFAULT_EXPERIMENT_VALUES["seed"]))
    general.add_argument("--n_steps", type=int, default=defaults.get("n_steps", DEFAULT_EXPERIMENT_VALUES["n_steps"]))
    general.add_argument("--beta", type=float, default=defaults.get("beta", DEFAULT_EXPERIMENT_VALUES["beta"]))
    general.add_argument("--gamma", type=float, default=defaults.get("gamma", DEFAULT_EXPERIMENT_VALUES["gamma"]))
    general.add_argument("--theta_i", type=float, default=defaults.get("theta_i", DEFAULT_EXPERIMENT_VALUES["theta_i"]))
    general.add_argument("--theta_r", type=float, default=defaults.get("theta_r", DEFAULT_EXPERIMENT_VALUES["theta_r"]))
    general.add_argument("--epsilon", type=float, default=defaults.get("epsilon", DEFAULT_EXPERIMENT_VALUES["epsilon"]))
    general.add_argument("--u_max", type=int, default=defaults.get("u_max", DEFAULT_EXPERIMENT_VALUES["u_max"]))
    general.add_argument("--steps_to_goal_threshold", type=int, default=defaults.get("steps_to_goal_threshold", DEFAULT_EXPERIMENT_VALUES["steps_to_goal_threshold"]))
    general.add_argument("--metric_calculation_frequency", type=int, default=defaults.get("metric_calculation_frequency", DEFAULT_EXPERIMENT_VALUES["metric_calculation_frequency"]))
    general.add_argument("--alp_mark_only_incorrect", type=parse_boolGPU4, default=defaults.get("alp_mark_only_incorrect", DEFAULT_EXPERIMENT_VALUES["alp_mark_only_incorrect"]))
    general.add_argument("--no_subsumption", type=parse_boolGPU4, nargs="?", const=True, default=defaults.get("no_subsumption", False))

    environment_defaults = environment_spec_from_mapping(env_defaults_raw).to_metadata()
    params_defaults = environment_defaults["parameters"]
    env_group = parser.add_argument_group("Environment Parameters")
    env_group.add_argument("--rows", type=int, default=params_defaults.get("rows", 7))
    env_group.add_argument("--cols", type=int, default=params_defaults.get("cols", 7))
    env_group.add_argument("--start_pos", type=int, nargs=2, default=params_defaults.get("start_pos", [0, 0]))
    env_group.add_argument("--goal_pos", type=int, nargs=2, default=params_defaults.get("goal_pos", [0, 6]))
    env_group.add_argument("--obstacles", type=int, nargs="*", default=params_defaults.get("obstacles", []))
    env_group.add_argument("--reset_to_random_start", type=parse_boolGPU4, default=environment_defaults.get("reset_to_random_start", True))
    env_group.add_argument("--address_bits", type=int, default=params_defaults.get("address_bits", 2))
    env_group.add_argument("--problem_kind", type=str, default=params_defaults.get("problem_kind", "even_parity"))
    env_group.add_argument("--input_bits", type=int, default=params_defaults.get("input_bits", 4))
    env_group.add_argument("--left_bits", type=int, default=params_defaults.get("left_bits"))
    env_group.add_argument("--right_bits", type=int, default=params_defaults.get("right_bits"))
    env_group.add_argument("--sampling", type=str, default=params_defaults.get("sampling", "random"))
    env_group.add_argument("--env_id", type=str, default=params_defaults.get("env_id"))
    env_group.add_argument("--observation_encoding", type=str, default=params_defaults.get("observation_encoding", "auto"))
    env_group.add_argument("--bins", type=int, nargs="*", default=params_defaults.get("bins", []))
    env_group.add_argument("--is_slippery", type=parse_boolGPU4, default=params_defaults.get("is_slippery"))

    ga_group = parser.add_argument_group("GA Parameters")
    ga_group.add_argument("--theta_ga", type=int, default=defaults.get("theta_ga", DEFAULT_EXPERIMENT_VALUES["theta_ga"]))
    ga_group.add_argument("--mu", type=float, default=defaults.get("mu", DEFAULT_EXPERIMENT_VALUES["mu"]))
    ga_group.add_argument("--chi", type=float, default=defaults.get("chi", DEFAULT_EXPERIMENT_VALUES["chi"]))
    ga_group.add_argument("--theta_as", type=int, default=defaults.get("theta_as", DEFAULT_EXPERIMENT_VALUES["theta_as"]))
    ga_group.add_argument("--theta_exp", type=int, default=defaults.get("theta_exp", DEFAULT_EXPERIMENT_VALUES["theta_exp"]))

    phase_group = parser.add_argument_group("Phase Parameters")
    for phase in ("explore", "exploit1", "exploit2"):
        phase_group.add_argument(f"--{phase}_episodes", type=int, default=defaults.get(f"{phase}_episodes", DEFAULT_EXPERIMENT_VALUES[f"{phase}_episodes"]))
        phase_group.add_argument(f"--{phase}_epsilon", type=float, default=defaults.get(f"{phase}_epsilon", DEFAULT_EXPERIMENT_VALUES[f"{phase}_epsilon"]))
        phase_group.add_argument(f"--{phase}_beta", type=float, default=defaults.get(f"{phase}_beta", DEFAULT_EXPERIMENT_VALUES[f"{phase}_beta"]))
        phase_group.add_argument(f"--{phase}_alp", type=parse_boolGPU4, default=defaults.get(f"{phase}_alp", DEFAULT_EXPERIMENT_VALUES[f"{phase}_alp"]))
        phase_group.add_argument(f"--{phase}_ga", type=parse_boolGPU4, default=defaults.get(f"{phase}_ga", DEFAULT_EXPERIMENT_VALUES[f"{phase}_ga"]))
        phase_group.add_argument(f"--{phase}_decay", type=parse_boolGPU4, default=defaults.get(f"{phase}_decay", DEFAULT_EXPERIMENT_VALUES[f"{phase}_decay"]))

    plot_group = parser.add_argument_group("Plotting Parameters (for --load_dashboard_data)")
    plot_group.add_argument("--plot_steps", action="store_true")
    plot_group.add_argument("--plot_population", action="store_true")
    plot_group.add_argument("--plot_knowledge", action="store_true")
    plot_group.add_argument("--plot_reward_quality", action="store_true")
    plot_group.add_argument("--plot_policy_map", action="store_true")
    plot_group.add_argument("--plot_top_rules", action="store_true")
    plot_group.add_argument("--plot_origin_distribution", action="store_true")
    plot_group.add_argument("--plot_origin_distribution_abs", action="store_true")
    plot_group.add_argument("--plot_creation_dist", type=str)
    plot_group.add_argument("--plot_all_dashboards", action="store_true")
    return parser


def experiment_config_from_argsGPU4(args: argparse.Namespace) -> ExperimentConfigGPU4:
    env_args = {
        "type": args.environment_type,
        "name": args.environment_name,
        "reset_to_random_start": args.reset_to_random_start,
        "address_bits": args.address_bits,
        "problem_kind": args.problem_kind,
        "input_bits": args.input_bits,
        "left_bits": args.left_bits,
        "right_bits": args.right_bits,
        "sampling": args.sampling,
        "env_id": args.env_id,
        "observation_encoding": args.observation_encoding,
        "bins": args.bins,
        "is_slippery": args.is_slippery,
    }
    if args.environment_type != "grid_maze":
        env_args.update(
            {
                "rows": args.rows,
                "cols": args.cols,
                "start_pos": args.start_pos,
                "goal_pos": args.goal_pos,
                "obstacles": _normalize_obstacle_cli(args.obstacles),
            }
        )
    environment = environment_spec_from_args(env_args, fallback_name=args.environment_type)
    phases = {
        phase_name: PhaseConfigGPU4(
            episodes=getattr(args, f"{phase_name}_episodes"),
            epsilon=getattr(args, f"{phase_name}_epsilon"),
            beta=getattr(args, f"{phase_name}_beta"),
            alp=getattr(args, f"{phase_name}_alp"),
            ga=getattr(args, f"{phase_name}_ga"),
            decay=getattr(args, f"{phase_name}_decay"),
        )
        for phase_name in ("explore", "exploit1", "exploit2")
    }
    return ExperimentConfigGPU4(
        n_exp=args.n_exp,
        seed=args.seed,
        n_steps=args.n_steps,
        beta=args.beta,
        gamma=args.gamma,
        theta_i=args.theta_i,
        theta_r=args.theta_r,
        epsilon=args.epsilon,
        u_max=args.u_max,
        theta_ga=args.theta_ga,
        mu=args.mu,
        chi=args.chi,
        theta_as=args.theta_as,
        theta_exp=args.theta_exp,
        alp_mark_only_incorrect=args.alp_mark_only_incorrect,
        no_subsumption=args.no_subsumption,
        metric_calculation_frequency=args.metric_calculation_frequency,
        max_population=args.max_population,
        environment=environment,
        phases=phases,
        device=args.device,
        steps_to_goal_threshold=args.steps_to_goal_threshold,
    ).validate()


def _normalize_obstacle_cli(value: Any) -> list[list[int]]:
    if value in (None, (), []):
        return []
    if value and isinstance(value[0], (list, tuple)):
        return [list(map(int, item)) for item in value]
    flat = [int(item) for item in value]
    if len(flat) % 2 != 0:
        raise ValueError("obstacles provided on CLI must contain an even number of integers")
    return [[flat[idx], flat[idx + 1]] for idx in range(0, len(flat), 2)]
