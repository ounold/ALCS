from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import yaml

from environment.maze_loader import load_acs2_maze_catalog
from src.configGPU4 import (
    build_arg_parserGPU4,
    experiment_config_from_argsGPU4,
    load_yaml_defaultsGPU4,
)
from src.universal_runner import UniversalRunner


MODE_SPECS: Tuple[Tuple[str, str], ...] = (
    ("cpu_single", "CPU Single"),
    ("cpu_mp", "CPU MP"),
    ("gpu", "GPU"),
    ("gpu_seq", "GPU Seq"),
    ("gpu_cpu", "GPU (PyTorch CPU)"),
    ("gpu_seq_cpu", "GPU Seq (PyTorch CPU)"),
    ("gpu_cuda", "GPU CUDA"),
    ("gpu_seq_cuda", "GPU Seq CUDA"),
)
MODE_SPEC_MAP = {key: label for key, label in MODE_SPECS}
AVAILABLE_MAZES = load_acs2_maze_catalog()


def build_experiment_config(config_path: Path, maze_name: str):
    defaults = load_yaml_defaultsGPU4(str(config_path))
    parser = build_arg_parserGPU4(defaults)
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--environment_name",
            maze_name,
        ]
    )
    return experiment_config_from_argsGPU4(args)


def build_experiment_config_with_device(config_path: Path, maze_name: str, device_override: str | None = None):
    config = build_experiment_config(config_path, maze_name)
    if device_override is not None:
        object.__setattr__(config, "device", device_override)
    return config



def iter_mode_specs(config_path: Path | None = None) -> Iterable[Tuple[str, str]]:
    if config_path is None or not config_path.exists():
        return MODE_SPECS
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selected = raw.get("modes")
    if not selected:
        return MODE_SPECS
    selected_keys = [str(mode) for mode in selected]
    unknown = [mode for mode in selected_keys if mode not in MODE_SPEC_MAP]
    if unknown:
        raise ValueError(f"Unknown mode names in config: {unknown}")
    return tuple((key, MODE_SPEC_MAP[key]) for key in selected_keys)


def exploit2_step_stats(stats: Dict[str, Any], config_gpu4) -> Tuple[float, float]:
    steps = np.asarray(stats["stats_steps"], dtype=np.float64)
    exploit2_start = config_gpu4.phases["explore"].episodes + config_gpu4.phases["exploit1"].episodes
    exploit2_steps = steps[:, exploit2_start:]
    if exploit2_steps.size == 0:
        exploit2_steps = steps
    return float(np.mean(exploit2_steps)), float(np.std(exploit2_steps))


def exploit2_population_stats(stats: Dict[str, Any], config_gpu4) -> Dict[str, float]:
    exploit2_start = config_gpu4.phases["explore"].episodes + config_gpu4.phases["exploit1"].episodes

    def _phase_mean(key: str) -> float:
        values = np.asarray(stats.get(key, np.array([])), dtype=np.float64)
        if values.size == 0:
            return 0.0
        exploit2_values = values[:, exploit2_start:] if values.ndim == 2 else np.array([])
        if exploit2_values.size == 0:
            exploit2_values = values
        return float(np.mean(exploit2_values))

    return {
        "micro_pop_exploit2_avg": _phase_mean("stats_micro"),
        "macro_pop_exploit2_avg": _phase_mean("stats_macro"),
        "micro_pop_rel_exploit2_avg": _phase_mean("stats_rmicro"),
        "macro_pop_rel_exploit2_avg": _phase_mean("stats_rmacro"),
    }


def run_mode(config_gpu4, explore_mode: str, exploit_mode: str) -> Dict[str, Any]:
    runner = UniversalRunner(
        config_gpu4,
        explore_mode,
        exploit_mode,
    )
    stats, summary, _, _, _ = runner.run()
    _, exploit_avg_std = exploit2_step_stats(stats, config_gpu4)
    population_stats = exploit2_population_stats(stats, config_gpu4)
    return {
        "total_time_s": float(summary["Total Time"]),
        "avg_exp_time_s": float(summary["Avg Time"]),
        "std_exp_time_s": float(summary["Std Time"]),
        "exploit_avg_steps": float(summary["Exploit Avg. Steps"]),
        "exploit_avg_steps_std": exploit_avg_std,
        **population_stats,
        "gpu_apply_learning_s": float(summary.get("Timing Breakdown", {}).get("apply_learning_s", 0.0)),
        "gpu_batch_add_s": float(summary.get("Timing Breakdown", {}).get("batch_add_classifiers_s", 0.0)),
        "gpu_subsumption_s": float(summary.get("Timing Breakdown", {}).get("subsumption_s", 0.0)),
        "gpu_reserve_slots_s": float(summary.get("Timing Breakdown", {}).get("reserve_slots_s", 0.0)),
        "gpu_metrics_s": float(summary.get("Timing Breakdown", {}).get("metrics_s", 0.0)),
    }


def benchmark_one_mode(
    config_path: Path,
    maze_name: str,
    mode_key: str,
    no_subsumption: bool = False,
) -> Dict[str, Any]:
    device_override = None
    backend_mode = mode_key
    if mode_key == "gpu_cpu":
        backend_mode = "gpu"
        device_override = "cpu"
    elif mode_key == "gpu_seq_cpu":
        backend_mode = "gpu_seq"
        device_override = "cpu"
    elif mode_key == "gpu_cuda":
        backend_mode = "gpu"
        device_override = "cuda"
    elif mode_key == "gpu_seq_cuda":
        backend_mode = "gpu_seq"
        device_override = "cuda"

    config = build_experiment_config_with_device(config_path, maze_name, device_override=device_override)
    object.__setattr__(config, "no_subsumption", no_subsumption)
    if backend_mode == "cpu_single":
        return run_mode(config, "cpu_single", "cpu_single")
    if backend_mode == "cpu_mp":
        return run_mode(config, "cpu_mp", "cpu_mp")
    if backend_mode == "gpu":
        return run_mode(config, "gpu", "gpu")
    if backend_mode == "gpu_seq":
        return run_mode(config, "gpu_seq", "gpu_seq")
    raise ValueError(f"Unsupported benchmark mode: {mode_key}")


def iter_maze_names(config_path: Path | None = None) -> Iterable[str]:
    all_names = sorted(AVAILABLE_MAZES.keys())
    if config_path is None or not config_path.exists():
        return all_names
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selected = raw.get("mazes")
    if not selected:
        return all_names
    selected_names = [str(name) for name in selected]
    unknown = [name for name in selected_names if name not in AVAILABLE_MAZES]
    if unknown:
        raise ValueError(f"Unknown maze names in config: {unknown}")
    return selected_names


def select_maze_names(config_path: Path, maze_name: str | None = None) -> list[str]:
    maze_names = list(iter_maze_names(config_path))
    if maze_name is None:
        return maze_names
    if maze_name not in AVAILABLE_MAZES:
        raise ValueError(f"Unknown maze name: {maze_name}")
    if maze_name not in maze_names:
        raise ValueError(f"Maze {maze_name!r} is not enabled by config {config_path}")
    return [maze_name]


def select_mode_specs(config_path: Path, mode_key: str | None = None) -> tuple[Tuple[str, str], ...]:
    mode_specs = tuple(iter_mode_specs(config_path))
    if mode_key is None:
        return mode_specs
    if mode_key not in MODE_SPEC_MAP:
        raise ValueError(f"Unknown mode name: {mode_key}")
    for selected_key, selected_label in mode_specs:
        if selected_key == mode_key:
            return ((selected_key, selected_label),)
    raise ValueError(f"Mode {mode_key!r} is not enabled by config {config_path}")


def create_output_path(output: str | None) -> Path:
    if output:
        return Path(output)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return reports_dir / f"maze_benchmarks_{timestamp}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch benchmark all ACS2 mazes in CPU and GPU modes.")
    parser.add_argument(
        "--config",
        default="experiments/configs/batch_mazes.yaml",
        help="Path to the YAML config used as the baseline for every maze run, including filtered single-maze runs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to reports/maze_benchmarks_<timestamp>.csv",
    )
    parser.add_argument(
        "--no_subsumption",
        action="store_true",
        default=None,
        help="Disable subsumption in all benchmarked backends. If omitted, keep the YAML setting.",
    )
    parser.add_argument(
        "--maze",
        default=None,
        help="Run only one maze, for example --maze Woods102.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_SPEC_MAP.keys()),
        default=None,
        help="Run only one backend mode, for example --mode gpu.",
    )
    return parser.parse_args()


def run_benchmarks(
    config_path: Path,
    output_csv: Path,
    no_subsumption: bool | None = None,
    maze_name: str | None = None,
    mode_key: str | None = None,
) -> None:
    maze_names = select_maze_names(config_path, maze_name)
    mode_specs = select_mode_specs(config_path, mode_key)
    total_runs = len(maze_names) * len(mode_specs)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    effective_no_subsumption = bool(raw_config.get("no_subsumption", False)) if no_subsumption is None else bool(no_subsumption)

    header = [
        "maze",
        "mode",
        "no_subsumption",
        "total_time_s",
        "avg_exp_time_s",
        "std_exp_time_s",
        "exploit_avg_steps",
        "exploit_avg_steps_std",
        "micro_pop_exploit2_avg",
        "macro_pop_exploit2_avg",
        "micro_pop_rel_exploit2_avg",
        "macro_pop_rel_exploit2_avg",
        "gpu_apply_learning_s",
        "gpu_batch_add_s",
        "gpu_subsumption_s",
        "gpu_reserve_slots_s",
        "gpu_metrics_s",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = 0

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        for maze_name in maze_names:
            for mode_key, mode_label in mode_specs:
                completed += 1
                print(
                    f"[{completed}/{total_runs}] Running maze={maze_name} mode={mode_label} "
                    f"no_subsumption={no_subsumption}"
                )
                try:
                    metrics = benchmark_one_mode(
                        config_path,
                        maze_name,
                        mode_key,
                        no_subsumption=effective_no_subsumption,
                    )
                except Exception as exc:
                    print(f"[{completed}/{total_runs}] FAILED maze={maze_name} mode={mode_label}: {exc}")
                    continue

                row = {
                    "maze": maze_name,
                    "mode": mode_label,
                    "no_subsumption": effective_no_subsumption,
                    **metrics,
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"[{completed}/{total_runs}] Finished maze={maze_name} mode={mode_label} "
                    f"no_subsumption={effective_no_subsumption} "
                    f"Exploit Avg.={metrics['exploit_avg_steps']:.4f} +/- {metrics['exploit_avg_steps_std']:.4f} "
                    f"Total Time={metrics['total_time_s']:.2f}s"
                )

    print(f"Benchmark completed. Results saved to {output_csv}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_csv = create_output_path(args.output)
    run_benchmarks(
        config_path,
        output_csv,
        no_subsumption=args.no_subsumption,
        maze_name=args.maze,
        mode_key=args.mode,
    )


if __name__ == "__main__":
    main()
