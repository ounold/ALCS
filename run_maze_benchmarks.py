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
)
MODE_SPEC_MAP = {key: label for key, label in MODE_SPECS}


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


def run_mode(config_gpu4, explore_mode: str, exploit_mode: str) -> Dict[str, Any]:
    runner = UniversalRunner(
        config_gpu4,
        explore_mode,
        exploit_mode,
    )
    stats, summary, _, _, _ = runner.run()
    _, exploit_avg_std = exploit2_step_stats(stats, config_gpu4)
    return {
        "total_time_s": float(summary["Total Time"]),
        "avg_exp_time_s": float(summary["Avg Time"]),
        "std_exp_time_s": float(summary["Std Time"]),
        "exploit_avg_steps": float(summary["Exploit Avg. Steps"]),
        "exploit_avg_steps_std": exploit_avg_std,
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
    config = build_experiment_config(config_path, maze_name)
    object.__setattr__(config, "no_subsumption", no_subsumption)
    if mode_key == "cpu_single":
        return run_mode(config, "cpu_single", "cpu_single")
    if mode_key == "cpu_mp":
        return run_mode(config, "cpu_mp", "cpu_mp")
    if mode_key == "gpu":
        return run_mode(config, "gpu", "gpu")
    raise ValueError(f"Unsupported benchmark mode: {mode_key}")


def iter_maze_names(config_path: Path | None = None) -> Iterable[str]:
    all_names = sorted(load_acs2_maze_catalog().keys())
    if config_path is None or not config_path.exists():
        return all_names
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selected = raw.get("mazes")
    if not selected:
        return all_names
    selected_names = [str(name) for name in selected]
    unknown = [name for name in selected_names if name not in load_acs2_maze_catalog()]
    if unknown:
        raise ValueError(f"Unknown maze names in config: {unknown}")
    return selected_names


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
        help="Path to the YAML config used as the baseline for every maze run.",
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
    return parser.parse_args()


def run_benchmarks(config_path: Path, output_csv: Path, no_subsumption: bool | None = None) -> None:
    maze_names = list(iter_maze_names(config_path))
    mode_specs = tuple(iter_mode_specs(config_path))
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
    run_benchmarks(config_path, output_csv, no_subsumption=args.no_subsumption)


if __name__ == "__main__":
    main()
