from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from src.configGPU4 import ExperimentConfigGPU4


def export_dashboard_dataGPU4(stats: Dict[str, Any], summary_stats: Dict[str, float], timestamp: str, experiment_config: ExperimentConfigGPU4, optimal_avg_steps: float, title_prefix: str = "") -> None:
    output_dir = os.path.join("reports", "saved_states")
    os.makedirs(output_dir, exist_ok=True)
    prefix = title_prefix.strip().replace(" ", "_") + "_" if title_prefix else ""
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            serializable = value
            if serializable.ndim == 3:
                serializable = serializable.reshape(serializable.shape[0], -1)
            elif serializable.ndim == 1:
                serializable = serializable.reshape(-1, 1)
            pd.DataFrame(serializable).to_csv(os.path.join(output_dir, f"dashboard_data_{prefix}{timestamp}_{key}.csv"), index=False)
    metadata = {
        "timestamp": timestamp,
        "title_prefix": title_prefix,
        "params_phases": experiment_config.params_phases,
        "n_exp": experiment_config.n_exp,
        "n_steps": experiment_config.n_steps,
        "optimal_avg_steps": optimal_avg_steps,
        "summary_stats": summary_stats,
        "total_episodes": experiment_config.total_episodes,
        "environment": experiment_config.environment.to_metadata(),
        "implementation": "GPU4",
    }
    with open(os.path.join(output_dir, f"dashboard_metadata_{prefix}{timestamp}.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, default_flow_style=False, sort_keys=False)


def import_dashboard_dataGPU4(timestamp: str, title_prefix: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    input_dir = os.path.join("reports", "saved_states")
    prefix = title_prefix.strip().replace(" ", "_") + "_" if title_prefix else ""
    metadata = {}
    for candidate in (
        os.path.join(input_dir, f"dashboard_metadata_{prefix}{timestamp}.yaml"),
        os.path.join(input_dir, f"dashboard_metadata_{timestamp}.yaml"),
    ):
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}
            break
    if not metadata:
        return {}, {}
    resolved_prefix = metadata.get("title_prefix", "").strip().replace(" ", "_") + "_" if metadata.get("title_prefix") else prefix
    loaded_stats: Dict[str, Any] = {}
    loaded_stats.update(metadata.get("summary_stats", {}))
    for filename in os.listdir(input_dir):
        if not (filename.startswith(f"dashboard_data_{resolved_prefix}{timestamp}_") or filename.startswith(f"dashboard_data_{timestamp}_")):
            continue
        filepath = os.path.join(input_dir, filename)
        key = filename.replace(f"dashboard_data_{resolved_prefix}{timestamp}_", "").replace(f"dashboard_data_{timestamp}_", "").replace(".csv", "")
        loaded_stats[key] = pd.read_csv(filepath).values
    return loaded_stats, metadata
