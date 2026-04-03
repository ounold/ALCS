from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from src.configCPU3 import ExperimentConfigCPU3


def export_dashboard_dataCPU3(stats: Dict[str, Any], summary_stats: Dict[str, Any], timestamp: str, experiment_config: Any, optimal_avg_steps: float, title_prefix: str = "") -> None:
    output_dir = os.path.join("reports", "saved_states")
    os.makedirs(output_dir, exist_ok=True)
    prefix = title_prefix.strip().replace(" ", "_") + "_" if title_prefix else ""
    
    # We only want to save RAW experiment series (usually prefixed with stats_)
    # We skip calculated means and aliases to save space and avoid clutter.
    save_keys = [k for key in stats.keys() if (k := str(key)).startswith("stats_") or k == "best_agent"]

    for key in save_keys:
        value = stats[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                # Save 3D arrays as .npy or .npz to preserve exact shape (e.g. creation_dist)
                filename = os.path.join(output_dir, f"dashboard_data_{prefix}{timestamp}_{key}.npy")
                np.save(filename, value)
            else:
                filename = os.path.join(output_dir, f"dashboard_data_{prefix}{timestamp}_{key}.csv")
                serializable = value
                if serializable.ndim == 1:
                    serializable = serializable.reshape(-1, 1)
                pd.DataFrame(serializable).to_csv(filename, index=False)
        elif key == "best_agent":
            filename = os.path.join(output_dir, f"dashboard_data_{prefix}{timestamp}_best_agent.pkl")
            with open(filename, "wb") as handle:
                pickle.dump(value, handle)
                
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
        "implementation": "Unified/CPU3",
    }
    # Optional: capture execution modes if available
    for attr in ["explore_mode", "exploit_mode"]:
        if hasattr(experiment_config, attr):
            metadata[attr] = getattr(experiment_config, attr)

    metadata_path = os.path.join(output_dir, f"dashboard_metadata_{prefix}{timestamp}.yaml")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, default_flow_style=False, sort_keys=False)


def import_dashboard_dataCPU3(timestamp: str, title_prefix: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    input_dir = os.path.join("reports", "saved_states")
    prefix = title_prefix.strip().replace(" ", "_") + "_" if title_prefix else ""
    metadata = {}
    
    # Try finding metadata with or without prefix
    search_patterns = [
        f"dashboard_metadata_{prefix}{timestamp}.yaml",
        f"dashboard_metadata_{timestamp}.yaml"
    ]
    for pattern in search_patterns:
        path = os.path.join(input_dir, pattern)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
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
        # Extract key: dashboard_data_PREFIX_TIMESTAMP_KEY.ext
        base_name = filename.rsplit(".", 1)[0]
        key = base_name.split(f"{timestamp}_")[-1]
        
        if filename.endswith(".csv"):
            loaded_stats[key] = pd.read_csv(filepath).values
        elif filename.endswith(".npy"):
            loaded_stats[key] = np.load(filepath)
        elif filename.endswith(".pkl"):
            with open(filepath, "rb") as handle:
                loaded_stats[key] = pickle.load(handle)
                
    return loaded_stats, metadata
