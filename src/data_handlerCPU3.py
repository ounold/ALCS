from __future__ import annotations

import os
import pickle
import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from src.configCPU3 import ExperimentConfigCPU3


def _safe_prefixCPU3(title_prefix: str) -> str:
    if not title_prefix:
        return ""
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", title_prefix.strip())
    cleaned = cleaned.replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return f"{cleaned}_" if cleaned else ""


def _rebuild_dashboard_seriesCPU3(loaded_stats: Dict[str, Any]) -> None:
    if "stats_steps" in loaded_stats and "mean_steps" not in loaded_stats:
        loaded_stats["mean_steps"] = np.mean(np.asarray(loaded_stats["stats_steps"]), axis=0)

    mean_aliases = [
        "micro", "rmicro", "macro", "rmacro", "know", "avg_r", "avg_rel_r",
        "avg_q_all", "avg_q_rel", "avg_fit_all", "avg_fit_rel", "generalization",
        "covering_perc", "ga_perc", "alp_unexpected_perc", "alp_expected_perc", "alp_covering_perc",
        "covering_abs", "ga_abs", "alp_unexpected_abs", "alp_expected_abs", "alp_covering_abs",
    ]
    for name in mean_aliases:
        stats_key = f"stats_{name}"
        mean_key = f"mean_{name}"
        if stats_key in loaded_stats and mean_key not in loaded_stats:
            loaded_stats[mean_key] = np.mean(np.asarray(loaded_stats[stats_key]), axis=0)

    if "stats_know" in loaded_stats and "std_know" not in loaded_stats:
        loaded_stats["std_know"] = np.std(np.asarray(loaded_stats["stats_know"]), axis=0)

    for origin in ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]:
        dist_key = f"stats_{origin}_creation_dist"
        abs_key = f"stats_{origin}_creation_dist_abs"
        if dist_key in loaded_stats and f"mean_{origin}_creation_dist" not in loaded_stats:
            loaded_stats[f"mean_{origin}_creation_dist"] = np.mean(np.asarray(loaded_stats[dist_key]), axis=0)
            loaded_stats[f"{origin}_creation_dist"] = loaded_stats[f"mean_{origin}_creation_dist"]
        if abs_key in loaded_stats and f"mean_{origin}_creation_dist_abs" not in loaded_stats:
            loaded_stats[f"mean_{origin}_creation_dist_abs"] = np.mean(np.asarray(loaded_stats[abs_key]), axis=0)
            loaded_stats[f"{origin}_creation_dist_abs"] = loaded_stats[f"mean_{origin}_creation_dist_abs"]

    if "mean_micro" in loaded_stats and "mean_micro_pop" not in loaded_stats:
        loaded_stats["mean_micro_pop"] = loaded_stats["mean_micro"]
    if "mean_rmicro" in loaded_stats and "mean_rel_micro_pop" not in loaded_stats:
        loaded_stats["mean_rel_micro_pop"] = loaded_stats["mean_rmicro"]
    if "mean_macro" in loaded_stats and "mean_macro_pop" not in loaded_stats:
        loaded_stats["mean_macro_pop"] = loaded_stats["mean_macro"]
    if "mean_rmacro" in loaded_stats and "mean_rel_macro_pop" not in loaded_stats:
        loaded_stats["mean_rel_macro_pop"] = loaded_stats["mean_rmacro"]


def export_dashboard_dataCPU3(stats: Dict[str, Any], summary_stats: Dict[str, Any], timestamp: str, experiment_config: Any, optimal_avg_steps: float, title_prefix: str = "") -> None:
    output_dir = os.path.join("reports", "saved_states")
    os.makedirs(output_dir, exist_ok=True)
    prefix = _safe_prefixCPU3(title_prefix)
    
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
    prefix = _safe_prefixCPU3(title_prefix)
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

    if not metadata and os.path.isdir(input_dir):
        for filename in os.listdir(input_dir):
            if filename.startswith("dashboard_metadata_") and filename.endswith(f"{timestamp}.yaml"):
                path = os.path.join(input_dir, filename)
                with open(path, "r", encoding="utf-8") as handle:
                    metadata = yaml.safe_load(handle) or {}
                break
            
    if not metadata:
        return {}, {}

    resolved_prefix = _safe_prefixCPU3(metadata.get("title_prefix", "")) if metadata.get("title_prefix") else prefix
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

    _rebuild_dashboard_seriesCPU3(loaded_stats)
    return loaded_stats, metadata
