from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional
from src.configCPU3 import PhaseConfigCPU3, ExperimentConfigCPU3
from src.configGPU4 import ExperimentConfigGPU4, PhaseConfigGPU4
from src.models.acs2.hybrid_transfer import gpu4_to_cpu3_agents

from src.models.acs2.confGPU4 import ACS2ConfigurationGPU4
from src.models.acs2.confCPU3 import ACS2ConfigurationCPU3

def build_agent_configGPU4(experiment_config: ExperimentConfigGPU4) -> ACS2ConfigurationGPU4:
    return ACS2ConfigurationGPU4(
        l_len=experiment_config.environment.state_length,
        num_actions=experiment_config.environment.num_actions,
        symbol_capacity=experiment_config.environment.symbol_capacity,
        total_episodes=experiment_config.total_episodes,
        beta=experiment_config.beta,
        gamma=experiment_config.gamma,
        theta_i=experiment_config.theta_i,
        theta_r=experiment_config.theta_r,
        epsilon=experiment_config.epsilon,
        u_max=experiment_config.u_max,
        theta_ga=experiment_config.theta_ga,
        mu=experiment_config.mu,
        chi=experiment_config.chi,
        theta_as=experiment_config.theta_as,
        theta_exp=experiment_config.theta_exp,
        do_subsumption=not experiment_config.no_subsumption,
        alp_mark_only_incorrect=experiment_config.alp_mark_only_incorrect,
        metric_calculation_frequency=experiment_config.metric_calculation_frequency,
        max_population=experiment_config.max_population,
    )

def build_agent_configCPU3(experiment_config: ExperimentConfigCPU3) -> ACS2ConfigurationCPU3:
    return ACS2ConfigurationCPU3(
        l_len=experiment_config.environment.state_length,
        num_actions=experiment_config.environment.num_actions,
        symbol_capacity=experiment_config.environment.symbol_capacity,
        total_episodes=experiment_config.total_episodes,
        beta=experiment_config.beta,
        gamma=experiment_config.gamma,
        theta_i=experiment_config.theta_i,
        theta_r=experiment_config.theta_r,
        epsilon=experiment_config.epsilon,
        u_max=experiment_config.u_max,
        theta_ga=experiment_config.theta_ga,
        mu=experiment_config.mu,
        chi=experiment_config.chi,
        theta_as=experiment_config.theta_as,
        theta_exp=experiment_config.theta_exp,
        do_subsumption=not experiment_config.no_subsumption,
        alp_mark_only_incorrect=experiment_config.alp_mark_only_incorrect,
        metric_calculation_frequency=experiment_config.metric_calculation_frequency,
    )

def gpu_config_to_cpu3(gpu_config: ExperimentConfigGPU4, active_phases: List[str]) -> ExperimentConfigCPU3:
    """
    Translates a GPU4 configuration into a CPU3 configuration for a specific set of phases.
    Episodes for phases not in 'active_phases' are set to 0.
    """
    cpu_phases = {}
    for name, phase in gpu_config.phases.items():
        episodes = phase.episodes if name in active_phases else 0
        cpu_phases[name] = PhaseConfigCPU3(
            episodes=episodes,
            epsilon=phase.epsilon,
            beta=phase.beta,
            alp=phase.alp,
            ga=phase.ga,
            decay=phase.decay,
        )
    return ExperimentConfigCPU3(
        n_exp=gpu_config.n_exp,
        seed=gpu_config.seed,
        n_steps=gpu_config.n_steps,
        beta=gpu_config.beta,
        gamma=gpu_config.gamma,
        theta_i=gpu_config.theta_i,
        theta_r=gpu_config.theta_r,
        epsilon=gpu_config.epsilon,
        u_max=gpu_config.u_max,
        theta_ga=gpu_config.theta_ga,
        mu=gpu_config.mu,
        chi=gpu_config.chi,
        theta_as=gpu_config.theta_as,
        theta_exp=gpu_config.theta_exp,
        alp_mark_only_incorrect=gpu_config.alp_mark_only_incorrect,
        no_subsumption=gpu_config.no_subsumption,
        metric_calculation_frequency=gpu_config.metric_calculation_frequency,
        environment=gpu_config.environment,
        phases=cpu_phases,
        steps_to_goal_threshold=gpu_config.steps_to_goal_threshold,
        global_total_episodes=gpu_config.total_episodes,
    ).validate()

def merge_experiment_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Robustly merges multiple experiment stat dictionaries by concatenating episode series
    and taking the final distribution matrices.
    """
    if not stats_list:
        return {}
    if len(stats_list) == 1:
        return stats_list[0]

    merged_stats = {}
    series_keys = [
        "steps", "micro", "rmicro", "macro", "rmacro", "know", "avg_r", "avg_rel_r", 
        "avg_q_all", "avg_q_rel", "avg_fit_all", "avg_fit_rel", "generalization",
        "covering_perc", "ga_perc", "alp_unexpected_perc", "alp_expected_perc", "alp_covering_perc",
        "covering_abs", "ga_abs", "alp_unexpected_abs", "alp_expected_abs", "alp_covering_abs"
    ]
    
    # Concatenate episode series (prefixed with 'stats_')
    for key in series_keys:
        s_key = f"stats_{key}"
        parts = []
        for s in stats_list:
            if s_key in s:
                parts.append(s[s_key])
        if parts:
            merged_stats[s_key] = np.concatenate(parts, axis=1)
            merged_stats[f"mean_{key}"] = np.mean(merged_stats[s_key], axis=0)

    # Special handling for knowledge std
    if "stats_know" in merged_stats:
        merged_stats["std_know"] = np.std(merged_stats["stats_know"], axis=0)
    
    # Dashboard Aliases
    if "mean_steps" in merged_stats:
        merged_stats["steps"] = merged_stats["mean_steps"]
    if "mean_micro" in merged_stats:
        merged_stats["mean_micro_pop"] = merged_stats["mean_micro"]
    if "mean_rmicro" in merged_stats:
        merged_stats["mean_rel_micro_pop"] = merged_stats["mean_rmicro"]
    if "mean_macro" in merged_stats:
        merged_stats["mean_macro_pop"] = merged_stats["mean_macro"]
    if "mean_rmacro" in merged_stats:
        merged_stats["mean_rel_macro_pop"] = merged_stats["mean_rmacro"]

    # Merge creation distributions (Take the last available matrix)
    creation_keys = ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]
    for origin in creation_keys:
        abs_key = f"stats_{origin}_creation_dist_abs"
        dist_key = f"stats_{origin}_creation_dist"
        
        last_s = None
        for s in reversed(stats_list):
            if abs_key in s:
                last_s = s
                break
        
        if last_s:
            merged_stats[abs_key] = last_s[abs_key]
            merged_stats[dist_key] = last_s[dist_key]
            merged_stats[f"mean_{origin}_creation_dist_abs"] = np.mean(last_s[abs_key], axis=0)
            merged_stats[f"mean_{origin}_creation_dist"] = np.mean(last_s[dist_key], axis=0)
            # Aliases
            merged_stats[f"{origin}_creation_dist_abs"] = merged_stats[f"mean_{origin}_creation_dist_abs"]
            merged_stats[f"{origin}_creation_dist"] = merged_stats[f"mean_{origin}_creation_dist"]

    return merged_stats

def calculate_hybrid_summary(merged_stats: Dict[str, Any], explore_episodes: int, exploit1_episodes: int) -> Dict[str, Any]:
    """
    Calculates summary statistics based on the combined history.
    """
    summary = {}
    if "stats_steps" in merged_stats:
        exploit2_start = int(explore_episodes) + int(exploit1_episodes)
        exploit2_slice = merged_stats["stats_steps"][:, exploit2_start:]
        if exploit2_slice.size == 0:
            exploit2_slice = merged_stats["stats_steps"]
        summary["Exploit Avg. Steps"] = float(np.mean(exploit2_slice))
    if "stats_know" in merged_stats:
        summary["Knowledge"] = float(np.mean(merged_stats["stats_know"][:, -1]))
    if "stats_generalization" in merged_stats:
        summary["Generalization"] = float(np.mean(merged_stats["stats_generalization"][:, -1]))
    if "stats_avg_r" in merged_stats:
        summary["Rew (All)"] = float(np.mean(merged_stats["stats_avg_r"][:, -1]))
    if "stats_avg_rel_r" in merged_stats:
        summary["Rew (Rel)"] = float(np.mean(merged_stats["stats_avg_rel_r"][:, -1]))
    if "stats_micro" in merged_stats:
        summary["Micro"] = float(np.mean(merged_stats["stats_micro"][:, -1]))
    if "stats_rmicro" in merged_stats:
        summary["Micro (Rel)"] = float(np.mean(merged_stats["stats_rmicro"][:, -1]))
    if "stats_macro" in merged_stats:
        summary["Macro"] = float(np.mean(merged_stats["stats_macro"][:, -1]))
    if "stats_rmacro" in merged_stats:
        summary["Macro (Rel)"] = float(np.mean(merged_stats["stats_rmacro"][:, -1]))
    return summary
