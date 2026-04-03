from __future__ import annotations

import copy
import math
import multiprocessing
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.configCPU3 import ExperimentConfigCPU3
from environment.runtime_cpu3 import EnvironmentCPU3, create_environmentCPU3
from src.metricsCPU3 import calculate_metricsCPU3, calculate_optimal_metricsCPU3, CREATION_KEYS_CPU3
from src.models.acs2.acs2CPU3 import ACS2CPU3
from src.models.acs2.confCPU3 import ACS2ConfigurationCPU3


@dataclass
class WorkerInputCPU3:
    experiment_config: ExperimentConfigCPU3
    agent_cfg: ACS2ConfigurationCPU3
    exp_idx: int
    initial_agents: Optional[List[ACS2CPU3]] = None
    episode_offset: int = 0


def _seed_cpu_worker(seed: int) -> None:
    bounded_seed = int(seed) % (2**32)
    random.seed(bounded_seed)
    np.random.seed(bounded_seed)


def _observation_trigger(total_episodes: int, curr_ep_idx: int) -> Tuple[bool, int]:
    observation_points = np.linspace(0, total_episodes - 1, 10, dtype=int)
    if curr_ep_idx in observation_points:
        return True, np.where(observation_points == curr_ep_idx)[0][0]
    return False, -1


def _empty_episode_series(total_episodes: int) -> Dict[str, np.ndarray]:
    return {
        "steps": np.zeros(total_episodes),
        "micro": np.zeros(total_episodes),
        "rmicro": np.zeros(total_episodes),
        "macro": np.zeros(total_episodes),
        "rmacro": np.zeros(total_episodes),
        "know": np.zeros(total_episodes),
        "avg_r": np.zeros(total_episodes),
        "avg_rel_r": np.zeros(total_episodes),
        "avg_q_all": np.zeros(total_episodes),
        "avg_q_rel": np.zeros(total_episodes),
        "avg_fit_all": np.zeros(total_episodes),
        "avg_fit_rel": np.zeros(total_episodes),
        "generalization": np.zeros(total_episodes),
        "covering_perc": np.zeros(total_episodes),
        "ga_perc": np.zeros(total_episodes),
        "alp_unexpected_perc": np.zeros(total_episodes),
        "alp_expected_perc": np.zeros(total_episodes),
        "alp_covering_perc": np.zeros(total_episodes),
        "covering_abs": np.zeros(total_episodes),
        "ga_abs": np.zeros(total_episodes),
        "alp_unexpected_abs": np.zeros(total_episodes),
        "alp_expected_abs": np.zeros(total_episodes),
        "alp_covering_abs": np.zeros(total_episodes),
    }


def _run_single_experimentCPU3(worker_input: WorkerInputCPU3) -> Dict[str, Any]:
    experiment_config = worker_input.experiment_config
    agent_cfg = worker_input.agent_cfg
    worker_seed = experiment_config.seed + worker_input.exp_idx + (worker_input.episode_offset * 1000003)
    _seed_cpu_worker(worker_seed)
    env = create_environmentCPU3(experiment_config.environment)
    
    if worker_input.initial_agents and len(worker_input.initial_agents) > worker_input.exp_idx:
        agent = worker_input.initial_agents[worker_input.exp_idx]
        agent.cfg = agent_cfg # Update config just in case
    else:
        agent = ACS2CPU3(agent_cfg)

    from src.metricsCPU3 import build_metric_contextCPU3, calculate_metricsCPU3, calculate_origin_distributionCPU3
    metric_context = build_metric_contextCPU3(agent, env)

    total_episodes = sum(p.episodes for p in experiment_config.phases.values())
    local_stats = _empty_episode_series(total_episodes)
    local_creation_perc = {key: np.zeros((10, 10), dtype=np.float64) for key in CREATION_KEYS_CPU3}
    local_creation_abs = {key: np.zeros((10, 10), dtype=np.float64) for key in CREATION_KEYS_CPU3}
    metric_cache = {
        "micro": 0.0,
        "rmicro": 0.0,
        "macro": 0.0,
        "rmacro": 0.0,
        "know": 0.0,
        "avg_r": 0.0,
        "avg_rel_r": 0.0,
        "avg_q_all": 0.0,
        "avg_q_rel": 0.0,
        "avg_fit_all": 0.0,
        "avg_fit_rel": 0.0,
        "generalization": 0.0,
    }
    origin_perc_cache = {key: 0.0 for key in CREATION_KEYS_CPU3}
    origin_abs_cache = {key: 0.0 for key in CREATION_KEYS_CPU3}
    
    curr_ep_idx = 0
    start_time = time.time()
    
    for phase_name, phase_cfg in experiment_config.phases.items():
        if phase_cfg.episodes <= 0:
            continue
            
        agent.epsilon = phase_cfg.epsilon
        agent.beta = phase_cfg.beta
        agent.cfg.do_alp = phase_cfg.alp
        agent.cfg.do_ga = phase_cfg.ga
        agent.cfg.do_decay_epsilon = phase_cfg.decay

        for _ in range(phase_cfg.episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < experiment_config.n_steps:
                action, action_set = agent.run_step(state)
                next_state, reward, done = env.step(action)
                agent.apply_learning(action_set, state, action, reward, next_state, done, curr_ep_idx=worker_input.episode_offset + curr_ep_idx)
                state = next_state
                steps += 1
            
            local_stats["steps"][curr_ep_idx] = steps
            
            # Metrics
            if (curr_ep_idx + 1) % experiment_config.metric_calculation_frequency == 0:
                perf = calculate_metricsCPU3(agent, metric_context)
                # 0: micro, 1: rel_micro, 2: macro, 3: rel_macro, 4: knowledge, 5: r_all, 6: r_rel, 7: q_all, 8: q_rel, 9: fit_all, 10: fit_rel, 11: gen
                metric_cache = {
                    "micro": perf[0],
                    "rmicro": perf[1],
                    "macro": perf[2],
                    "rmacro": perf[3],
                    "know": perf[4],
                    "avg_r": perf[5],
                    "avg_rel_r": perf[6],
                    "avg_q_all": perf[7],
                    "avg_q_rel": perf[8],
                    "avg_fit_all": perf[9],
                    "avg_fit_rel": perf[10],
                    "generalization": perf[11],
                }
                
                origin_perc, origin_counts = calculate_origin_distributionCPU3(agent)
                origin_perc_cache = {origin: origin_perc.get(origin, 0.0) for origin in CREATION_KEYS_CPU3}
                origin_abs_cache = {origin: origin_counts.get(origin, 0.0) for origin in CREATION_KEYS_CPU3}

            for key, value in metric_cache.items():
                local_stats[key][curr_ep_idx] = value
            for origin in CREATION_KEYS_CPU3:
                local_stats[f"{origin}_perc"][curr_ep_idx] = origin_perc_cache[origin]
                local_stats[f"{origin}_abs"][curr_ep_idx] = origin_abs_cache[origin]
            
            # Observation triggers for distributions
            should_capture, observation_index = _observation_trigger(total_episodes, curr_ep_idx)
            if should_capture:
                effective_total = experiment_config.global_total_episodes or total_episodes
                creation_bucket_size = max(1, math.ceil(effective_total / 10))
                for origin in CREATION_KEYS_CPU3:
                    classifiers = [classifier for classifier in agent.population if classifier.origin_source == origin]
                    total_count = sum(classifier.num for classifier in classifiers)
                    if total_count <= 0:
                        continue
                    bucket_counts = np.zeros(10, dtype=np.float64)
                    for classifier in classifiers:
                        creation_bucket = min(classifier.creation_episode // creation_bucket_size, 9)
                        bucket_counts[int(creation_bucket)] += classifier.num
                    local_creation_abs[origin][observation_index, :] = bucket_counts
                    local_creation_perc[origin][observation_index, :] = (bucket_counts / total_count) * 100.0
            
            curr_ep_idx += 1
            agent.curr_ep_idx = curr_ep_idx

    duration = time.time() - start_time
    return {
        "stats": local_stats,
        "creation_perc": local_creation_perc,
        "creation_abs": local_creation_abs,
        "best_agent": agent,
        "duration": duration,
        "last_env": env
    }


def run_experimentCPU3(
    experiment_config: ExperimentConfigCPU3,
    agent_cfg: Optional[ACS2ConfigurationCPU3] = None,
    no_mp: bool = False,
    processes: Optional[int] = None,
    initial_agents: Optional[List[ACS2CPU3]] = None,
    episode_offset: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, float], int, ACS2CPU3, float, EnvironmentCPU3]:
    if agent_cfg is None:
        from src.hybrid_utils import build_agent_configCPU3
        agent_cfg = build_agent_configCPU3(experiment_config)
        
    dummy_env = create_environmentCPU3(experiment_config.environment)
    optimal_avg_steps = calculate_optimal_metricsCPU3(dummy_env)
    total_episodes = sum(p.episodes for p in experiment_config.phases.values())
    idx_p2_start = experiment_config.phases["explore"].episodes
    idx_p3_start = idx_p2_start + experiment_config.phases["exploit1"].episodes
    
    stats: Dict[str, np.ndarray] = {
        f"stats_{key}": np.zeros((experiment_config.n_exp, total_episodes)) 
        for key in _empty_episode_series(total_episodes).keys()
    }
    
    # 3D Creation distributions
    for origin in CREATION_KEYS_CPU3:
        stats[f"stats_{origin}_creation_dist"] = np.zeros((experiment_config.n_exp, 10, 10))
        stats[f"stats_{origin}_creation_dist_abs"] = np.zeros((experiment_config.n_exp, 10, 10))

    worker_inputs = [
        WorkerInputCPU3(experiment_config, agent_cfg, i, initial_agents, episode_offset)
        for i in range(experiment_config.n_exp)
    ]

    results = []
    if no_mp:
        for wi in worker_inputs:
            results.append(_run_single_experimentCPU3(wi))
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(_run_single_experimentCPU3, worker_inputs)

    # Aggregate
    durations = []
    last_env = None
    best_agent = None
    best_score = float("inf")
    all_agents: List[ACS2CPU3] = []
    for i, res in enumerate(results):
        durations.append(res["duration"])
        last_env = res["last_env"]
        all_agents.append(res["best_agent"])
        for key in res["stats"]:
            stats[f"stats_{key}"][i, :] = res["stats"][key]
        for origin in CREATION_KEYS_CPU3:
            stats[f"stats_{origin}_creation_dist"][i, :, :] = res["creation_perc"][origin]
            stats[f"stats_{origin}_creation_dist_abs"][i, :, :] = res["creation_abs"][origin]
        exploit_slice = res["stats"]["steps"][idx_p2_start:]
        if exploit_slice.size == 0:
            score = float(np.mean(res["stats"]["steps"])) if res["stats"]["steps"].size > 0 else float("inf")
        else:
            score = float(np.mean(exploit_slice))
        if score < best_score:
            best_score = score
            best_agent = res["best_agent"]

    stats["all_agents"] = all_agents

    # Safely calculate summary statistics
    def safe_mean(data):
        return float(np.mean(data)) if data.size > 0 else 0.0

    exploit2_slice = stats["stats_steps"][:, idx_p3_start:] if idx_p3_start < total_episodes else stats["stats_steps"]

    summary_stats = {
        "Total Time": sum(durations),
        "Avg Time": np.mean(durations) if durations else 0.0,
        "Std Time": np.std(durations) if len(durations) > 1 else 0.0,
        "Exploit Avg. Steps": safe_mean(exploit2_slice),
        "Knowledge": safe_mean(stats["stats_know"][:, -1:]),
        "Generalization": safe_mean(stats["stats_generalization"][:, -1:]),
        "Rew (All)": safe_mean(stats["stats_avg_r"][:, -1:]),
        "Rew (Rel)": safe_mean(stats["stats_avg_rel_r"][:, -1:]),
        "Micro": safe_mean(stats["stats_micro"][:, -1:]),
        "Macro": safe_mean(stats["stats_macro"][:, -1:]),
        "Micro (Rel)": safe_mean(stats["stats_rmicro"][:, -1:]),
        "Macro (Rel)": safe_mean(stats["stats_rmacro"][:, -1:]),
        "ALP Marking": "Restricted (on incorrect)" if agent_cfg.alp_mark_only_incorrect else "Full Action Set",
        "Subsumption": "Disabled" if not agent_cfg.do_subsumption else "Enabled",
    }
    
    return stats, summary_stats, total_episodes, best_agent, optimal_avg_steps, last_env
