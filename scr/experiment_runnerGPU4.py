from __future__ import annotations

import time
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.configGPU4 import ExperimentConfigGPU4
from environment.runtime_gpu4 import EnvironmentGPU4, create_environmentGPU4
from src.metricsGPU4 import build_metric_contextGPU4, calculate_metricsGPU4, calculate_optimal_metricsGPU4, calculate_creation_distribution_snapshotGPU4
from src.models.acs2.acs2GPU4 import ACS2GPU4, AgentSelectionGPU4
from src.models.acs2.confGPU4 import ACS2ConfigurationGPU4


def _seed_gpu_run(seed: int) -> None:
    bounded_seed = int(seed) % (2**32)
    random.seed(bounded_seed)
    np.random.seed(bounded_seed)
    torch.manual_seed(bounded_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bounded_seed)


def resolve_deviceGPU4(device_name: str) -> str:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return "cpu"
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _empty_statsGPU4(n_exp: int, total_episodes: int, device: torch.device) -> Dict[str, torch.Tensor]:
    stats = {
        "stats_steps": torch.zeros((n_exp, total_episodes), device=device),
        "stats_micro": torch.zeros((n_exp, total_episodes), device=device),
        "stats_rmicro": torch.zeros((n_exp, total_episodes), device=device),
        "stats_macro": torch.zeros((n_exp, total_episodes), device=device),
        "stats_rmacro": torch.zeros((n_exp, total_episodes), device=device),
        "stats_know": torch.zeros((n_exp, total_episodes), device=device),
        "stats_avg_r": torch.zeros((n_exp, total_episodes), device=device),
        "stats_avg_rel_r": torch.zeros((n_exp, total_episodes), device=device),
        "stats_avg_q_all": torch.zeros((n_exp, total_episodes), device=device),
        "stats_avg_q_rel": torch.zeros((n_exp, total_episodes), device=device),
        "stats_avg_fit_all": torch.zeros((n_exp, total_episodes), device=device),
        "stats_avg_fit_rel": torch.zeros((n_exp, total_episodes), device=device),
        "stats_generalization": torch.zeros((n_exp, total_episodes), device=device),
        "stats_covering_perc": torch.zeros((n_exp, total_episodes), device=device),
        "stats_ga_perc": torch.zeros((n_exp, total_episodes), device=device),
        "stats_alp_unexpected_perc": torch.zeros((n_exp, total_episodes), device=device),
        "stats_alp_expected_perc": torch.zeros((n_exp, total_episodes), device=device),
        "stats_alp_covering_perc": torch.zeros((n_exp, total_episodes), device=device),
        "stats_covering_abs": torch.zeros((n_exp, total_episodes), device=device),
        "stats_ga_abs": torch.zeros((n_exp, total_episodes), device=device),
        "stats_alp_unexpected_abs": torch.zeros((n_exp, total_episodes), device=device),
        "stats_alp_expected_abs": torch.zeros((n_exp, total_episodes), device=device),
        "stats_alp_covering_abs": torch.zeros((n_exp, total_episodes), device=device),
    }
    for origin in ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]:
        stats[f"stats_{origin}_creation_dist"] = torch.zeros((n_exp, 10, 10), device=device)
        stats[f"stats_{origin}_creation_dist_abs"] = torch.zeros((n_exp, 10, 10), device=device)
    return stats


def _observation_triggerGPU4(total_episodes: int, curr_ep_idx: int) -> Tuple[bool, int]:
    observation_points = np.linspace(0, total_episodes - 1, 10, dtype=int)
    if curr_ep_idx in observation_points:
        return True, int(np.where(observation_points == curr_ep_idx)[0][0])
    return False, -1


def run_experimentGPU4(
    experiment_config: ExperimentConfigGPU4,
    agent_cfg: Optional[ACS2ConfigurationGPU4] = None,
    stop_after_phase: Optional[str] = None,
    initial_agent: Optional[ACS2GPU4] = None,
    episode_offset: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, float], int, Optional[AgentSelectionGPU4], float, EnvironmentGPU4]:
    if agent_cfg is None:
        from src.hybrid_utils import build_agent_configGPU4
        agent_cfg = build_agent_configGPU4(experiment_config)

    phase_seed = experiment_config.seed + (episode_offset * 1000003)
    _seed_gpu_run(phase_seed)
        
    device_name = resolve_deviceGPU4(experiment_config.device)
    environment = create_environmentGPU4(experiment_config.environment, n_exp=experiment_config.n_exp, device=device_name)
    agent = initial_agent if initial_agent is not None else ACS2GPU4(agent_cfg, experiment_config.n_exp, device=device_name)
    if initial_agent is not None:
        agent.cfg = agent_cfg
        agent.device = torch.device(device_name)
    total_episodes = experiment_config.total_episodes
    idx_p2_start = experiment_config.phases["explore"].episodes
    idx_p3_start = idx_p2_start + experiment_config.phases["exploit1"].episodes
    metric_context = build_metric_contextGPU4(environment)
    stats_gpu = _empty_statsGPU4(experiment_config.n_exp, total_episodes, environment.device)
    origin_perc_cache = torch.zeros((experiment_config.n_exp, len(agent.origin_map)), device=environment.device)
    origin_abs_cache = torch.zeros((experiment_config.n_exp, len(agent.origin_map)), device=environment.device)
    metric_cache = tuple(torch.zeros(experiment_config.n_exp, device=environment.device) for _ in range(12))
    timing_breakdown = {
        "reset_step_s": 0.0,
        "run_step_s": 0.0,
        "env_step_s": 0.0,
        "apply_learning_s": 0.0,
        "metrics_s": 0.0,
        "creation_snapshot_s": 0.0,
        "transfer_to_numpy_s": 0.0,
    }

    start_wall_time = time.time()
    curr_ep_idx = 0
    stop_integrate = False
    for phase_name in ("explore", "exploit1", "exploit2"):
        phase = experiment_config.phases[phase_name]
        if phase.episodes <= 0:
            continue

        agent.epsilon = phase.epsilon
        agent.beta = phase.beta
        agent.cfg.do_alp = phase.alp
        agent.cfg.do_ga = phase.ga
        agent.cfg.do_decay_epsilon = phase.decay

        for _ in range(phase.episodes):
            agent.curr_ep_idx = episode_offset + curr_ep_idx
            reset_start = time.perf_counter()
            prev_states = environment.reset()
            dones = torch.zeros(experiment_config.n_exp, dtype=torch.bool, device=environment.device)
            steps = torch.zeros(experiment_config.n_exp, dtype=torch.int32, device=environment.device)
            timing_breakdown["reset_step_s"] += time.perf_counter() - reset_start

            while not torch.all(dones):
                active_mask = ~dones
                step_start = time.perf_counter()
                actions, action_mask = agent.run_step(prev_states, active_mask=active_mask)
                timing_breakdown["run_step_s"] += time.perf_counter() - step_start
                env_start = time.perf_counter()
                next_states, rewards, step_dones = environment.step(actions)
                timing_breakdown["env_step_s"] += time.perf_counter() - env_start

                # Freeze inactive experiments so they cannot accrue post-terminal updates.
                next_states = torch.where(active_mask.unsqueeze(1), next_states, prev_states)
                rewards = torch.where(active_mask, rewards, torch.zeros_like(rewards))
                step_dones = torch.where(active_mask, step_dones, torch.zeros_like(step_dones))
                learn_start = time.perf_counter()
                agent.apply_learning(active_mask, action_mask, prev_states, actions, rewards, next_states, step_dones)
                timing_breakdown["apply_learning_s"] += time.perf_counter() - learn_start
                
                steps = torch.where(active_mask, steps + 1, steps)
                dones = dones | step_dones
                prev_states = next_states
                
                if torch.any(steps >= experiment_config.n_steps):
                    dones = dones | (steps >= experiment_config.n_steps)

            stats_gpu["stats_steps"][:, curr_ep_idx] = steps.float()
            
            # Periodically calculate metrics
            if (curr_ep_idx + 1) % experiment_config.metric_calculation_frequency == 0:
                metrics_start = time.perf_counter()
                (
                    know, generalization, micro, macro, rmicro, rmacro, 
                    avg_r, avg_rel_r, avg_q_all, avg_q_rel, avg_fit_all, avg_fit_rel,
                    origin_perc_cache, origin_abs_cache
                ) = calculate_metricsGPU4(agent, metric_context)
                timing_breakdown["metrics_s"] += time.perf_counter() - metrics_start
                
                stats_gpu["stats_know"][:, curr_ep_idx] = know
                stats_gpu["stats_generalization"][:, curr_ep_idx] = generalization
                stats_gpu["stats_micro"][:, curr_ep_idx] = micro
                stats_gpu["stats_macro"][:, curr_ep_idx] = macro
                stats_gpu["stats_rmicro"][:, curr_ep_idx] = rmicro
                stats_gpu["stats_rmacro"][:, curr_ep_idx] = rmacro
                stats_gpu["stats_avg_r"][:, curr_ep_idx] = avg_r
                stats_gpu["stats_avg_rel_r"][:, curr_ep_idx] = avg_rel_r
                stats_gpu["stats_avg_q_all"][:, curr_ep_idx] = avg_q_all
                stats_gpu["stats_avg_q_rel"][:, curr_ep_idx] = avg_q_rel
                stats_gpu["stats_avg_fit_all"][:, curr_ep_idx] = avg_fit_all
                stats_gpu["stats_avg_fit_rel"][:, curr_ep_idx] = avg_fit_rel
                
                for origin in agent.origin_map.keys():
                    stats_gpu[f"stats_{origin}_perc"][:, curr_ep_idx] = origin_perc_cache[:, agent.origin_map[origin]]
                    stats_gpu[f"stats_{origin}_abs"][:, curr_ep_idx] = origin_abs_cache[:, agent.origin_map[origin]]
                
                # Cache for non-calculating steps
                metric_cache = (
                    know, generalization, micro, macro, rmicro, rmacro, 
                    avg_r, avg_rel_r, avg_q_all, avg_q_rel, avg_fit_all, avg_fit_rel
                )
            else:
                # Use cached metrics
                stats_gpu["stats_know"][:, curr_ep_idx] = metric_cache[0]
                stats_gpu["stats_generalization"][:, curr_ep_idx] = metric_cache[1]
                stats_gpu["stats_micro"][:, curr_ep_idx] = metric_cache[2]
                stats_gpu["stats_macro"][:, curr_ep_idx] = metric_cache[3]
                stats_gpu["stats_rmicro"][:, curr_ep_idx] = metric_cache[4]
                stats_gpu["stats_rmacro"][:, curr_ep_idx] = metric_cache[5]
                stats_gpu["stats_avg_r"][:, curr_ep_idx] = metric_cache[6]
                stats_gpu["stats_avg_rel_r"][:, curr_ep_idx] = metric_cache[7]
                stats_gpu["stats_avg_q_all"][:, curr_ep_idx] = metric_cache[8]
                stats_gpu["stats_avg_q_rel"][:, curr_ep_idx] = metric_cache[9]
                stats_gpu["stats_avg_fit_all"][:, curr_ep_idx] = metric_cache[10]
                stats_gpu["stats_avg_fit_rel"][:, curr_ep_idx] = metric_cache[11]
                for origin in agent.origin_map.keys():
                    stats_gpu[f"stats_{origin}_perc"][:, curr_ep_idx] = origin_perc_cache[:, agent.origin_map[origin]]
                    stats_gpu[f"stats_{origin}_abs"][:, curr_ep_idx] = origin_abs_cache[:, agent.origin_map[origin]]

            should_capture, observation_index = _observation_triggerGPU4(total_episodes, curr_ep_idx)
            if should_capture:
                creation_start = time.perf_counter()
                creation_dist, creation_dist_abs = calculate_creation_distribution_snapshotGPU4(
                    agent,
                    observation_index,
                    total_episodes,
                )
                timing_breakdown["creation_snapshot_s"] += time.perf_counter() - creation_start
                for origin in agent.origin_map.keys():
                    stats_gpu[f"stats_{origin}_creation_dist"] += creation_dist[origin]
                    stats_gpu[f"stats_{origin}_creation_dist_abs"] += creation_dist_abs[origin]

            curr_ep_idx += 1
            agent.curr_ep_idx = episode_offset + curr_ep_idx

        if stop_after_phase == phase_name:
            stop_integrate = True
            break

    total_wall_time = time.time() - start_wall_time
    effective_episodes = curr_ep_idx
    
    # Convert stats to numpy for compatibility with CPU-based dashboard/merging
    transfer_start = time.perf_counter()
    stats_np = {k: v[:, :effective_episodes].cpu().numpy() for k, v in stats_gpu.items() if v.ndim <= 2}
    # Handling 3D creation distributions
    for k, v in stats_gpu.items():
        if v.ndim == 3:
            stats_np[k] = v.cpu().numpy()
    timing_breakdown["transfer_to_numpy_s"] += time.perf_counter() - transfer_start
    stats_np["all_agents"] = agent

    exploit_slice = stats_gpu["stats_steps"][:, idx_p2_start:curr_ep_idx]
    if exploit_slice.numel() == 0:
        per_exp_mean_steps = torch.mean(stats_gpu["stats_steps"][:, :curr_ep_idx], dim=1)
    else:
        per_exp_mean_steps = torch.mean(exploit_slice, dim=1)
    best_agent = AgentSelectionGPU4(agent, int(torch.argmin(per_exp_mean_steps).item()))
    
    # Final Summary
    exploit2_slice = stats_gpu["stats_steps"][:, idx_p3_start:curr_ep_idx]
    if exploit2_slice.numel() == 0:
        exploit2_slice = stats_gpu["stats_steps"][:, :curr_ep_idx]

    summary_stats = {
        "Total Time": total_wall_time,
        "Avg Time": total_wall_time / experiment_config.n_exp,
        "Std Time": 0.0, # GPU batching doesn't have per-experiment timing
        "Exploit Avg. Steps": float(torch.mean(exploit2_slice).item()),
        "Knowledge": float(torch.mean(stats_gpu["stats_know"][:, curr_ep_idx-1]).item()),
        "Generalization": float(torch.mean(stats_gpu["stats_generalization"][:, curr_ep_idx-1]).item()),
        "Rew (All)": float(torch.mean(stats_gpu["stats_avg_r"][:, curr_ep_idx-1]).item()),
        "Rew (Rel)": float(torch.mean(stats_gpu["stats_avg_rel_r"][:, curr_ep_idx-1]).item()),
        "Micro": float(torch.mean(stats_gpu["stats_micro"][:, curr_ep_idx-1]).item()),
        "Macro": float(torch.mean(stats_gpu["stats_macro"][:, curr_ep_idx-1]).item()),
        "Micro (Rel)": float(torch.mean(stats_gpu["stats_rmicro"][:, curr_ep_idx-1]).item()),
        "Macro (Rel)": float(torch.mean(stats_gpu["stats_rmacro"][:, curr_ep_idx-1]).item()),
        "ALP Marking": "Restricted (on incorrect)" if agent_cfg.alp_mark_only_incorrect else "Full Action Set",
        "Subsumption": "Disabled" if not agent_cfg.do_subsumption else "Enabled",
    }
    summary_stats["Timing Breakdown"] = {
        **timing_breakdown,
        **agent.profile_stats,
    }

    optimal_avg_steps = calculate_optimal_metricsGPU4(environment)
    return stats_np, summary_stats, effective_episodes, best_agent, optimal_avg_steps, environment
