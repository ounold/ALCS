from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from environment.runtime_gpu4 import EnvironmentGPU4
from src.models.acs2.acs2GPU4 import ACS2GPU4


@dataclass(frozen=True)
class MetricContextGPU4:
    states_to_check: torch.Tensor
    actions_to_check: torch.Tensor
    next_states_true: torch.Tensor
    num_checks: int


ORIGIN_KEYS_GPU4 = ("covering", "alp_unexpected", "alp_expected", "alp_covering", "ga")


def build_metric_contextGPU4(environment: EnvironmentGPU4) -> MetricContextGPU4:
    if not environment.supports_metric_evaluation:
        empty_states = torch.empty((0, environment.state_len), device=environment.device, dtype=torch.long)
        empty_actions = torch.empty((0,), device=environment.device, dtype=torch.long)
        return MetricContextGPU4(empty_states, empty_actions, empty_states, 0)
    metric_states = environment.metric_states()
    num_valid = len(metric_states)
    states_to_check = metric_states.repeat_interleave(environment.num_actions, dim=0)
    actions_to_check = torch.arange(environment.num_actions, device=environment.device).repeat(num_valid)
    next_states_true = environment.peek_step(states_to_check, actions_to_check)
    return MetricContextGPU4(states_to_check, actions_to_check, next_states_true, len(states_to_check))


def calculate_metricsGPU4(agent: ACS2GPU4, metric_context: MetricContextGPU4) -> Tuple[torch.Tensor, ...]:
    state_checks = metric_context.states_to_check
    action_checks = metric_context.actions_to_check
    next_states_true = metric_context.next_states_true
    match_cond = (agent.C.unsqueeze(2) == state_checks.unsqueeze(0).unsqueeze(0)) | (agent.C.unsqueeze(2) == -1)
    matches = torch.all(match_cond, dim=3) & agent.active_mask.unsqueeze(2)
    correct_action = agent.A.unsqueeze(2) == action_checks.unsqueeze(0).unsqueeze(0)
    # A classifier is reliable for a specific (state, action) if:
    # 1. It matches the state
    # 2. It has the correct action
    # 3. Its quality q > theta_r
    # 4. Its effect E correctly predicts the next state
    
    # Correct anticipation check:
    # If E[i] == -1 (wildcard effect), the predicted next state for that feature is the same as current state.
    # If E[i] != -1, the predicted next state for that feature is E[i].
    predicted_next = torch.where(agent.E.unsqueeze(2) == -1, state_checks.unsqueeze(0).unsqueeze(0), agent.E.unsqueeze(2))
    correct_anticipation = torch.all(predicted_next == next_states_true.unsqueeze(0).unsqueeze(0), dim=3)
    
    # In ACS2, a "reliable" prediction for knowledge usually also implies 
    # the classifier is NOT just a covering classifier that hasn't been tested.
    # q > theta_r is the standard check for reliability.
    is_reliable_cl = (agent.q > agent.cfg.theta_r).unsqueeze(2)
    
    # Combined mask: (N_exp, N_pop, N_checks)
    reliable_predictions = matches & correct_action & is_reliable_cl & correct_anticipation
    
    # For each experiment and each check, is there AT LEAST ONE reliable classifier?
    check_covered = torch.any(reliable_predictions, dim=1)
    
    # Knowledge is the fraction of (state, action) pairs covered by at least one reliable classifier
    knowledge = torch.sum(check_covered, dim=1).float() / max(metric_context.num_checks, 1)
    micro_pop = torch.sum(agent.num * agent.active_mask.long(), dim=1).float()
    macro_pop = torch.sum(agent.active_mask.long(), dim=1).float()
    reliable_mask = agent.active_mask & (agent.q > agent.cfg.theta_r)
    rel_micro = torch.sum(agent.num * reliable_mask.long(), dim=1).float()
    rel_macro = torch.sum(reliable_mask.long(), dim=1).float()
    avg_r = torch.sum(agent.r * agent.active_mask.float(), dim=1) / macro_pop.clamp(min=1.0)
    avg_rel_r = torch.sum(agent.r * reliable_mask.float(), dim=1) / rel_macro.clamp(min=1.0)
    avg_q = torch.sum(agent.q * agent.active_mask.float(), dim=1) / macro_pop.clamp(min=1.0)
    avg_q_rel = torch.sum(agent.q * reliable_mask.float(), dim=1) / rel_macro.clamp(min=1.0)
    fitness = agent.q * agent.r
    avg_fit = torch.sum(fitness * agent.active_mask.float(), dim=1) / macro_pop.clamp(min=1.0)
    avg_fit_rel = torch.sum(fitness * reliable_mask.float(), dim=1) / rel_macro.clamp(min=1.0)
    total_symbols = macro_pop * agent.l_len
    wildcards = torch.sum((agent.C == -1).float() * agent.active_mask.unsqueeze(2), dim=(1, 2))
    generalization = wildcards / total_symbols.clamp(min=1.0)
    
    # Origin distributions
    origin_perc, origin_counts_abs = calculate_origin_distributionGPU4(agent)
    
    return (
        knowledge, generalization, micro_pop, macro_pop, rel_micro, rel_macro, 
        avg_r, avg_rel_r, avg_q, avg_q_rel, avg_fit, avg_fit_rel,
        origin_perc, origin_counts_abs
    )


def calculate_origin_distributionGPU4(agent: ACS2GPU4) -> Tuple[torch.Tensor, torch.Tensor]:
    origin_one_hot = torch.nn.functional.one_hot(agent.origin_source.clamp(min=0), num_classes=len(agent.origin_map)).float()
    weighted = origin_one_hot * (agent.num * agent.active_mask.long()).unsqueeze(2).float()
    origin_counts_abs = torch.sum(weighted, dim=1)
    totals = torch.sum(origin_counts_abs, dim=1).clamp(min=1.0)
    origin_perc = origin_counts_abs / totals.unsqueeze(1) * 100.0
    return origin_perc, origin_counts_abs


def calculate_creation_distributionGPU4(agent: ACS2GPU4) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Returns 10x10 creation distribution matrices for each origin source.
    Rows: Observation points (currently just the latest)
    Cols: Creation intervals (relative to total episodes)
    """
    total_ep = agent.cfg.total_episodes
    bucket_size = max(1, (total_ep + 9) // 10)
    
    # Initialize dictionaries for origins
    creation_dist = {}
    creation_dist_abs = {}
    
    for origin_name, origin_idx in agent.origin_map.items():
        # (N_exp, 10, 10)
        dist_abs = torch.zeros((agent.n_exp, 10, 10), device=agent.device)
        
        # Current observation point is index 0 for now (simplified for GPU batching)
        # Actually, let's map creation_episode to 0-9 buckets
        buckets = (agent.creation_episode // bucket_size).clamp(max=9)
        
        # Mask for this origin and active
        mask = agent.active_mask & (agent.origin_source == origin_idx)
        
        # We need to scatter 'num' into the correct buckets
        # This is tricky to vectorize perfectly without loops over 10 buckets
        for b in range(10):
            b_mask = mask & (buckets == b)
            dist_abs[:, 0, b] = torch.sum(torch.where(b_mask, agent.num, torch.zeros_like(agent.num)), dim=1).float()
            
        creation_dist_abs[origin_name] = dist_abs
        totals = torch.sum(dist_abs, dim=2).clamp(min=1.0)
        creation_dist[origin_name] = (dist_abs / totals.unsqueeze(2)) * 100.0
        
    return creation_dist, creation_dist_abs


def calculate_creation_distribution_snapshotGPU4(agent: ACS2GPU4, observation_index: int, total_episodes: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    creation_dist: Dict[str, torch.Tensor] = {}
    creation_dist_abs: Dict[str, torch.Tensor] = {}
    bucket_size = max(1, (total_episodes + 9) // 10)

    for origin_name, origin_idx in agent.origin_map.items():
        dist_abs = torch.zeros((agent.n_exp, 10, 10), device=agent.device)
        buckets = (agent.creation_episode // bucket_size).clamp(max=9)
        mask = agent.active_mask & (agent.origin_source == origin_idx)

        for bucket in range(10):
            bucket_mask = mask & (buckets == bucket)
            dist_abs[:, observation_index, bucket] = torch.sum(
                torch.where(bucket_mask, agent.num, torch.zeros_like(agent.num)),
                dim=1,
            ).float()

        creation_dist_abs[origin_name] = dist_abs
        totals = torch.sum(dist_abs[:, observation_index, :], dim=1).clamp(min=1.0)
        dist = torch.zeros_like(dist_abs)
        dist[:, observation_index, :] = (dist_abs[:, observation_index, :] / totals.unsqueeze(1)) * 100.0
        creation_dist[origin_name] = dist

    return creation_dist, creation_dist_abs


def calculate_optimal_metricsGPU4(environment: EnvironmentGPU4) -> float:
    return float(environment.optimal_avg_steps())
