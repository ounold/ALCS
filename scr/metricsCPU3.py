from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from environment.runtime_cpu3 import EnvironmentCPU3
from src.models.acs2.acs2CPU3 import ACS2CPU3


@dataclass(frozen=True)
class MetricContextCPU3:
    states_np: np.ndarray
    next_states_np: np.ndarray
    num_valid_states: int
    num_actions: int


CREATION_KEYS_CPU3 = ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]


def build_metric_contextCPU3(agent: ACS2CPU3, env: EnvironmentCPU3) -> MetricContextCPU3:
    if not env.supports_metric_evaluation:
        return MetricContextCPU3(np.array([], dtype=object), np.array([], dtype=object), 0, env.num_actions)
    states = env.metric_states()
    if not states:
        return MetricContextCPU3(np.array([], dtype=object), np.array([], dtype=object), 0, env.num_actions)
    valid_states = [agent.state_to_bits(state) for state in states]
    next_states_bits = [[agent.state_to_bits(env.peek_step(state, action)) for action in range(env.num_actions)] for state in states]
    return MetricContextCPU3(
        states_np=np.array(valid_states, dtype=object),
        next_states_np=np.array(next_states_bits, dtype=object),
        num_valid_states=len(valid_states),
        num_actions=env.num_actions,
    )


def calculate_metricsCPU3(agent: ACS2CPU3, metric_context: MetricContextCPU3) -> Tuple[int, int, int, int, float, float, float, float, float, float, float, float]:
    population = agent.population
    if not population:
        return (0,) * 12

    pop_infos = [
        {
            "cond": cl.condition_bits,
            "wild": cl.wildcard_mask,
            "effect": cl.effect_bits,
            "effect_wild": cl.effect_wildcard_mask,
            "action": cl.A,
            "q": cl.q,
            "r": cl.r,
            "num": cl.num,
        }
        for cl in population
    ]

    knowledge = 0.0
    if metric_context.num_valid_states > 0:
        known_situations = 0
        for state_idx, state_bits in enumerate(metric_context.states_np):
            reliable_by_action = {action: False for action in range(metric_context.num_actions)}
            next_state_list = metric_context.next_states_np[state_idx]
            for info in pop_infos:
                action = info["action"]
                if action < 0 or action >= metric_context.num_actions:
                    continue
                if not (
                    (state_bits[0] & info["wild"][0]) == info["cond"][0]
                    and (state_bits[1] & info["wild"][1]) == info["cond"][1]
                ):
                    continue
                if info["q"] <= agent.cfg.theta_r:
                    continue
                pred_0 = (state_bits[0] & ~info["effect_wild"][0]) | info["effect"][0]
                pred_1 = (state_bits[1] & ~info["effect_wild"][1]) | info["effect"][1]
                target_bits = next_state_list[action]
                if pred_0 == target_bits[0] and pred_1 == target_bits[1]:
                    reliable_by_action[action] = True
            known_situations += sum(1 for reached in reliable_by_action.values() if reached)
        total_possible_situations = metric_context.num_valid_states * metric_context.num_actions
        knowledge = known_situations / total_possible_situations if total_possible_situations > 0 else 0.0

    macro_pop_size = len(population)
    micro_pop_size = int(sum(info["num"] for info in pop_infos))
    reliable_indices = [info for info in pop_infos if info["q"] > agent.cfg.theta_r]
    rel_macro_pop_size = len(reliable_indices)
    rel_micro_pop_size = int(sum(info["num"] for info in reliable_indices))

    avg_pop_reward = float(np.mean([info["r"] for info in pop_infos])) if macro_pop_size > 0 else 0.0
    avg_rel_reward = float(np.mean([info["r"] for info in reliable_indices])) if rel_macro_pop_size > 0 else 0.0
    avg_q_all = float(np.mean([info["q"] for info in pop_infos])) if macro_pop_size > 0 else 0.0
    avg_q_rel = float(np.mean([info["q"] for info in reliable_indices])) if rel_macro_pop_size > 0 else 0.0

    pop_fitness = [info["q"] * info["r"] for info in pop_infos]
    avg_fit_all = float(np.mean(pop_fitness)) if macro_pop_size > 0 else 0.0
    avg_fit_rel = float(np.mean([info["q"] * info["r"] for info in reliable_indices])) if rel_macro_pop_size > 0 else 0.0

    total_attributes = macro_pop_size * agent.cfg.l_len
    specified_count = sum(classifier.specified_attribute_count() for classifier in population)
    wildcard_count = total_attributes - specified_count
    generalization_ratio = wildcard_count / total_attributes if total_attributes > 0 else 0.0

    return (
        micro_pop_size,
        rel_micro_pop_size,
        macro_pop_size,
        rel_macro_pop_size,
        float(knowledge),
        avg_pop_reward,
        avg_rel_reward,
        avg_q_all,
        avg_q_rel,
        avg_fit_all,
        avg_fit_rel,
        float(generalization_ratio),
    )


def calculate_origin_distributionCPU3(agent: ACS2CPU3) -> Tuple[Dict[str, float], Dict[str, int]]:
    origin_counts = {origin: 0 for origin in CREATION_KEYS_CPU3}
    if not agent or not agent.population:
        return {origin: 0.0 for origin in CREATION_KEYS_CPU3}, origin_counts
    for classifier in agent.population:
        if classifier.origin_source in origin_counts:
            origin_counts[classifier.origin_source] += classifier.num
    total_classifiers = sum(origin_counts.values())
    if total_classifiers == 0:
        return {origin: 0.0 for origin in CREATION_KEYS_CPU3}, origin_counts
    return ({origin: (count / total_classifiers) * 100.0 for origin, count in origin_counts.items()}, origin_counts)


def calculate_optimal_metricsCPU3(environment: EnvironmentCPU3) -> float:
    return float(environment.optimal_avg_steps())
