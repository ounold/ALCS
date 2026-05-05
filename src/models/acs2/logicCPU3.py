from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

import numpy as np

from .classifierCPU3 import SYMBOL_BITS_CPU3, SYMBOL_MASK_CPU3

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None

if TYPE_CHECKING:
    from .acs2CPU3 import ACS2CPU3
    from .classifierCPU3 import ClassifierCPU3
    from .confCPU3 import ACS2ConfigurationCPU3


def apply_gaCPU3(agent: "ACS2CPU3", action_set: List["ClassifierCPU3"]) -> None:
    total_num = sum(cl.num for cl in action_set)
    if total_num > 0:
        avg_t = sum(cl.t_ga * cl.num for cl in action_set) / total_num
        if agent.time - avg_t > agent.cfg.theta_ga and len(action_set) >= 2:
            ga_evolveCPU3(agent, action_set)
    while sum(cl.num for cl in action_set) > agent.cfg.theta_as:
        delete_victimCPU3(agent, action_set)


def ga_evolveCPU3(agent: "ACS2CPU3", action_set: List["ClassifierCPU3"]) -> None:
    parent_one = select_offspringCPU3(action_set)
    parent_two = select_offspringCPU3(action_set)
    child_one = parent_one.copy()
    child_two = parent_two.copy()
    for child in (child_one, child_two):
        child.num = 1
        child.exp = 0
        child.t_ga = agent.time
        child.parents = (parent_one.id, parent_two.id)
        child.origin_source = "ga"
        child.creation_episode = agent.curr_ep_idx
    if random.random() < agent.cfg.chi:
        crossoverCPU3(child_one, child_two)
    mutateCPU3(child_one, agent.cfg)
    mutateCPU3(child_two, agent.cfg)
    child_one.q = 0.5
    child_two.q = 0.5
    agent.add_to_population(child_one)
    agent.add_to_population(child_two)


def select_offspringCPU3(action_set: List["ClassifierCPU3"]) -> "ClassifierCPU3":
    qualities = np.array([cl.q for cl in action_set], dtype=np.float64)
    weights = qualities ** 3
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return random.choice(action_set)
    probabilities = weights / total_weight
    return action_set[int(np.random.choice(len(action_set), p=probabilities))]


def mutateCPU3(cl: "ClassifierCPU3", cfg: "ACS2ConfigurationCPU3") -> None:
    for i in range(cfg.l_len):
        shift = i * SYMBOL_BITS_CPU3
        if random.random() < cfg.mu:
            if shift < 64:
                if (cl.wildcard_mask[0] >> shift) & SYMBOL_MASK_CPU3:
                    cl.wildcard_mask = (cl.wildcard_mask[0] & ~(SYMBOL_MASK_CPU3 << shift), cl.wildcard_mask[1])
                    cl.condition_bits = (cl.condition_bits[0] & ~(SYMBOL_MASK_CPU3 << shift), cl.condition_bits[1])
            else:
                if (cl.wildcard_mask[1] >> (shift - 64)) & SYMBOL_MASK_CPU3:
                    cl.wildcard_mask = (cl.wildcard_mask[0], cl.wildcard_mask[1] & ~(SYMBOL_MASK_CPU3 << (shift - 64)))
                    cl.condition_bits = (cl.condition_bits[0], cl.condition_bits[1] & ~(SYMBOL_MASK_CPU3 << (shift - 64)))
    cl.sync_from_bits()


def crossoverCPU3(c1: "ClassifierCPU3", c2: "ClassifierCPU3") -> None:
    point = random.randint(0, c1.cfg.l_len)
    shift = point * SYMBOL_BITS_CPU3
    if shift < 64:
        mask = (1 << shift) - 1
        c1_bits, c2_bits = c1.condition_bits, c2.condition_bits
        c1.condition_bits = ((c1_bits[0] & mask) | (c2_bits[0] & ~mask), c1_bits[1])
        c2.condition_bits = ((c2_bits[0] & mask) | (c1_bits[0] & ~mask), c2_bits[1])
        c1_wild, c2_wild = c1.wildcard_mask, c2.wildcard_mask
        c1.wildcard_mask = ((c1_wild[0] & mask) | (c2_wild[0] & ~mask), c1_wild[1])
        c2.wildcard_mask = ((c2_wild[0] & mask) | (c1_wild[0] & ~mask), c2_wild[1])
    else:
        mask = (1 << (shift - 64)) - 1
        c1_bits, c2_bits = c1.condition_bits, c2.condition_bits
        c1.condition_bits = (c2_bits[0], (c1_bits[1] & mask) | (c2_bits[1] & ~mask))
        c2.condition_bits = (c1_bits[0], (c2_bits[1] & mask) | (c1_bits[1] & ~mask))
        c1_wild, c2_wild = c1.wildcard_mask, c2.wildcard_mask
        c1.wildcard_mask = (c2_wild[0], (c1_wild[1] & mask) | (c2_wild[1] & ~mask))
        c2.wildcard_mask = (c1_wild[0], (c2_wild[1] & mask) | (c1_wild[1] & ~mask))
    c1.sync_from_bits()
    c2.sync_from_bits()


def delete_victimCPU3(agent: "ACS2CPU3", action_set: List["ClassifierCPU3"]) -> None:
    victim = min(action_set, key=lambda cl: cl.q)
    if victim.num > 1:
        victim.num -= 1
    else:
        agent.remove_from_population(victim)
        if victim in action_set:
            action_set.remove(victim)


def _fast_subsume_python(
    gen_action: int,
    gen_bits_0: int, gen_bits_1: int,
    gen_wild_0: int, gen_wild_1: int,
    gen_exp: int,
    gen_q: float,
    spec_action: int,
    spec_bits_0: int, spec_bits_1: int,
    spec_wild_0: int, spec_wild_1: int,
    theta_exp: int,
    theta_r: float,
) -> bool:
    if gen_action != spec_action:
        return False
    if gen_exp <= theta_exp or gen_q <= theta_r:
        return False
    if (spec_bits_0 & gen_wild_0) != gen_bits_0 or (spec_bits_1 & gen_wild_1) != gen_bits_1:
        return False
    if (gen_wild_0 & spec_wild_0) != gen_wild_0 or (gen_wild_1 & spec_wild_1) != gen_wild_1:
        return False
    if gen_wild_0 == spec_wild_0 and gen_wild_1 == spec_wild_1:
        return False
    return True


fast_subsumeCPU3 = _fast_subsume_python


def does_subsumeCPU3(gen: "ClassifierCPU3", spec: "ClassifierCPU3", cfg: "ACS2ConfigurationCPU3") -> bool:
    if gen.effect_bits != spec.effect_bits or gen.effect_wildcard_mask != spec.effect_wildcard_mask:
        return False
    if any(len(mark_set) > 0 for mark_set in gen.M):
        return False
    return bool(
        fast_subsumeCPU3(
            gen.A,
            gen.condition_bits[0], gen.condition_bits[1],
            gen.wildcard_mask[0], gen.wildcard_mask[1],
            gen.exp,
            gen.q,
            spec.A,
            spec.condition_bits[0], spec.condition_bits[1],
            spec.wildcard_mask[0], spec.wildcard_mask[1],
            cfg.theta_exp,
            cfg.theta_r,
        )
    )


def generate_covering_classifiersCPU3(agent: "ACS2CPU3", state: List[str]) -> List["ClassifierCPU3"]:
    action = random.randint(0, agent.cfg.num_actions - 1)
    return [create_covering_classifierCPU3(agent.cfg, state, action, agent.time, agent.curr_ep_idx)]


def create_covering_classifierCPU3(cfg: "ACS2ConfigurationCPU3", state: List[str], action: int, time_created: int, episode: int) -> "ClassifierCPU3":
    from .classifierCPU3 import ClassifierCPU3

    condition = ["#" for _ in range(cfg.l_len)]
    for idx in random.sample(range(cfg.l_len), min(cfg.u_max, cfg.l_len)):
        condition[idx] = state[idx]
    effect = list(state) if cfg.do_simple_mode else ["#" for _ in range(cfg.l_len)]
    classifier = ClassifierCPU3(condition, action, effect, cfg, time_created, origin_source="covering", creation_episode=episode)
    classifier.q = 0.5
    classifier.r = 0.5
    return classifier
