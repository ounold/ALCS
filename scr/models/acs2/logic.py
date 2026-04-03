import random
import copy
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .acs2 import ACS2
    from .classifier import Classifier
    from .conf import ACS2Configuration


def apply_ga(agent: 'ACS2', action_set: List['Classifier']):
    """
    Applies the genetic algorithm to the action set.
    """
    total_num = sum(cl.num for cl in action_set)
    if total_num > 0:
        avg_t = sum(cl.t_ga * cl.num for cl in action_set) / total_num
        if agent.time - avg_t > agent.cfg.theta_ga and len(action_set) >= 2:
            ga_evolve(agent, action_set)
    
    while sum(cl.num for cl in action_set) > agent.cfg.theta_as:
        delete_victim(agent, action_set)


def ga_evolve(agent: 'ACS2', aset: List['Classifier']):
    """
    Evolves the action set using GA operators.
    """
    p1 = select_offspring(aset)
    p2 = select_offspring(aset)
    c1 = copy.deepcopy(p1)
    c2 = copy.deepcopy(p2)
    
    c1.num = c2.num = 1
    c1.exp = c2.exp = 0
    c1.t_ga = agent.time
    c1.parents = (p1.id, p2.id)
    c2.parents = (p1.id, p2.id)
    c1.origin_source = "ga" # Set origin source
    c2.origin_source = "ga" # Set origin source
    c1.creation_episode = agent.curr_ep_idx # Set creation episode
    c2.creation_episode = agent.curr_ep_idx # Set creation episode

    if random.random() < agent.cfg.chi:
        crossover(c1, c2)
        
    mutate(c1, agent.cfg)
    mutate(c2, agent.cfg)
    
    c1.q = c2.q = 0.5
    
    # Subsumption is handled within add_to_population
    agent.add_to_population(c1)
    agent.add_to_population(c2)


def select_offspring(aset: List['Classifier']) -> 'Classifier':
    """
    Selects a classifier from the action set for reproduction.
    """
    weights = [cl.q ** 3 for cl in aset]
    if sum(weights) == 0:
        return random.choice(aset)
    return random.choices(aset, weights=weights, k=1)[0]


def mutate(cl: 'Classifier', cfg: 'ACS2Configuration'):
    """
    Mutates the condition of a classifier.
    """
    for i in range(cfg.l_len):
        if cl.C[i] != '#' and random.random() < cfg.mu:
            cl.C[i] = '#'


def crossover(c1: 'Classifier', c2: 'Classifier'):
    """
    Performs crossover between two classifiers.
    """
    pt = random.randint(0, len(c1.C) - 1)
    c1.C[pt:], c2.C[pt:] = c2.C[pt:], c1.C[pt:]


def delete_victim(agent: 'ACS2', aset: List['Classifier']):
    """
    Deletes a classifier from the population.
    """
    victim = min(aset, key=lambda cl: cl.q)
    if victim.num > 1:
        victim.num -= 1
    else:
        if victim in agent.population:
            agent.population.remove(victim)
        if victim in aset:
            aset.remove(victim)


def does_subsume(gen: 'Classifier', spec: 'Classifier', cfg: 'ACS2Configuration') -> bool:
    """
    Checks if a general classifier subsumes a specific one.
    """
    if gen.A != spec.A or gen.E != spec.E:
        return False
    
    if any(len(m) > 0 for m in gen.M):
        return False

    if gen.exp <= cfg.theta_exp or gen.q <= cfg.theta_r:
        return False

    is_consistent = all(g == '#' or g == s for g, s in zip(gen.C, spec.C))
    if not is_consistent:
        return False

    is_more_general = any(g == '#' and s != '#' for g, s in zip(gen.C, spec.C))
    return is_more_general


def generate_covering_classifiers(agent: 'ACS2', state: List[str]) -> List['Classifier']:
    """
    Generates covering classifiers for a given state.
    """
    action = random.randint(0, agent.cfg.num_actions - 1)
    return [create_covering_classifier(agent.cfg, state, action, agent.time, agent.curr_ep_idx)]


def create_covering_classifier(cfg: 'ACS2Configuration', state: List[str], action: int, time: int, episode: int) -> 'Classifier':
    """
    Creates a single covering classifier.
    """
    from .classifier import Classifier
    cond = ['#' for _ in range(cfg.l_len)]
    idxs = random.sample(range(cfg.l_len), min(cfg.u_max, cfg.l_len))
    for i in idxs:
        cond[i] = state[i]
        
    eff = list(state) if cfg.do_simple_mode else ['#' for _ in range(cfg.l_len)]
    
    cl = Classifier(cond, action, eff, cfg, time, origin_source="covering", creation_episode=episode)
    cl.q = cl.r = 0.5
    return cl

# ... other logic functions can be placed here
