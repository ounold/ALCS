from typing import List, Dict
import random

from .conf import ACS2Configuration
from .classifier import Classifier
from . import logic

class ACS2:
    """
    The main ACS2 agent class.
    """
    def __init__(self, cfg: ACS2Configuration):
        self.cfg = cfg
        self.population: List[Classifier] = []
        self.time: int = 0
        self.curr_ep_idx: int = 0

    def run_step(self, env_state: List[str], explore: bool = True) -> (int, List[Classifier]):
        """
        Executes one step of the ACS2 algorithm.
        """
        match_set = [cl for cl in self.population if cl.matches(env_state)]

        if not match_set:
            match_set = logic.generate_covering_classifiers(self, env_state)
            self.population.extend(match_set)

        # Build a map of action -> max fitness
        possible_actions: Dict[int, float] = {}
        for cl in match_set:
            val = cl.fitness
            if val > possible_actions.get(cl.A, -1.0):
                possible_actions[cl.A] = val

        # Select an action
        if not possible_actions:
            # If no actions are possible, choose a random one
            action = random.randint(0, self.cfg.num_actions - 1)
        elif explore and random.random() < self.cfg.epsilon:
            # Randomly choose among possible actions or any action
            if random.random() < 0.5:
                action = random.choice(list(possible_actions.keys()))
            else:
                action = random.randint(0, self.cfg.num_actions - 1)
        else:
            # Choose the best action
            action = max(possible_actions, key=possible_actions.get)

        action_set = [cl for cl in match_set if cl.A == action]

        # If action set is empty, create a new classifier
        if not action_set:
            new_cl = logic.create_covering_classifier(self.cfg, env_state, action, self.time, self.curr_ep_idx)
            self.add_to_population(new_cl)
            action_set.append(new_cl)

        return action, action_set

    def apply_learning(self, action_set: List[Classifier], prev_state: List[str], action: int, reward: float, next_state: List[str], is_terminal: bool = False, curr_ep_idx: int = 0):
        """
        Applies reinforcement learning and other mechanisms.
        """
        self.curr_ep_idx = curr_ep_idx

        # In simple mode, a simplified Q-learning like update is used
        if self.cfg.do_simple_mode:
            self._apply_simple(action_set, prev_state, action, reward, next_state, is_terminal)
            return

        # Anticipatory Learning Process (ALP)
        if self.cfg.do_alp:
            self._apply_alp(action_set, prev_state, action, next_state)

        # Reinforcement Learning (RL) update
        next_match_set = [cl for cl in self.population if cl.matches(next_state)]
        
        max_q_r = 0.0
        if not is_terminal and next_match_set:
            max_q_r = max(cl.fitness for cl in next_match_set)

        for cl in action_set:
            cl.r += self.cfg.beta * (reward + self.cfg.gamma * max_q_r - cl.r)
            cl.ir += self.cfg.beta * (reward - cl.ir)

        # Genetic Algorithm (GA)
        if self.cfg.do_ga:
            logic.apply_ga(self, action_set)
        else:
            # If GA is off, still need to control action set size
            while sum(cl.num for cl in action_set) > self.cfg.theta_as:
                logic.delete_victim(self, action_set)


    def _apply_simple(self, action_set: List[Classifier], prev_state: List[str], action: int, reward: float, next_state: List[str], is_terminal: bool = False):
        """ A simplified learning process for debugging and baseline. """
        for cl in action_set:
            cl.exp += 1
            
        has_correct = any(cl.get_anticipation(prev_state) == next_state for cl in action_set)
        
        if not has_correct:
            new_cl = Classifier(prev_state, action, next_state, self.cfg, self.time, origin_source="simple_mode_covering", creation_episode=self.curr_ep_idx)
            self.add_to_population(new_cl)
            action_set.append(new_cl)

        next_match = [cl for cl in self.population if cl.matches(next_state)]
        
        max_rew = 0.0
        if not is_terminal and next_match:
            max_rew = max(cl.r for cl in next_match)

        for cl in action_set:
            cl.r += self.cfg.beta * (reward + self.cfg.gamma * max_rew - cl.r)
            if cl.get_anticipation(prev_state) == next_state:
                cl.q += self.cfg.beta * (1 - cl.q)
            else:
                cl.q += self.cfg.beta * (0 - cl.q)

    def _apply_alp(self, action_set: List[Classifier], prev_state: List[str], action: int, current_state: List[str]):
        """ The Anticipatory Learning Process. """
        did_anticipate = False
        
        # Operate on a copy as the set might be modified
        for cl in action_set[:]: 
            cl.exp += 1
            
            # Update average age
            if cl.exp < 1.0 / self.cfg.beta:
                cl.aav = (cl.aav * (cl.exp - 1) + (self.time - cl.t_alp)) / cl.exp
            else:
                cl.aav += self.cfg.beta * ((self.time - cl.t_alp) - cl.aav)
            cl.t_alp = self.time

            # Check if anticipation was correct
            if cl.get_anticipation(prev_state) == current_state:
                cl.q += self.cfg.beta * (1 - cl.q)
                if cl.q > self.cfg.theta_r:
                    self._alp_expected(cl, prev_state)
                did_anticipate = True
            else:
                cl.q -= self.cfg.beta * cl.q
                self._alp_mark(cl, prev_state)
                self._alp_unexpected(cl, prev_state, current_state)
                
                # Remove if quality is too low
                if cl.q < self.cfg.theta_i:
                    if cl in self.population: self.population.remove(cl)
                    if cl in action_set: action_set.remove(cl)

        if not did_anticipate:
            self._generate_alp_cover(action_set, prev_state, action, current_state)

    def _alp_expected(self, cl: Classifier, state: List[str]):
        """ ALP: Specialization for correct anticipations. """
        if all(len(m) == 0 for m in cl.M):
            return

        off_C = list(cl.C)
        changed = False
        for i in range(self.cfg.l_len):
            if cl.C[i] == '#' and len(cl.M[i]) > 0:
                if any(v != state[i] for v in cl.M[i]):
                    off_C[i] = state[i]
                    changed = True
        
        if changed:
            child = Classifier(off_C, cl.A, cl.E, self.cfg, self.time, origin_source="alp_expected", creation_episode=self.curr_ep_idx)
            child.q = cl.q
            child.r = cl.r
            self.add_to_population(child)

    def _alp_mark(self, cl: Classifier, prev_state: List[str]):
        """ ALP: Mark attribute values that were present. """
        for i, val in enumerate(prev_state):
            cl.M[i].add(val)

    def _alp_unexpected(self, cl: Classifier, prev_state: List[str], next_state: List[str]):
        """ ALP: Specialization for incorrect anticipations. """
        new_E = list(cl.E)
        changed = False
        for i in range(self.cfg.l_len):
            if new_E[i] == '#' and prev_state[i] != next_state[i]:
                new_E[i] = next_state[i]
                changed = True
        
        if changed:
            child = Classifier(cl.C, cl.A, new_E, self.cfg, self.time, origin_source="alp_unexpected", creation_episode=self.curr_ep_idx)
            self.add_to_population(child)
            
    def _generate_alp_cover(self, aset: List[Classifier], prev: List[str], act: int, nxt: List[str]):
        """ ALP: Generate a new classifier when no existing one anticipated correctly. """
        cond = ['#' for _ in range(self.cfg.l_len)]
        eff = ['#' for _ in range(self.cfg.l_len)]
        for i in range(self.cfg.l_len):
            if prev[i] != nxt[i]:
                cond[i] = prev[i]
                eff[i] = nxt[i]
                
        new_cl = Classifier(cond, act, eff, self.cfg, self.time, origin_source="alp_covering", creation_episode=self.curr_ep_idx)
        self.add_to_population(new_cl)
        aset.append(new_cl)

    def add_to_population(self, cl: Classifier):
        """
        Adds a classifier to the population, checking for duplicates and subsumption.
        """
        # Check for identical classifiers
        for ex in self.population:
            if ex == cl:
                ex.num += 1
                return

        # Check for subsumption by an existing classifier
        if self.cfg.do_subsumption:
            for ex in self.population:
                if logic.does_subsume(ex, cl, self.cfg):
                    ex.num += 1
                    return
        
        self.population.append(cl)
