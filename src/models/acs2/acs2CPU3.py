from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

from .classifierCPU3 import ClassifierCPU3, SYMBOL_BITS_CPU3
from .confCPU3 import ACS2ConfigurationCPU3
from .logicCPU3 import apply_gaCPU3, create_covering_classifierCPU3, delete_victimCPU3, does_subsumeCPU3, generate_covering_classifiersCPU3


class ACS2CPU3:
    def __init__(self, cfg: ACS2ConfigurationCPU3):
        self.cfg = cfg
        self.population_dict: Dict[Tuple[Tuple[int, int], Tuple[int, int], int, Tuple[int, int], Tuple[int, int]], ClassifierCPU3] = {}
        self.population_by_action: Dict[int, List[ClassifierCPU3]] = {action: [] for action in range(cfg.num_actions)}
        self.time = 0
        self.curr_ep_idx = 0

    @property
    def population(self) -> List[ClassifierCPU3]:
        return list(self.population_dict.values())

    @staticmethod
    def state_to_bits(state: List[str]) -> Tuple[int, int]:
        state_bits_0 = 0
        state_bits_1 = 0
        for i, symbol in enumerate(state):
            shift = i * SYMBOL_BITS_CPU3
            if shift < 64:
                state_bits_0 |= int(symbol) << shift
            else:
                state_bits_1 |= int(symbol) << (shift - 64)
        return (state_bits_0, state_bits_1)

    def _match_population(self, state_bits: Tuple[int, int]) -> List[ClassifierCPU3]:
        return [classifier for classifier in self.population_dict.values() if classifier.matches_bits(state_bits)]

    def run_step(self, env_state: List[str], explore: bool = True) -> Tuple[int, List[ClassifierCPU3]]:
        state_bits = self.state_to_bits(env_state)
        match_set = self._match_population(state_bits)
        if not match_set:
            match_set = generate_covering_classifiersCPU3(self, env_state)
            for classifier in match_set:
                self.add_to_population(classifier)
        possible_actions: Dict[int, float] = {}
        for classifier in match_set:
            possible_actions[classifier.A] = max(classifier.fitness, possible_actions.get(classifier.A, -1.0))
        if not possible_actions:
            action = random.randint(0, self.cfg.num_actions - 1)
        elif explore and random.random() < self.cfg.epsilon:
            action = random.randint(0, self.cfg.num_actions - 1)
        else:
            action = max(possible_actions, key=possible_actions.get)
        action_set = [classifier for classifier in match_set if classifier.A == action]
        if not action_set:
            new_classifier = create_covering_classifierCPU3(self.cfg, env_state, action, self.time, self.curr_ep_idx)
            self.add_to_population(new_classifier)
            action_set.append(new_classifier)
        return action, action_set

    def apply_learning(self, action_set: List[ClassifierCPU3], prev_state: List[str], action: int, reward: float, next_state: List[str], is_terminal: bool = False, curr_ep_idx: int = 0) -> None:
        self.curr_ep_idx = curr_ep_idx
        self.time += 1
        if self.cfg.do_simple_mode:
            self._apply_simple(action_set, prev_state, action, reward, next_state, is_terminal)
            return
        if self.cfg.do_alp:
            self._apply_alp(action_set, prev_state, action, next_state)
        next_match_set = self._match_population(self.state_to_bits(next_state))
        max_q_r = max((classifier.fitness for classifier in next_match_set), default=0.0) if not is_terminal else 0.0
        for classifier in action_set:
            classifier.r += self.cfg.beta * (reward + self.cfg.gamma * max_q_r - classifier.r)
            classifier.ir += self.cfg.beta * (reward - classifier.ir)
        if self.cfg.do_ga:
            apply_gaCPU3(self, action_set)
        else:
            while sum(classifier.num for classifier in action_set) > self.cfg.theta_as:
                delete_victimCPU3(self, action_set)

    def _apply_simple(self, action_set: List[ClassifierCPU3], prev_state: List[str], action: int, reward: float, next_state: List[str], is_terminal: bool = False) -> None:
        for classifier in action_set:
            classifier.exp += 1
        has_correct = any(classifier.get_anticipation(prev_state) == next_state for classifier in action_set)
        if not has_correct:
            new_classifier = ClassifierCPU3(prev_state, action, next_state, self.cfg, self.time, origin_source="simple_mode_covering", creation_episode=self.curr_ep_idx)
            self.add_to_population(new_classifier)
            action_set.append(new_classifier)
        next_match = self._match_population(self.state_to_bits(next_state))
        max_reward = max((classifier.r for classifier in next_match), default=0.0) if not is_terminal else 0.0
        for classifier in action_set:
            classifier.r += self.cfg.beta * (reward + self.cfg.gamma * max_reward - classifier.r)
            classifier.q += self.cfg.beta * ((1.0 if classifier.get_anticipation(prev_state) == next_state else 0.0) - classifier.q)

    def _apply_alp(self, action_set: List[ClassifierCPU3], prev_state: List[str], action: int, current_state: List[str]) -> None:
        did_anticipate = False
        if not self.cfg.alp_mark_only_incorrect:
            for classifier in action_set:
                self._alp_mark(classifier, prev_state)
        for classifier in action_set[:]:
            classifier.exp += 1
            if classifier.exp < 1.0 / self.cfg.beta:
                classifier.aav = (classifier.aav * (classifier.exp - 1) + (self.time - classifier.t_alp)) / classifier.exp
            else:
                classifier.aav += self.cfg.beta * ((self.time - classifier.t_alp) - classifier.aav)
            classifier.t_alp = self.time
            if classifier.get_anticipation(prev_state) == current_state:
                classifier.q += self.cfg.beta * (1 - classifier.q)
                if classifier.q > self.cfg.theta_r:
                    self._alp_expected(classifier, prev_state)
                did_anticipate = True
            else:
                classifier.q -= self.cfg.beta * classifier.q
                if self.cfg.alp_mark_only_incorrect:
                    self._alp_mark(classifier, prev_state)
                self._alp_unexpected(classifier, prev_state, current_state)
                if classifier.q < self.cfg.theta_i:
                    self.remove_from_population(classifier)
                    if classifier in action_set:
                        action_set.remove(classifier)
        if not did_anticipate:
            self._generate_alp_cover(action_set, prev_state, action, current_state)

    def _alp_expected(self, classifier: ClassifierCPU3, state: List[str]) -> None:
        if all(len(mark_set) == 0 for mark_set in classifier.M):
            return
        off_condition = list(classifier.C)
        changed = False
        for i in range(self.cfg.l_len):
            if classifier.C[i] == "#" and classifier.M[i] and any(value != state[i] for value in classifier.M[i]):
                off_condition[i] = state[i]
                changed = True
        if changed:
            child = ClassifierCPU3(off_condition, classifier.A, classifier.E, self.cfg, self.time, origin_source="alp_expected", creation_episode=self.curr_ep_idx)
            child.q = classifier.q
            child.r = classifier.r
            self.add_to_population(child)

    def _alp_mark(self, classifier: ClassifierCPU3, prev_state: List[str]) -> None:
        for i, value in enumerate(prev_state):
            classifier.M[i].add(value)

    def _alp_unexpected(self, classifier: ClassifierCPU3, prev_state: List[str], next_state: List[str]) -> None:
        new_effect = list(classifier.E)
        changed = False
        for i in range(self.cfg.l_len):
            if new_effect[i] == "#" and prev_state[i] != next_state[i]:
                new_effect[i] = next_state[i]
                changed = True
        if changed:
            child = ClassifierCPU3(classifier.C, classifier.A, new_effect, self.cfg, self.time, origin_source="alp_unexpected", creation_episode=self.curr_ep_idx)
            self.add_to_population(child)

    def _generate_alp_cover(self, action_set: List[ClassifierCPU3], prev_state: List[str], action: int, next_state: List[str]) -> None:
        condition = ["#" for _ in range(self.cfg.l_len)]
        effect = ["#" for _ in range(self.cfg.l_len)]
        for i in range(self.cfg.l_len):
            if prev_state[i] != next_state[i]:
                condition[i] = prev_state[i]
                effect[i] = next_state[i]
        classifier = ClassifierCPU3(condition, action, effect, self.cfg, self.time, origin_source="alp_covering", creation_episode=self.curr_ep_idx)
        self.add_to_population(classifier)
        action_set.append(classifier)

    def add_to_population(self, classifier: ClassifierCPU3) -> None:
        if classifier.key in self.population_dict:
            self.population_dict[classifier.key].num += 1
            return
        if self.cfg.do_subsumption:
            for existing in self.population_by_action[classifier.A]:
                if does_subsumeCPU3(existing, classifier, self.cfg):
                    existing.num += 1
                    return
        self.population_dict[classifier.key] = classifier
        self.population_by_action[classifier.A].append(classifier)

    def remove_from_population(self, classifier: ClassifierCPU3) -> None:
        if classifier.key in self.population_dict:
            del self.population_dict[classifier.key]
            if classifier in self.population_by_action[classifier.A]:
                self.population_by_action[classifier.A].remove(classifier)

    def copy(self) -> "ACS2CPU3":
        new_agent = ACS2CPU3(copy.deepcopy(self.cfg))
        for classifier in self.population_dict.values():
            classifier_copy = classifier.copy()
            new_agent.population_dict[classifier_copy.key] = classifier_copy
            new_agent.population_by_action[classifier_copy.A].append(classifier_copy)
        new_agent.time = self.time
        new_agent.curr_ep_idx = self.curr_ep_idx
        return new_agent
