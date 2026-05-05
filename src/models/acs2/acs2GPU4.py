from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F

from .confGPU4 import ACS2ConfigurationGPU4


@dataclass(frozen=True)
class AgentSelectionGPU4:
    agent: "ACS2GPU4"
    experiment_index: int


class ACS2GPU4:
    def __init__(self, cfg: ACS2ConfigurationGPU4, n_exp: int, device: str = "cpu"):
        self.cfg = cfg
        self.n_exp = n_exp
        self.device = torch.device(device)
        self.l_len = cfg.l_len
        self.symbol_capacity = cfg.symbol_capacity
        self.max_pop = cfg.max_population

        # Population Tensors: (N_exp, N_pop, ...)
        # C: Condition, -1 is wildcard
        self.C = torch.full((n_exp, self.max_pop, self.l_len), -1, dtype=torch.long, device=self.device)
        # A: Action
        self.A = torch.zeros((n_exp, self.max_pop), dtype=torch.long, device=self.device)
        # E: Effect, -1 is identity (no change predicted)
        self.E = torch.full((n_exp, self.max_pop, self.l_len), -1, dtype=torch.long, device=self.device)

        self.q = torch.full((n_exp, self.max_pop), 0.5, device=self.device)
        self.r = torch.full((n_exp, self.max_pop), 0.5, device=self.device)
        self.ir = torch.zeros((n_exp, self.max_pop), device=self.device)
        self.exp = torch.zeros((n_exp, self.max_pop), dtype=torch.long, device=self.device)
        self.num = torch.zeros((n_exp, self.max_pop), dtype=torch.long, device=self.device)
        self.aav = torch.zeros((n_exp, self.max_pop), device=self.device)
        self.t_ga = torch.zeros((n_exp, self.max_pop), dtype=torch.long, device=self.device)
        self.t_alp = torch.zeros((n_exp, self.max_pop), dtype=torch.long, device=self.device)
        # M: Marks [N_exp, N_pop, L_len, Capacity]
        self.M = torch.zeros((n_exp, self.max_pop, self.l_len, self.symbol_capacity), dtype=torch.bool, device=self.device)

        self.active_mask = torch.zeros((n_exp, self.max_pop), dtype=torch.bool, device=self.device)
        self.origin_source = torch.zeros((n_exp, self.max_pop), dtype=torch.long, device=self.device)
        self.creation_episode = torch.zeros((n_exp, self.max_pop), dtype=torch.long, device=self.device)

        self.time = torch.zeros(n_exp, dtype=torch.long, device=self.device)
        self.epsilon = 0.1
        self.beta = 0.05
        self.curr_ep_idx = 0

        self.origin_map = {"covering": 0, "alp_unexpected": 1, "alp_expected": 2, "alp_covering": 3, "ga": 4}
        self.reverse_origin_map = {v: k for k, v in self.origin_map.items()}
        self._half = torch.tensor(0.5, device=self.device)
        self._zero = torch.tensor(0.0, device=self.device)
        self.profile_stats = {
            "batch_add_classifiers_s": 0.0,
            "subsumption_s": 0.0,
            "reserve_slots_s": 0.0,
        }

    def _coalesce_candidate_batch(
        self,
        cond: torch.Tensor,
        action: torch.Tensor,
        effect: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        origin: torch.Tensor,
        agent_indices: torch.Tensor,
        candidate_num: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_candidates = action.shape[0]
        if num_candidates <= 1:
            return cond, action, effect, q, r, origin, agent_indices, candidate_num

        if num_candidates <= 8:
            signature_rows = torch.cat(
                [
                    agent_indices.unsqueeze(1),
                    action.unsqueeze(1),
                    cond,
                    effect,
                ],
                dim=1,
            ).detach().cpu().tolist()
            candidate_num_list = candidate_num.detach().cpu().tolist()
            keep_indices: List[int] = []
            counts: List[int] = []
            for idx in range(num_candidates):
                duplicate_at = -1
                for kept_pos, kept_idx in enumerate(keep_indices):
                    if signature_rows[idx] == signature_rows[kept_idx]:
                        duplicate_at = kept_pos
                        break
                if duplicate_at >= 0:
                    counts[duplicate_at] += candidate_num_list[idx]
                else:
                    keep_indices.append(idx)
                    counts.append(candidate_num_list[idx])
            if len(keep_indices) == num_candidates:
                return cond, action, effect, q, r, origin, agent_indices, candidate_num
                keep = torch.tensor(keep_indices, device=self.device, dtype=torch.long)
                merged_num = torch.tensor(counts, device=self.device, dtype=torch.long)
                return (
                    cond[keep],
                    action[keep],
                effect[keep],
                q[keep],
                r[keep],
                origin[keep],
                    agent_indices[keep],
                    merged_num,
                )
        return cond, action, effect, q, r, origin, agent_indices, candidate_num

    def get_match_mask(self, states: torch.Tensor) -> torch.Tensor:
        match_cond = (self.C == states.unsqueeze(1)) | (self.C == -1)
        return torch.all(match_cond, dim=2) & self.active_mask

    def action_fitness_for_states(self, experiment_index: int, states: torch.Tensor) -> torch.Tensor:
        # states: (N_states, L_len)
        # result: (N_states, Num_actions)
        n_states = states.shape[0]
        # (1, N_pop, L_len) vs (N_states, 1, L_len)
        match_cond = (self.C[experiment_index:experiment_index+1] == states.unsqueeze(1)) | (self.C[experiment_index:experiment_index+1] == -1)
        matches = torch.all(match_cond, dim=2) & self.active_mask[experiment_index:experiment_index+1] # (N_states, N_pop)
        
        fitness = self.q[experiment_index] * self.r[experiment_index]
        # (N_states, N_pop)
        fit_expanded = fitness.unsqueeze(0).expand(n_states, -1)
        
        # (N_states, Num_actions)
        action_fit = torch.zeros((n_states, self.cfg.num_actions), device=self.device)
        action_indices = self.A[experiment_index].unsqueeze(0).expand(n_states, -1)
        weighted_fit = torch.where(matches, fit_expanded, self._zero)
        action_fit.scatter_add_(1, action_indices, weighted_fit)
        return action_fit

    def run_step(self, states: torch.Tensor, explore: bool = True, active_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if active_mask is None:
            active_mask = torch.ones((self.n_exp,), dtype=torch.bool, device=self.device)

        match_mask = self.get_match_mask(states) & active_mask.unsqueeze(1)
        missing_match = active_mask & (~torch.any(match_mask, dim=1))
        if torch.any(missing_match):
            self._generate_covering(states, missing_match)
            match_mask = self.get_match_mask(states)
            match_mask = match_mask & active_mask.unsqueeze(1)

        fitness = self.q * self.r
        # (N_exp, N_pop)
        fit_masked = torch.where(match_mask, fitness, torch.full_like(fitness, float("-inf")))

        # (N_exp, Num_actions)
        action_fitness = torch.full((self.n_exp, self.cfg.num_actions), float("-inf"), device=self.device)
        action_fitness.scatter_reduce_(1, self.A, fit_masked, reduce="amax", include_self=True)
        action_fitness = torch.where(torch.isfinite(action_fitness), action_fitness, self._zero)

        if explore:
            do_explore = torch.rand((self.n_exp,), device=self.device) < self.epsilon
            greedy_actions = torch.argmax(action_fitness, dim=1)
            random_actions = torch.randint(0, self.cfg.num_actions, (self.n_exp,), device=self.device)
            actions = torch.where(do_explore, random_actions, greedy_actions)
        else:
            actions = torch.argmax(action_fitness, dim=1)

        actions = torch.where(active_mask, actions, torch.zeros_like(actions))

        action_mask = match_mask & (self.A == actions.unsqueeze(1))
        # Handle empty action sets (should not happen with covering)
        missing_action = active_mask & (~torch.any(action_mask, dim=1))
        if torch.any(missing_action):
            self._generate_covering_for_action(states, actions, missing_action)
            action_mask = match_mask & (self.A == actions.unsqueeze(1))

        action_mask = action_mask & active_mask.unsqueeze(1)
        return actions, action_mask

    def apply_learning(self, active_mask: torch.Tensor, action_mask: torch.Tensor, prev_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> None:
        # active_mask: (N_exp,)
        # action_mask: (N_exp, N_pop)
        self.time = torch.where(active_mask, self.time + 1, self.time)
        candidate_list = []
        if self.cfg.do_alp:
            candidate_list.extend(self._apply_alp(action_mask, prev_states, actions, next_states))

        next_match_mask = self.get_match_mask(next_states)
        fitness = self.q * self.r
        next_fit = torch.where(next_match_mask, fitness, self._zero)
        max_q_r = torch.max(next_fit, dim=1).values
        max_q_r = torch.where(dones, self._zero, max_q_r)

        target_r = rewards + self.cfg.gamma * max_q_r
        self.r = torch.where(action_mask, self.r + self.cfg.beta * (target_r.unsqueeze(1) - self.r), self.r)
        self.ir = torch.where(action_mask, self.ir + self.cfg.beta * (rewards.unsqueeze(1) - self.ir), self.ir)

        if self.cfg.do_ga:
            candidate_list.extend(self._apply_ga(action_mask))
        if candidate_list:
            self.batch_add_classifiers(candidate_list, action_mask=action_mask)
        self._control_action_set_size(action_mask)

    def _apply_action_set_subsumption(self, action_mask: torch.Tensor) -> None:
        # Find potential subsumers: active, in action set, experienced, reliable
        enough_exp = self.exp > self.cfg.theta_exp
        enough_q = self.q > self.cfg.theta_r
        subsumer_candidates = action_mask & enough_exp & enough_q

        if not torch.any(subsumer_candidates):
            return

        exp_idx, s_idx = torch.where(subsumer_candidates)
        if len(exp_idx) == 0:
            return

        for e_idx, s_val in zip(exp_idx.tolist(), s_idx.tolist()):
            if not self.active_mask[e_idx, s_val]:
                continue
            
            gen_cond = self.C[e_idx, s_val]
            gen_effect = self.E[e_idx, s_val]
            others_mask = self.active_mask[e_idx] & action_mask[e_idx] & (self.q[e_idx] > self.cfg.theta_r)
            others_mask[s_val] = False 

            if not torch.any(others_mask):
                continue
                
            other_cond = self.C[e_idx]
            other_effect = self.E[e_idx]
            
            same_effect = torch.all(gen_effect == other_effect, dim=1)
            matches = torch.all((gen_cond == -1) | (gen_cond == other_cond), dim=1)
            strictly_more_general = torch.any((gen_cond == -1) & (other_cond != -1), dim=1)
            
            to_subsume = others_mask & same_effect & matches & strictly_more_general
            
            if torch.any(to_subsume):
                subsumed_indices = torch.where(to_subsume)[0]
                self.num[e_idx, s_val] += torch.sum(self.num[e_idx, subsumed_indices])
                self.active_mask[e_idx, subsumed_indices] = False


    def batch_add_classifiers(self, candidate_list, action_mask: Optional[torch.Tensor] = None) -> None:
        import time
        start_time = time.perf_counter()
        if not candidate_list:
            return
        cond = torch.cat([item[0] for item in candidate_list], dim=0)
        action = torch.cat([item[1] for item in candidate_list], dim=0)
        effect = torch.cat([item[2] for item in candidate_list], dim=0)
        q = torch.cat([item[3] for item in candidate_list], dim=0)
        r = torch.cat([item[4] for item in candidate_list], dim=0)
        origin = torch.cat([item[5] for item in candidate_list], dim=0)
        agent_indices = torch.cat([item[6] for item in candidate_list], dim=0)
        if action.numel() == 0:
            return
        candidate_num = torch.ones(action.shape[0], device=self.device, dtype=torch.long)

        # Coalesce identical candidates before touching the population. This removes
        # repeated work in the hottest insertion path and preserves numerosity.
        cond, action, effect, q, r, origin, agent_indices, candidate_num = self._coalesce_candidate_batch(
            cond,
            action,
            effect,
            q,
            r,
            origin,
            agent_indices,
            candidate_num,
        )
        if action.shape[0] > 8:
            signature = torch.cat(
                [
                    agent_indices.unsqueeze(1),
                    action.unsqueeze(1),
                    cond,
                    effect,
                ],
                dim=1,
            )
            unique_signature, inverse, counts = torch.unique(
                signature,
                dim=0,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            if unique_signature.shape[0] != signature.shape[0]:
                first_indices = torch.full(
                    (unique_signature.shape[0],),
                    signature.shape[0],
                    device=self.device,
                    dtype=torch.long,
                )
                first_indices.scatter_reduce_(
                    0,
                    inverse,
                    torch.arange(signature.shape[0], device=self.device, dtype=torch.long),
                    reduce="amin",
                    include_self=True,
                )
                keep = first_indices
                cond = cond[keep]
                action = action[keep]
                effect = effect[keep]
                q = q[keep]
                r = r[keep]
                origin = origin[keep]
                agent_indices = agent_indices[keep]
                candidate_num = counts.long()

        target_cond = self.C[agent_indices]
        target_action = self.A[agent_indices]
        target_effect = self.E[agent_indices]
        target_active = self.active_mask[agent_indices]
        identical = torch.all(cond.unsqueeze(1) == target_cond, dim=-1) & (action.unsqueeze(1) == target_action) & torch.all(effect.unsqueeze(1) == target_effect, dim=-1) & target_active
        already_in = torch.any(identical, dim=1)
        if torch.any(already_in):
            pop_agents = agent_indices[already_in]
            pop_indices = torch.argmax(identical[already_in].long(), dim=1)
            self.num[pop_agents, pop_indices] += candidate_num[already_in]

        to_add = torch.where(~already_in)[0]
        if self.cfg.do_subsumption and len(to_add) > 0:
            subsumed_mask, subsumer_slots = self._apply_subsumption_for_candidates(
                cond[to_add],
                action[to_add],
                effect[to_add],
                agent_indices[to_add],
            )
            if torch.any(subsumed_mask):
                subsumed_idx = to_add[subsumed_mask]
                subsumed_agents = agent_indices[subsumed_idx]
                chosen_slots = subsumer_slots[subsumed_mask]
                self.num[subsumed_agents, chosen_slots] += candidate_num[subsumed_idx]
            to_add = to_add[~subsumed_mask]
        if len(to_add) == 0:
            return

        sorted_agents = agent_indices[to_add]
        unique_agents, counts = torch.unique_consecutive(sorted_agents, return_counts=True)
        target_slots = self._reserve_grouped_slots_for_candidates(
            unique_agents,
            counts,
            action[to_add],
            action_mask,
        )
        self.C[sorted_agents, target_slots] = cond[to_add]
        self.A[sorted_agents, target_slots] = action[to_add]
        self.E[sorted_agents, target_slots] = effect[to_add]
        self.q[sorted_agents, target_slots] = q[to_add]
        self.r[sorted_agents, target_slots] = r[to_add]
        self.ir[sorted_agents, target_slots] = 0.0
        self.exp[sorted_agents, target_slots] = 0
        self.num[sorted_agents, target_slots] = candidate_num[to_add]
        self.M[sorted_agents, target_slots] = False
        self.active_mask[sorted_agents, target_slots] = True
        self.origin_source[sorted_agents, target_slots] = origin[to_add]
        self.creation_episode[sorted_agents, target_slots] = self.curr_ep_idx
        self.t_ga[sorted_agents, target_slots] = self.time[sorted_agents]
        self.t_alp[sorted_agents, target_slots] = self.time[sorted_agents]
        self.profile_stats["batch_add_classifiers_s"] += time.perf_counter() - start_time

    def _reserve_grouped_slots_for_candidates(
        self,
        unique_agents: torch.Tensor,
        counts: torch.Tensor,
        sorted_actions: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        reserved_groups = []
        start = 0
        agent_list = unique_agents.tolist()
        count_list = counts.tolist()
        for agent_index, count in zip(agent_list, count_list):
            local_actions = sorted_actions[start:start + count]
            start += count
            reserved_groups.append(self._reserve_slots_for_candidates(
                agent_index,
                count,
                local_actions,
                action_mask,
            ))
        return torch.cat(reserved_groups, dim=0)

    def _reserve_slots_for_candidates(
        self,
        agent_index: int,
        needed: int,
        candidate_actions: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        import time
        start_time = time.perf_counter()
        free_slots = torch.where(~self.active_mask[agent_index])[0]
        if len(free_slots) >= needed:
            self.profile_stats["reserve_slots_s"] += time.perf_counter() - start_time
            return free_slots[:needed]

        deficit = needed - len(free_slots)
        if action_mask is not None:
            same_action_actions = candidate_actions.unique()
            same_action = torch.isin(self.A[agent_index], same_action_actions)
            candidate_victims = self.active_mask[agent_index] & same_action
        else:
            candidate_victims = self.active_mask[agent_index]

        if int(torch.sum(candidate_victims).item()) < deficit:
            candidate_victims = self.active_mask[agent_index]

        victim_scores = torch.where(
            candidate_victims,
            self.q[agent_index],
            torch.full_like(self.q[agent_index], 1e9),
        )
        victim_idx = torch.topk(victim_scores, deficit, largest=False).indices
        self.active_mask[agent_index, victim_idx] = False
        self.profile_stats["reserve_slots_s"] += time.perf_counter() - start_time
        return torch.cat([free_slots, victim_idx], dim=0)[:needed]

    def _apply_subsumption_for_candidates(
        self,
        cond: torch.Tensor,
        action: torch.Tensor,
        effect: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import time
        start_time = time.perf_counter()
        subsumed = torch.zeros(cond.shape[0], device=self.device, dtype=torch.bool)
        subsumer_slots = torch.full(cond.shape[:1], -1, device=self.device, dtype=torch.long)
        if cond.numel() == 0:
            return subsumed, subsumer_slots
        ordered = torch.argsort(agent_indices, stable=True)
        sorted_agent_indices = agent_indices[ordered]
        unique_agents, counts = torch.unique_consecutive(sorted_agent_indices, return_counts=True)
        start = 0
        for agent_index, count in zip(unique_agents.tolist(), counts.tolist()):
            group_idx = ordered[start:start + count]
            start += count
            active = self.active_mask[agent_index]
            if not torch.any(active):
                continue

            pop_cond = self.C[agent_index]
            pop_effect = self.E[agent_index]
            pop_action = self.A[agent_index]
            enough_exp = self.exp[agent_index] > self.cfg.theta_exp
            enough_q = self.q[agent_index] > self.cfg.theta_r
            no_marks = ~torch.any(self.M[agent_index], dim=(1, 2))
            valid_generals = active & enough_exp & enough_q & no_marks
            if not torch.any(valid_generals):
                continue

            cand_cond = cond[group_idx]
            cand_effect = effect[group_idx]
            cand_action = action[group_idx]

            same_action = pop_action.unsqueeze(0) == cand_action.unsqueeze(1)
            same_effect = torch.all(pop_effect.unsqueeze(0) == cand_effect.unsqueeze(1), dim=2)
            gen_matches_spec = torch.all(
                (pop_cond.unsqueeze(0) == -1) | (pop_cond.unsqueeze(0) == cand_cond.unsqueeze(1)),
                dim=2,
            )
            wildcard_superset = torch.all(
                (cand_cond.unsqueeze(1) != -1) | (pop_cond.unsqueeze(0) == -1),
                dim=2,
            )
            strictly_more_general = torch.any(
                (pop_cond.unsqueeze(0) == -1) & (cand_cond.unsqueeze(1) != -1),
                dim=2,
            )

            can_subsume = (
                valid_generals.unsqueeze(0)
                & same_action
                & same_effect
                & gen_matches_spec
                & wildcard_superset
                & strictly_more_general
            )
            has_subsumer = torch.any(can_subsume, dim=1)
            if not torch.any(has_subsumer):
                continue
            chosen_local = torch.where(has_subsumer)[0]
            subsumer_indices = torch.argmax(can_subsume[chosen_local].long(), dim=1)
            subsumed[group_idx[chosen_local]] = True
            subsumer_slots[group_idx[chosen_local]] = subsumer_indices

        self.profile_stats["subsumption_s"] += time.perf_counter() - start_time
        return subsumed, subsumer_slots

    def _generate_covering(self, states: torch.Tensor, mask: torch.Tensor) -> None:
        agent_indices = torch.where(mask)[0]
        if len(agent_indices) == 0:
            return
        actions = torch.randint(0, self.cfg.num_actions, (len(agent_indices),), device=self.device)
        cond = torch.full((len(agent_indices), self.l_len), -1, device=self.device, dtype=torch.long)
        num_specified = min(self.cfg.u_max, self.l_len)
        if num_specified > 0:
            random_scores = torch.rand((len(agent_indices), self.l_len), device=self.device)
            selected = torch.topk(random_scores, k=num_specified, dim=1).indices
            row_idx = torch.arange(len(agent_indices), device=self.device).unsqueeze(1).expand(-1, num_specified)
            cond[row_idx, selected] = states[agent_indices.unsqueeze(1), selected]
        effect = torch.full((len(agent_indices), self.l_len), -1, device=self.device, dtype=torch.long)
        self.batch_add_classifiers([(cond, actions, effect, self._half.expand(len(agent_indices)), self._half.expand(len(agent_indices)), torch.full((len(agent_indices),), self.origin_map["covering"], device=self.device, dtype=torch.long), agent_indices)])

    def _generate_covering_for_action(self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> None:
        agent_indices = torch.where(mask)[0]
        if len(agent_indices) == 0:
            return
        cond = torch.full((len(agent_indices), self.l_len), -1, device=self.device, dtype=torch.long)
        num_specified = min(self.cfg.u_max, self.l_len)
        if num_specified > 0:
            random_scores = torch.rand((len(agent_indices), self.l_len), device=self.device)
            selected = torch.topk(random_scores, k=num_specified, dim=1).indices
            row_idx = torch.arange(len(agent_indices), device=self.device).unsqueeze(1).expand(-1, num_specified)
            cond[row_idx, selected] = states[agent_indices.unsqueeze(1), selected]
        effect = torch.full((len(agent_indices), self.l_len), -1, device=self.device, dtype=torch.long)
        self.batch_add_classifiers([(cond, actions[agent_indices], effect, self._half.expand(len(agent_indices)), self._half.expand(len(agent_indices)), torch.full((len(agent_indices),), self.origin_map["covering"], device=self.device, dtype=torch.long), agent_indices)])

    def _apply_alp(self, action_mask: torch.Tensor, prev_states: torch.Tensor, actions: torch.Tensor, observed_next_states: torch.Tensor):
        expected_next = torch.where(self.E == -1, prev_states.unsqueeze(1), self.E)
        is_correct = torch.all(expected_next == observed_next_states.unsqueeze(1), dim=2)
        correct_mask = action_mask & is_correct
        incorrect_mask = action_mask & (~is_correct)
        updated_exp = torch.where(action_mask, self.exp + 1, self.exp)
        delta_t = (self.time.unsqueeze(1) - self.t_alp).float()
        exact_avg_mask = action_mask & (updated_exp.float() < (1.0 / self.cfg.beta))
        exact_aav = (self.aav * (updated_exp.float() - 1.0) + delta_t) / updated_exp.float().clamp(min=1.0)
        ema_aav = self.aav + self.cfg.beta * (delta_t - self.aav)
        self.exp = updated_exp
        self.aav = torch.where(exact_avg_mask, exact_aav, self.aav)
        self.aav = torch.where(action_mask & (~exact_avg_mask), ema_aav, self.aav)
        self.t_alp = torch.where(action_mask, self.time.unsqueeze(1), self.t_alp)
        self.q = torch.where(correct_mask, self.q + self.cfg.beta * (1.0 - self.q), self.q)
        self.q = torch.where(incorrect_mask, self.q - self.cfg.beta * self.q, self.q)

        mark_source = incorrect_mask if self.cfg.alp_mark_only_incorrect else action_mask
        agent_idx, classifier_idx = torch.where(mark_source)
        if agent_idx.numel() > 0:
            feature_idx = torch.arange(self.l_len, device=self.device).unsqueeze(0).expand(agent_idx.shape[0], -1)
            state_vals = prev_states[agent_idx]
            self.M[agent_idx.unsqueeze(1), classifier_idx.unsqueeze(1), feature_idx, state_vals] = True

        candidates = []
        high_q_correct = correct_mask & (self.q > self.cfg.theta_r)
        if torch.any(high_q_correct):
            ag_idx, cl_idx = torch.where(high_q_correct)
            mark_subset = self.M[ag_idx, cl_idx]
            state_subset = prev_states[ag_idx]
            state_one_hot = F.one_hot(state_subset.clamp(min=0), self.symbol_capacity).bool()
            has_other = torch.any(mark_subset & (~state_one_hot), dim=-1)
            spec_needed = (self.C[ag_idx, cl_idx] == -1) & has_other
            chosen_mask = torch.any(spec_needed, dim=1)
            if torch.any(chosen_mask):
                final_agents = ag_idx[chosen_mask]
                final_cls = cl_idx[chosen_mask]
                new_cond = self.C[final_agents, final_cls].clone()
                chosen_spec_needed = spec_needed[chosen_mask]
                new_cond[chosen_spec_needed] = state_subset[chosen_mask][chosen_spec_needed]
                candidates.append((new_cond, self.A[final_agents, final_cls], self.E[final_agents, final_cls], self.q[final_agents, final_cls], self.r[final_agents, final_cls], torch.full((len(final_agents),), self.origin_map["alp_expected"], device=self.device, dtype=torch.long), final_agents))

        if torch.any(incorrect_mask):
            diff = prev_states.unsqueeze(1) != observed_next_states.unsqueeze(1)
            can_fix = (self.E == -1) & diff
            fix_mask = incorrect_mask & torch.any(can_fix, dim=2)
            if torch.any(fix_mask):
                ag_idx, cl_idx = torch.where(fix_mask)
                new_effect = self.E[ag_idx, cl_idx].clone()
                local_fix = can_fix[ag_idx, cl_idx]
                next_vals = observed_next_states[ag_idx]
                new_effect[local_fix] = next_vals[local_fix]
                candidates.append((self.C[ag_idx, cl_idx], self.A[ag_idx, cl_idx], new_effect, self._half.expand(len(ag_idx)), self._half.expand(len(ag_idx)), torch.full((len(ag_idx),), self.origin_map["alp_unexpected"], device=self.device, dtype=torch.long), ag_idx))

        no_anticipation = ~torch.any(correct_mask, dim=1)
        if torch.any(no_anticipation):
            ag_idx = torch.where(no_anticipation)[0]
            prev_no_ant = prev_states[ag_idx]
            next_no_ant = observed_next_states[ag_idx]
            diff = prev_no_ant != next_no_ant
            new_cond = torch.where(diff, prev_no_ant, torch.full_like(prev_no_ant, -1))
            new_effect = torch.where(diff, next_no_ant, torch.full_like(next_no_ant, -1))
            candidates.append((new_cond, actions[ag_idx], new_effect, self._half.expand(len(ag_idx)), self._half.expand(len(ag_idx)), torch.full((len(ag_idx),), self.origin_map["alp_covering"], device=self.device, dtype=torch.long), ag_idx))
        self.active_mask &= ~(incorrect_mask & (self.q < self.cfg.theta_i))
        return candidates

    def _apply_ga(self, action_mask: torch.Tensor):
        total_num = torch.sum(self.num * action_mask.long(), dim=1)
        avg_t = torch.sum(self.t_ga * self.num * action_mask.long(), dim=1) / total_num.clamp(min=1)
        do_ga = (total_num > 0) & (self.time - avg_t > self.cfg.theta_ga)
        candidates = []
        if not torch.any(do_ga):
            return candidates
        ga_agent_indices = torch.where(do_ga)[0]
        active_action_mask = action_mask[ga_agent_indices] & self.active_mask[ga_agent_indices]
        enough_rules = torch.sum(active_action_mask.long(), dim=1) >= 2
        if not torch.any(enough_rules):
            return candidates
        selected_agents = ga_agent_indices[enough_rules]
        selected_mask = action_mask[selected_agents] & self.active_mask[selected_agents]
        probs = (self.q[selected_agents] ** 3) * selected_mask.float()
        probs[selected_mask] += 1e-9
        parent_idx = torch.multinomial(probs, 2)
        cond_one = self.C[selected_agents, parent_idx[:, 0]].clone()
        cond_two = self.C[selected_agents, parent_idx[:, 1]].clone()
        action_one = self.A[selected_agents, parent_idx[:, 0]]
        action_two = self.A[selected_agents, parent_idx[:, 1]]
        effect_one = self.E[selected_agents, parent_idx[:, 0]].clone()
        effect_two = self.E[selected_agents, parent_idx[:, 1]].clone()

        do_crossover = torch.rand(len(selected_agents), device=self.device) < self.cfg.chi
        crossover_agents = torch.where(do_crossover)[0]
        if len(crossover_agents) > 0:
            split = torch.randint(0, self.l_len, (len(crossover_agents),), device=self.device)
            positions = torch.arange(self.l_len, device=self.device).unsqueeze(0)
            crossover_mask = positions >= split.unsqueeze(1)
            idx = crossover_agents.unsqueeze(1)
            first_slice = cond_one[idx, positions].clone()
            second_slice = cond_two[idx, positions].clone()
            cond_one[idx, positions] = torch.where(crossover_mask, second_slice, first_slice)
            cond_two[idx, positions] = torch.where(crossover_mask, first_slice, second_slice)

        cond_one[torch.rand(cond_one.shape, device=self.device) < self.cfg.mu] = -1
        cond_two[torch.rand(cond_two.shape, device=self.device) < self.cfg.mu] = -1
        candidates.append((torch.cat([cond_one, cond_two], dim=0), torch.cat([action_one, action_two], dim=0), torch.cat([effect_one, effect_two], dim=0), self._half.expand(len(selected_agents) * 2), self._half.expand(len(selected_agents) * 2), torch.full((len(selected_agents) * 2,), self.origin_map["ga"], device=self.device, dtype=torch.long), torch.cat([selected_agents, selected_agents], dim=0)))
        return candidates

    def _control_action_set_size(self, action_mask: torch.Tensor) -> None:
        pop_counts = torch.sum(self.num * action_mask.long(), dim=1)
        overfull_indices = torch.where(pop_counts > self.cfg.theta_as)[0]
        for idx in overfull_indices.tolist():
            local_indices = torch.where(action_mask[idx] & self.active_mask[idx])[0]
            if local_indices.numel() == 0:
                continue
            local_nums = self.num[idx, local_indices]
            num_in_set = int(local_nums.sum().item())
            if num_in_set <= self.cfg.theta_as:
                continue
            local_q = self.q[idx, local_indices]
            sorted_order = torch.argsort(local_q)
            sorted_idx = local_indices[sorted_order]
            nums = local_nums[sorted_order]
            cumulative = torch.cumsum(nums, dim=0)
            to_remove = num_in_set - self.cfg.theta_as
            delete_mask = cumulative <= to_remove
            self.active_mask[idx, sorted_idx[delete_mask]] = False
            remaining = to_remove - int(torch.sum(nums[delete_mask]).item())
            if remaining > 0:
                first_keep = torch.where(~delete_mask)[0][0]
                self.num[idx, sorted_idx[first_keep]] -= remaining

    def get_best_agent(self) -> AgentSelectionGPU4:
        # For simplicity, return the first experiment's agent wrapped in selection
        return AgentSelectionGPU4(self, 0)
