import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from .conf import ACS2Configuration

class ACS2GPU:
    """
    Vectorized ACS2 implementation using PyTorch.
    Manages N_exp agents in parallel.
    """
    def __init__(self, cfg: ACS2Configuration, n_exp: int, max_pop: int = 1000, device: str = 'cuda'):
        self.cfg = cfg
        self.n_exp = n_exp
        self.max_pop = max_pop
        self.device = device
        self.l_len = 2 

        # Population Tensors
        self.C = torch.full((n_exp, max_pop, self.l_len), -1, device=device, dtype=torch.long)
        self.A = torch.zeros((n_exp, max_pop), device=device, dtype=torch.long)
        self.E = torch.full((n_exp, max_pop, self.l_len), -1, device=device, dtype=torch.long)
        
        self.q = torch.zeros((n_exp, max_pop), device=device, dtype=torch.float32)
        self.r = torch.zeros((n_exp, max_pop), device=device, dtype=torch.float32)
        self.ir = torch.zeros((n_exp, max_pop), device=device, dtype=torch.float32)
        
        self.exp = torch.zeros((n_exp, max_pop), device=device, dtype=torch.long)
        self.num = torch.zeros((n_exp, max_pop), device=device, dtype=torch.long)
        
        self.t_ga = torch.zeros((n_exp, max_pop), device=device, dtype=torch.long)
        self.t_alp = torch.zeros((n_exp, max_pop), device=device, dtype=torch.long)
        self.aav = torch.zeros((n_exp, max_pop), device=device, dtype=torch.float32)
        
        self.origin_source = torch.zeros((n_exp, max_pop), device=device, dtype=torch.long)
        self.creation_episode = torch.zeros((n_exp, max_pop), device=device, dtype=torch.long)

        self.M = torch.zeros((n_exp, max_pop, self.l_len, 7), device=device, dtype=torch.bool)
        self.active_mask = torch.zeros((n_exp, max_pop), device=device, dtype=torch.bool)
        self.time = torch.zeros(n_exp, device=device, dtype=torch.long)
        self.curr_ep_idx = 0

        self.origin_map = {
            "covering": 0, "alp_unexpected": 1, "alp_expected": 2, "alp_covering": 3, "ga": 4, "simple_mode_covering": 5, "unknown": 6
        }
        self.rev_origin_map = {v: k for k, v in self.origin_map.items()}

    def get_match_mask(self, states: torch.Tensor) -> torch.Tensor:
        match_cond = (self.C == states.unsqueeze(1)) | (self.C == -1)
        return torch.all(match_cond, dim=2) & self.active_mask

    def run_step(self, states: torch.Tensor, explore: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        match_mask = self.get_match_mask(states)
        no_match = ~torch.any(match_mask, dim=1)
        if torch.any(no_match):
            self._generate_covering(states, no_match)
            match_mask = self.get_match_mask(states)

        fitness = self.q * self.r
        
        # Fully vectorized action fitness calculation
        # fitness: (n_exp, max_pop), self.A: (n_exp, max_pop)
        # We want action_fitness: (n_exp, num_actions)
        
        # Use a large negative value for non-matching classifiers
        masked_fitness = torch.where(match_mask, fitness, torch.tensor(-1e9, device=self.device))
        
        # One-hot actions to use as a mask
        # a_mask: (n_exp, max_pop, num_actions)
        a_mask = F.one_hot(self.A, num_classes=self.cfg.num_actions).bool()
        
        # Expanded masked fitness: (n_exp, max_pop, num_actions)
        exp_fitness = masked_fitness.unsqueeze(2).expand(-1, -1, self.cfg.num_actions)
        
        # Final action fitness: (n_exp, num_actions)
        # Using a huge negative value for non-existent actions
        action_fitness, _ = torch.max(torch.where(a_mask, exp_fitness, torch.tensor(-1e9, device=self.device)), dim=1)

        best_actions = torch.argmax(action_fitness, dim=1)
        rand_actions_any = torch.randint(0, self.cfg.num_actions, (self.n_exp,), device=self.device)
        has_rules_mask = action_fitness > -1e8
        
        # Fully vectorized picking known actions
        any_known = torch.any(has_rules_mask, dim=1)
        probs = has_rules_mask.float()
        # For agents with no rules, multinomial would fail, so we give them uniform probs and then ignore the result
        safe_probs = torch.where(any_known.unsqueeze(1), probs, torch.ones_like(probs))
        rand_actions_known = torch.multinomial(safe_probs, 1).squeeze(1)
        rand_actions_known = torch.where(any_known, rand_actions_known, rand_actions_any)

        do_pick_known = torch.rand(self.n_exp, device=self.device) < 0.5
        rand_actions = torch.where(do_pick_known, rand_actions_known, rand_actions_any)
        
        if explore:
            do_explore = torch.rand(self.n_exp, device=self.device) < self.cfg.epsilon
            actions = torch.where(do_explore, rand_actions, best_actions)
        else:
            actions = best_actions

        action_mask = match_mask & (self.A == actions.unsqueeze(1))
        no_action_match = ~torch.any(action_mask, dim=1)
        if torch.any(no_action_match):
            self._generate_covering_for_action(states, actions, no_action_match)
            action_mask = self.get_match_mask(states) & (self.A == actions.unsqueeze(1))

        return actions, action_mask

    def apply_learning(self, action_mask: torch.Tensor, prev_states: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, curr_ep_idx: int, active_indices: torch.Tensor):
        self.curr_ep_idx = curr_ep_idx
        self.time[active_indices] += 1

        import time
        t_start = time.time()
        cand_list = []
        if self.cfg.do_alp:
            cand_list.extend(self._apply_alp_vectorized(action_mask, prev_states, actions, next_states))
        t_alp = time.time()

        next_match_mask = self.get_match_mask(next_states)
        fitness = self.q * self.r
        next_fit = torch.where(next_match_mask, fitness, torch.tensor(0.0, device=self.device))
        max_q_r, _ = torch.max(next_fit, dim=1)
        max_q_r = torch.where(dones, torch.tensor(0.0, device=self.device), max_q_r)

        target_r = rewards + self.cfg.gamma * max_q_r
        self.r = torch.where(action_mask, self.r + self.cfg.beta * (target_r.unsqueeze(1) - self.r), self.r)
        self.ir = torch.where(action_mask, self.ir + self.cfg.beta * (rewards.unsqueeze(1) - self.ir), self.ir)
        t_update = time.time()

        if self.cfg.do_ga:
            cand_list.extend(self._apply_ga_vectorized(action_mask))
        t_ga = time.time()

        if cand_list:
            self.batch_add_classifiers(cand_list)
        t_batch = time.time()

        self._control_action_set_size_vectorized(action_mask)
        t_control = time.time()
        
        if curr_ep_idx % 10 == 0:
           print(f"  AL Stats: ALP:{t_alp-t_start:.4f} Upd:{t_update-t_alp:.4f} GA:{t_ga-t_update:.4f} Batch:{t_batch-t_ga:.4f} Ctrl:{t_control-t_batch:.4f}")

    def batch_add_classifiers(self, cand_list):
        if not cand_list: return
        
        c_C = torch.cat([c[0] for c in cand_list], dim=0)
        c_A = torch.cat([c[1] for c in cand_list], dim=0)
        c_E = torch.cat([c[2] for c in cand_list], dim=0)
        c_q = torch.cat([c[3] for c in cand_list], dim=0)
        c_r = torch.cat([c[4] for c in cand_list], dim=0)
        c_orig = torch.cat([c[5] for c in cand_list], dim=0)
        c_ag_idx = torch.cat([c[6] for c in cand_list], dim=0)
        
        if c_A.shape[0] == 0: return

        # Fully vectorized Identity Check
        target_C = self.C[c_ag_idx]
        target_A = self.A[c_ag_idx]
        target_E = self.E[c_ag_idx]
        target_mask = self.active_mask[c_ag_idx]
        
        ident = torch.all(c_C.unsqueeze(1) == target_C, dim=-1) & \
                (c_A.unsqueeze(1) == target_A) & \
                torch.all(c_E.unsqueeze(1) == target_E, dim=-1) & \
                target_mask
        
        already_in = torch.any(ident, dim=1)
        if torch.any(already_in):
            # pop_idxs: index of existing classifier for each candidate that is 'already_in'
            pop_idxs = torch.argmax(ident[already_in].long(), dim=1)
            ag_ids = c_ag_idx[already_in]
            self.num[ag_ids, pop_idxs] += 1
            
        to_add_mask = ~already_in
        
        if self.cfg.do_subsumption and torch.any(to_add_mask):
            ccc_C, ccc_A, ccc_E, ccc_ag = c_C[to_add_mask], c_A[to_add_mask], c_E[to_add_mask], c_ag_idx[to_add_mask]
            
            s_C, s_A, s_E = self.C[ccc_ag], self.A[ccc_ag], self.E[ccc_ag]
            s_q, s_exp, s_mask = self.q[ccc_ag], self.exp[ccc_ag], self.active_mask[ccc_ag]
            s_M = self.M[ccc_ag]
            
            sub_A_E = (s_A == ccc_A.unsqueeze(1)) & torch.all(s_E == ccc_E.unsqueeze(1), dim=-1)
            sub_exp_q = (s_q > self.cfg.theta_r) & (s_exp > self.cfg.theta_exp)
            sub_no_M = torch.sum(s_M, dim=(2, 3)) == 0
            consistent = torch.all((s_C == -1) | (s_C == ccc_C.unsqueeze(1)), dim=-1)
            more_general = torch.any((s_C == -1) & (ccc_C.unsqueeze(1) != -1), dim=-1)
            
            subsumer_exists_mask = sub_A_E & sub_exp_q & sub_no_M & consistent & more_general & s_mask
            subsumed = torch.any(subsumer_exists_mask, dim=1)
            
            if torch.any(subsumed):
                sub_pop_idxs = torch.argmax(subsumer_exists_mask[subsumed].long(), dim=1)
                sub_ag_ids = ccc_ag[subsumed]
                self.num[sub_ag_ids, sub_pop_idxs] += 1
                
                # Update to_add_mask to exclude subsumed candidates
                temp_idxs = torch.where(to_add_mask)[0]
                to_add_mask[temp_idxs[subsumed]] = False

        if torch.any(to_add_mask):
            final_cands_idx = torch.where(to_add_mask)[0]
            uc_ag, uc_inv = c_ag_idx[final_cands_idx].unique(return_inverse=True)
            num_new_per_agent = torch.zeros_like(uc_ag, dtype=torch.long).scatter_add_(0, uc_inv, torch.ones_like(uc_inv, dtype=torch.long))

            free_counts = torch.sum(~self.active_mask[uc_ag], dim=1)
            needs_deletion_mask = free_counts < num_new_per_agent
            
            if torch.any(needs_deletion_mask):
                del_ag_idx = uc_ag[needs_deletion_mask]
                needed = num_new_per_agent[needs_deletion_mask] - free_counts[needs_deletion_mask]
                
                # This part is tricky to fully vectorize efficiently without a loop if `needed` varies per agent.
                # A loop over the small number of agents needing deletion is a reasonable compromise.
                for i, ag_idx in enumerate(del_ag_idx):
                    q_vals = torch.where(self.active_mask[ag_idx], self.q[ag_idx], torch.tensor(1e9, device=self.device))
                    _, v_idxs = torch.topk(q_vals, needed[i].item(), largest=False)
                    self.active_mask[ag_idx, v_idxs] = False
            
            # Re-calculate free slots after potential deletion
            free_slots_per_agent = torch.where(~self.active_mask[uc_ag], torch.arange(self.max_pop, device=self.device).unsqueeze(0), self.max_pop)
            sorted_free_slots, _ = torch.sort(free_slots_per_agent, dim=1)

            # Assign new classifiers to free slots
            max_new = num_new_per_agent.max()
            slot_indices = torch.arange(max_new, device=self.device)
            target_slots = sorted_free_slots[torch.arange(len(uc_ag)).unsqueeze(1), slot_indices]
            
            # Create a flat list of (agent_idx, slot_idx) for scatter updates
            # This part still requires careful indexing, a loop might be clearer and not a huge bottleneck if `uc_ag` is small
            # For now, we'll keep the final loop for clarity and correctness, as full vectorization here is complex.
            involved_agents = c_ag_idx[final_cands_idx].unique()
            for ag in involved_agents:
                ag_c_mask = (c_ag_idx[final_cands_idx] == ag)
                rem_idxs = final_cands_idx[ag_c_mask]
                num_new = len(rem_idxs)
                
                free_slots = torch.where(~self.active_mask[ag])[0]
                # This deletion logic should be rare, but we keep it for correctness
                if len(free_slots) < num_new:
                    needed = num_new - len(free_slots)
                    q_vals = torch.where(self.active_mask[ag], self.q[ag], torch.tensor(1e9, device=self.device))
                    _, v_idxs = torch.topk(q_vals, needed, largest=False)
                    self.active_mask[ag, v_idxs] = False
                    free_slots = torch.where(~self.active_mask[ag])[0]

                target_slots = free_slots[:num_new]
                self.C[ag, target_slots] = c_C[rem_idxs]
                self.A[ag, target_slots] = c_A[rem_idxs]
                self.E[ag, target_slots] = c_E[rem_idxs]
                self.q[ag, target_slots] = c_q[rem_idxs]
                self.r[ag, target_slots] = c_r[rem_idxs]
                self.ir[ag, target_slots] = 0.0
                self.exp[ag, target_slots] = 0
                self.num[ag, target_slots] = 1
                self.M[ag, target_slots] = False
                self.active_mask[ag, target_slots] = True
                self.origin_source[ag, target_slots] = c_orig[rem_idxs]
                self.creation_episode[ag, target_slots] = self.curr_ep_idx
                self.t_ga[ag, target_slots] = self.time[ag]
                self.t_alp[ag, target_slots] = self.time[ag]

    def _generate_covering(self, states: torch.Tensor, mask: torch.Tensor):
        indices = torch.where(mask)[0]
        if len(indices) == 0: return
        actions = torch.randint(0, self.cfg.num_actions, (len(indices),), device=self.device)
        conds = torch.full((len(indices), self.l_len), -1, device=self.device, dtype=torch.long)
        
        u_idxs = torch.randint(0, self.l_len, (len(indices),), device=self.device)
        row_idx = torch.arange(len(indices), device=self.device)
        conds[row_idx, u_idxs] = states[indices, u_idxs]
        
        self.batch_add_classifiers([(conds, actions, 
                                     torch.full((len(indices), self.l_len), -1, device=self.device, dtype=torch.long),
                                     torch.full((len(indices),), 0.5, device=self.device),
                                     torch.full((len(indices),), 0.5, device=self.device),
                                     torch.full((len(indices),), self.origin_map["covering"], device=self.device, dtype=torch.long),
                                     indices)])

    def _generate_covering_for_action(self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        indices = torch.where(mask)[0]
        if len(indices) == 0: return
        conds = torch.full((len(indices), self.l_len), -1, device=self.device, dtype=torch.long)
        
        u_idxs = torch.randint(0, self.l_len, (len(indices),), device=self.device)
        row_idx = torch.arange(len(indices), device=self.device)
        conds[row_idx, u_idxs] = states[indices, u_idxs]
        
        self.batch_add_classifiers([(conds, actions[indices], 
                                     torch.full((len(indices), self.l_len), -1, device=self.device, dtype=torch.long),
                                     torch.full((len(indices),), 0.5, device=self.device),
                                     torch.full((len(indices),), 0.5, device=self.device),
                                     torch.full((len(indices),), self.origin_map["covering"], device=self.device, dtype=torch.long),
                                     indices)])

    def _apply_alp_vectorized(self, action_mask, prev_states, actions, next_states):
        expected_next = torch.where(self.E == -1, prev_states.unsqueeze(1), self.E)
        is_correct = torch.all(expected_next == next_states.unsqueeze(1), dim=2)
        correct_mask, incorrect_mask = action_mask & is_correct, action_mask & (~is_correct)
        
        self.exp = torch.where(action_mask, self.exp + 1, self.exp)
        self.aav = torch.where(action_mask, self.aav + self.cfg.beta * ((self.time.unsqueeze(1) - self.t_alp) - self.aav), self.aav)
        self.t_alp = torch.where(action_mask, self.time.unsqueeze(1), self.t_alp)
        self.q = torch.where(correct_mask, self.q + self.cfg.beta * (1.0 - self.q), self.q)
        self.q = torch.where(incorrect_mask, self.q - self.cfg.beta * self.q, self.q)

        agent_idxs, cl_idxs = torch.where(action_mask)
        if len(agent_idxs) > 0:
            # Narrowing down to only the relevant states for updating M
            for k in range(self.l_len):
                self.M[agent_idxs, cl_idxs, k, prev_states[agent_idxs, k]] = True

        res_cands = []
        high_q_correct = correct_mask & (self.q > self.cfg.theta_r)
        if torch.any(high_q_correct):
            ag_idx, cl_idx = torch.where(high_q_correct)
            # Only process the classifiers that are high_q and correct
            m_subset = self.M[ag_idx, cl_idx] # (num_high_q, l_len, 7)
            ps_subset = prev_states[ag_idx] # (num_high_q, l_len)
            ps_oh_subset = F.one_hot(ps_subset, 7).bool() # (num_high_q, l_len, 7)
            has_other_subset = torch.any(m_subset & (~ps_oh_subset), dim=-1) # (num_high_q, l_len)
            
            # check (self.C[ag_idx, cl_idx] == -1) & has_other_subset
            spec_needed_subset = (self.C[ag_idx, cl_idx] == -1) & has_other_subset
            if torch.any(spec_needed_subset):
                # indices in subset that need specialization
                sub_idxs = torch.where(torch.any(spec_needed_subset, dim=1))[0]
                
                final_ag_idx = ag_idx[sub_idxs]
                final_cl_idx = cl_idx[sub_idxs]
                
                new_C = self.C[final_ag_idx, final_cl_idx].clone()
                # Apply specialization only where needed
                s_mask = spec_needed_subset[sub_idxs]
                new_C[s_mask] = ps_subset[sub_idxs][s_mask]
                
                res_cands.append((new_C, self.A[final_ag_idx, final_cl_idx], self.E[final_ag_idx, final_cl_idx], 
                                  self.q[final_ag_idx, final_cl_idx], self.r[final_ag_idx, final_cl_idx],
                                  torch.full((len(final_ag_idx),), self.origin_map["alp_expected"], device=self.device, dtype=torch.long), final_ag_idx))

        if torch.any(incorrect_mask):
            diff = (prev_states.unsqueeze(1) != next_states.unsqueeze(1))
            can_fix = (self.E == -1) & diff
            fix_mask = incorrect_mask & torch.any(can_fix, dim=2)
            if torch.any(fix_mask):
                ag_idx, cl_idx = torch.where(fix_mask)
                new_E = self.E[ag_idx, cl_idx].clone()
                
                # We need to apply next_states where can_fix is true for these specific classifiers
                c_fix = can_fix[ag_idx, cl_idx] # (num_fix, l_len)
                ns_vals = next_states[ag_idx] # (num_fix, l_len)
                new_E[c_fix] = ns_vals[c_fix]
                
                res_cands.append((self.C[ag_idx, cl_idx], self.A[ag_idx, cl_idx], new_E, 
                                  torch.full((len(ag_idx),), 0.5, device=self.device),
                                  torch.full((len(ag_idx),), 0.5, device=self.device), 
                                  torch.full((len(ag_idx),), self.origin_map["alp_unexpected"], device=self.device, dtype=torch.long), ag_idx))

        did_ant = torch.any(correct_mask, dim=1)
        no_ant = ~did_ant
        if torch.any(no_ant):
            diff = prev_states[no_ant] != next_states[no_ant]
            new_C = torch.where(diff, prev_states[no_ant], torch.tensor(-1, device=self.device))
            new_E = torch.where(diff, next_states[no_ant], torch.tensor(-1, device=self.device))
            ag_idx = torch.where(no_ant)[0]
            res_cands.append((new_C, actions[no_ant], new_E, torch.full((len(ag_idx),), 0.5, device=self.device),
                              torch.full((len(ag_idx),), 0.5, device=self.device), torch.full((len(ag_idx),), self.origin_map["alp_covering"], device=self.device, dtype=torch.long), ag_idx))
        
        self.active_mask &= ~(self.active_mask & (self.q < self.cfg.theta_i))
        return res_cands

    def _apply_ga_vectorized(self, action_mask):
        total_num = torch.sum(self.num * action_mask.long(), dim=1)
        avg_t = torch.sum(self.t_ga * self.num * action_mask.long(), dim=1) / total_num.clamp(min=1)
        do_ga = (total_num > 0) & (self.time - avg_t > self.cfg.theta_ga)
        indices = torch.where(do_ga)[0]
        if len(indices) == 0: return []
        
        ga_mask = action_mask & do_ga.unsqueeze(1)
        if torch.any(ga_mask):
            self.t_ga[ga_mask] = self.time.unsqueeze(1).expand(-1, self.max_pop)[ga_mask]
        
        m_all = action_mask[indices] & self.active_mask[indices]
        can_do = (torch.sum(m_all.long(), dim=1) >= 2)
        if not torch.any(can_do): return []
        
        idx_ga = indices[can_do]
        m_ga = action_mask[idx_ga] & self.active_mask[idx_ga]
        probs = (self.q[idx_ga] ** 3) * m_ga.float()
        probs[m_ga] += 1e-9
        
        p_idx = torch.multinomial(probs, 2)
        p1_C, p2_C = self.C[idx_ga, p_idx[:, 0]], self.C[idx_ga, p_idx[:, 1]]
        p1_A, p1_E = self.A[idx_ga, p_idx[:, 0]], self.E[idx_ga, p_idx[:, 0]]
        
        do_chi = torch.rand(len(idx_ga), device=self.device) < self.cfg.chi
        if torch.any(do_chi):
            pt = torch.randint(0, self.l_len, (torch.sum(do_chi),), device=self.device)
            swap_mask = (pt == 1)
            sw_idx = idx_ga[do_chi][swap_mask]
            if len(sw_idx) > 0:
                p1_C[do_chi][swap_mask, 1], p2_C[do_chi][swap_mask, 1] = p2_C[do_chi][swap_mask, 1], p1_C[do_chi][swap_mask, 1]
        
        p1_C[torch.rand(p1_C.shape, device=self.device) < self.cfg.mu] = -1
        p2_C[torch.rand(p2_C.shape, device=self.device) < self.cfg.mu] = -1
        
        cat_C = torch.cat([p1_C, p2_C], dim=0)
        cat_A = torch.cat([p1_A, p1_A], dim=0)
        cat_E = torch.cat([p1_E, p1_E], dim=0)
        cat_ag = torch.cat([idx_ga, idx_ga], dim=0)
        
        return [(cat_C, cat_A, cat_E, 
                 torch.full((len(cat_ag),), 0.5, device=self.device),
                 torch.full((len(cat_ag),), 0.5, device=self.device),
                 torch.full((len(cat_ag),), self.origin_map["ga"], device=self.device, dtype=torch.long),
                 cat_ag)]

    def _control_action_set_size_vectorized(self, action_mask: torch.Tensor):
        pop_counts = torch.sum(self.num * action_mask.long(), dim=1)
        indices = torch.where(pop_counts > self.cfg.theta_as)[0]
        if len(indices) == 0: return
        
        for idx in indices:
            m = action_mask[idx] & self.active_mask[idx]
            num_in_set = torch.sum(self.num[idx, m]).item()
            if num_in_set <= self.cfg.theta_as: continue
            
            q_vals = torch.where(m, self.q[idx], torch.tensor(1e9, device=self.device))
            _, sorted_idxs = torch.sort(q_vals)
            set_idxs = sorted_idxs[:torch.sum(m)]
            nums = self.num[idx, set_idxs]
            cum_nums = torch.cumsum(nums, dim=0)
            
            to_remove = num_in_set - self.cfg.theta_as
            delete_mask = (cum_nums <= to_remove)
            self.active_mask[idx, set_idxs[delete_mask]] = False
            
            rem = to_remove - torch.sum(nums[delete_mask]).item()
            if rem > 0:
                first_not_del = torch.where(~delete_mask)[0][0]
                self.num[idx, set_idxs[first_not_del]] -= rem

    def get_population_stats(self):
        return [(torch.sum(self.num[i, self.active_mask[i]]).item(), torch.sum(self.active_mask[i]).item()) for i in range(self.n_exp)]
