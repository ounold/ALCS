import torch
from typing import List, Tuple, Optional, Dict

class GridEnvironmentGPU:
    """
    A vectorized grid world environment for ACS2 agents using PyTorch.
    Handles N_exp environments in parallel.
    """
    def __init__(self, n_exp: int, rows: int, cols: int, 
                 start_pos: Tuple[int, int], goal_pos: Tuple[int, int], 
                 obstacles: List[Tuple[int, int]], device: str = 'cuda'):
        
        self.n_exp = n_exp
        self.rows = rows
        self.cols = cols
        self.device = device
        
        self.start_pos = torch.tensor(start_pos, device=device, dtype=torch.long)
        self.goal_pos = torch.tensor(goal_pos, device=device, dtype=torch.long)
        
        # Obstacles as a 2D boolean mask
        self.obs_mask = torch.zeros((rows, cols), device=device, dtype=torch.bool)
        for r, c in obstacles:
            self.obs_mask[r, c] = True
            
        self.current_pos = torch.zeros((n_exp, 2), device=device, dtype=torch.long)
        
        self.num_actions = 8
        # Action map as a tensor (dy, dx)
        self.action_deltas = torch.tensor([
            [0, -1], [0, 1], [-1, 0], [1, 0],   # 0:L, 1:R, 2:U, 3:D
            [-1, -1], [-1, 1], [1, -1], [1, 1]  # 4:UL, 5:UR, 6:DL, 7:DR
        ], device=device, dtype=torch.long)

    def reset_all(self) -> torch.Tensor:
        """
        Resets all environments to the start position.
        Returns the batch of states as a tensor (N_exp, 2).
        """
        self.current_pos[:] = self.start_pos
        return self.current_pos.clone()

    def reset_random(self) -> torch.Tensor:
        """
        Resets all environments to random valid positions.
        """
        # Simple rejection sampling or just find all valid indices and pick
        valid_indices = torch.where(~self.obs_mask & (torch.arange(self.rows * self.cols, device=self.device).view(self.rows, self.cols) != (self.goal_pos[0] * self.cols + self.goal_pos[1])))
        valid_coords = torch.stack(valid_indices, dim=1) # (N_valid, 2)
        
        idx = torch.randint(0, len(valid_coords), (self.n_exp,), device=self.device)
        self.current_pos = valid_coords[idx]
        return self.current_pos.clone()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a step for all environments.
        actions: (N_exp,) tensor
        returns: (next_states, rewards, dones)
        """
        deltas = self.action_deltas[actions] # (N_exp, 2)
        new_pos = self.current_pos + deltas
        
        # Bounds check
        in_bounds = (new_pos[:, 0] >= 0) & (new_pos[:, 0] < self.rows) & \
                    (new_pos[:, 1] >= 0) & (new_pos[:, 1] < self.cols)
        
        # Obstacle check
        # We need to mask out-of-bounds indices before checking obs_mask to avoid crash
        valid_new_pos = torch.where(in_bounds.unsqueeze(1), new_pos, self.current_pos)
        no_obs = ~self.obs_mask[valid_new_pos[:, 0], valid_new_pos[:, 1]]
        
        can_move = in_bounds & no_obs
        
        self.current_pos = torch.where(can_move.unsqueeze(1), new_pos, self.current_pos)
        
        # Check goal
        dones = torch.all(self.current_pos == self.goal_pos, dim=1)
        rewards = torch.where(dones, torch.tensor(1000.0, device=self.device), torch.tensor(-1.0, device=self.device))
        
        return self.current_pos.clone(), rewards, dones

    def peek_step(self, pos: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes the next state without moving.
        pos: (N, 2)
        actions: (N,)
        """
        deltas = self.action_deltas[actions]
        new_pos = pos + deltas
        in_bounds = (new_pos[:, 0] >= 0) & (new_pos[:, 0] < self.rows) & \
                    (new_pos[:, 1] >= 0) & (new_pos[:, 1] < self.cols)
        
        valid_new_pos = torch.where(in_bounds.unsqueeze(1), new_pos, pos)
        no_obs = ~self.obs_mask[valid_new_pos[:, 0], valid_new_pos[:, 1]]
        
        can_move = in_bounds & no_obs
        return torch.where(can_move.unsqueeze(1), new_pos, pos)
