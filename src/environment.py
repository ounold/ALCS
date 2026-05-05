import random
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

class GridEnvironment:
    """
    A simple grid world environment for the ACS2 agent.
    Can be initialized with explicit dimensions and positions, or by loading from a map file.
    """
    def __init__(self, rows: Optional[int] = None, cols: Optional[int] = None, 
                 start_pos: Optional[Tuple[int, int]] = None, goal_pos: Optional[Tuple[int, int]] = None, 
                 obstacles: Optional[List[Tuple[int, int]]] = None, map_file: Optional[str] = None):
        
        if map_file:
            self._load_from_map_file(map_file)
        else:
            if any(arg is None for arg in [rows, cols, start_pos, goal_pos]):
                raise ValueError("When not using a map_file, rows, cols, start_pos, and goal_pos must be provided.")
            self.rows: int = rows
            self.cols: int = cols
            self.start_pos: Tuple[int, int] = start_pos
            self.goal_pos: Tuple[int, int] = goal_pos
            self.obstacles: List[Tuple[int, int]] = obstacles if obstacles else []
        
        self.current_pos: List[int] = list(self.start_pos)
        self.num_actions: int = 8
        self.action_map: Dict[int, Tuple[int, int]] = {
            0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0),
            4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)
        }

    def _load_from_map_file(self, map_file: str):
        with open(map_file, 'r') as f:
            lines = f.readlines()
        
        self.rows = len(lines)
        self.cols = len(lines[0].strip())
        self.obstacles = []
        
        for r, line in enumerate(lines):
            for c, char in enumerate(line.strip()):
                if char == 'S':
                    self.start_pos = (r, c)
                elif char == 'G':
                    self.goal_pos = (r, c)
                elif char == '1': # Assuming '1' means obstacle from the web search result
                    self.obstacles.append((r, c))
                # Assuming '0' means path, no need to store

        if not hasattr(self, 'start_pos'):
            raise ValueError("Start position 'S' not found in map file.")
        if not hasattr(self, 'goal_pos'):
            raise ValueError("Goal position 'G' not found in map file.")

    def reset(self) -> List[str]:
        """
        Resets the agent to a random valid starting position.
        """
        while True:
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (r, c) not in self.obstacles and (r, c) != self.goal_pos:
                self.current_pos = [r, c]
                break
        return [str(self.current_pos[0]), str(self.current_pos[1])]

    def step(self, action: int) -> Tuple[List[str], float, bool]:
        """
        Executes a single step in the environment.
        """
        state, _, _ = self._compute_step(self.current_pos, action)
        self.current_pos = [int(state[0]), int(state[1])]
        is_goal = (self.current_pos[0], self.current_pos[1]) == self.goal_pos
        return state, 1000.0 if is_goal else -1.0, is_goal

    def peek_step(self, pos: List[int], action: int) -> List[str]:
        """
        Computes the next state without actually moving the agent.
        """
        s, _, _ = self._compute_step(pos, action)
        return s

    def _compute_step(self, pos: List[int], action: int) -> Tuple[List[str], float, bool]:
        """
        Internal helper to compute the result of a move.
        """
        move = self.action_map.get(action)
        if not move:
            return [str(pos[0]), str(pos[1])], -1.0, False
        nr, nc = pos[0] + move[0], pos[1] + move[1]

        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.obstacles:
            final_pos = [nr, nc]
        else:
            final_pos = list(pos)
        
        is_goal = (final_pos[0], final_pos[1]) == self.goal_pos
        reward = 1000.0 if is_goal else -1.0
        
        return [str(final_pos[0]), str(final_pos[1])], reward, is_goal

def calculate_optimal_metrics(env: GridEnvironment) -> float:
    """
    Calculates the average number of steps in the optimal path from all valid starting points to the goal
    using Breadth-First Search (BFS).
    """
    valid_starts = []
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) != env.goal_pos and (r, c) not in env.obstacles:
                valid_starts.append((r, c))

    if not valid_starts:
        return 0.0

    total_steps = 0
    moves = list(env.action_map.values())

    for start_node in valid_starts:
        queue = deque([(start_node, 0)])
        visited = {start_node}
        found = False

        while queue:
            (curr_r, curr_c), dist = queue.popleft()

            if (curr_r, curr_c) == env.goal_pos:
                total_steps += dist
                found = True
                break

            for dr, dc in moves:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < env.rows and 0 <= nc < env.cols and (nr, nc) not in visited and (nr, nc) not in env.obstacles:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        
        if not found:
            # This case should ideally not happen in a solvable maze.
            # We can either assign a penalty or ignore this start position.
            pass

    return total_steps / len(valid_starts) if valid_starts else 0.0
