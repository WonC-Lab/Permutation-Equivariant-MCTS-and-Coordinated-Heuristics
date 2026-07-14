import numpy as np
import torch
import math

class MultiAgentNavigationEnv:
    """
    Decentralized Multi-Agent 2D Grid Navigation Environment.
    - Size: Default 13x13 grid.
    - Agents: M agents.
    - Actions: 8 discrete directions.
    - Collision Types: Static obstacles, Vertex collisions, Edge collisions.
    - Immutable States: Represented as nested tuples for MCTS compatibility.
      state = (
          agent_positions,      # ((r0, c0), (r1, c1), ...)
          goal_positions,       # ((rg0, cg0), (rg1, cg1), ...)
          obstacles_positions,  # ((ro0, co0), (ro1, co1), ...)
          active_mask           # (True, True, ...)
      )
    """
    def __init__(self, size=13, num_agents=4):
        self.size = size
        self.num_agents = num_agents
        
        # 8 action vectors: 0: Up, 1: Down, 2: Left, 3: Right, 4: Up-Left, 5: Up-Right, 6: Down-Left, 7: Down-Right
        self.action_vectors = [
            (-1, 0),  # 0: Up
            (1, 0),   # 1: Down
            (0, -1),  # 2: Left
            (0, 1),   # 3: Right
            (-1, -1), # 4: Up-Left
            (-1, 1),  # 5: Up-Right
            (1, -1),  # 6: Down-Left
            (1, 1)    # 7: Down-Right
        ]
        
        # Dynamic static obstacles scaled to grid size
        self.obstacles = set()
        orig_obs = [
            (3, 3), (3, 4), (3, 5),
            (5, 7), (6, 7), (7, 7),
            (9, 2), (9, 3), (9, 4),
            (8, 9), (9, 9), (10, 9)
        ]
        for r, c in orig_obs:
            nr = int(round(r * (self.size - 1) / 12))
            nc = int(round(c * (self.size - 1) / 12))
            self.obstacles.add((nr, nc))
        
        # Predefined starts and goals scaled to grid size
        self.default_starts = [
            (1, 1), (1, self.size - 2), (self.size - 2, 1), (self.size - 2, self.size - 2),
            (2, 2), (2, self.size - 3), (self.size - 3, 2), (self.size - 3, self.size - 3)
        ]
        self.default_goals = [
            (self.size - 2, self.size - 2), (self.size - 2, 1), (1, self.size - 2), (1, 1),
            (self.size - 3, self.size - 3), (self.size - 3, 2), (2, self.size - 3), (2, 2)
        ]
        
        self.starts = tuple(self.default_starts[:self.num_agents])
        self.goals = tuple(self.default_goals[:self.num_agents])

    def generate_initial_state(self):
        """Generates initial immutable state tuple."""
        agent_positions = self.starts
        goal_positions = self.goals
        obstacles_positions = tuple(sorted(list(self.obstacles)))
        active_mask = tuple([True] * self.num_agents)
        return (agent_positions, goal_positions, obstacles_positions, active_mask)

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def get_valid_actions(self, state, agent_idx):
        """Returns valid action indices that keep the agent inside the grid bounds."""
        agent_positions, _, _, active_mask = state
        if not active_mask[agent_idx]:
            return [0]  # If inactive, dummy action (does not move anyway)
            
        r, c = agent_positions[agent_idx]
        valid = []
        for i, (dr, dc) in enumerate(self.action_vectors):
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                valid.append(i)
        return valid

    def step(self, state, joint_action):
        """
        Executes all active agents' actions simultaneously with priority-based conflict resolution.
        Higher priority agents (lower index) move first. Lower priority agents wait if a conflict occurs.
        If waiting is also blocked, the agent crashes and is deactivated.
        Inactive agents stay at their current positions and do not cause conflicts.
        Returns: next_state, rewards, done, info
        """
        agent_positions, goal_positions, obstacles_positions, active_mask = state
        
        next_active_mask = list(active_mask)
        rewards = [0.0] * self.num_agents
        final_positions = list(agent_positions)
        
        # Process agents in priority order (0 to M-1)
        for i in range(self.num_agents):
            if not active_mask[i]:
                # Already inactive agent stays at its current position (which is its goal or crash position)
                final_positions[i] = agent_positions[i]
                continue
                
            r, c = agent_positions[i]
            dr, dc = self.action_vectors[joint_action[i]]
            pos_i = (r + dr, c + dc)
            
            # 1. Boundary check
            if not self.in_bounds(pos_i[0], pos_i[1]):
                rewards[i] = -1.0
                next_active_mask[i] = False
                final_positions[i] = agent_positions[i]
                continue
                
            # 2. Static obstacle check
            if pos_i in self.obstacles:
                rewards[i] = -1.0
                next_active_mask[i] = False
                final_positions[i] = agent_positions[i]
                continue
                
            # 3. Conflict checks with other agents
            conflict = False
            
            # A. Vertex conflict with already resolved active agents (0 to i-1)
            for j in range(i):
                if next_active_mask[j] and final_positions[j] == pos_i:
                    conflict = True
                    break
                    
            if not conflict:
                # B. Swap (edge) conflict with lower-priority active agents (i+1 to M-1)
                for j in range(i + 1, self.num_agents):
                    if active_mask[j] and pos_i == agent_positions[j]:
                        # Check agent j's proposed position
                        r_j, c_j = agent_positions[j]
                        dr_j, dc_j = self.action_vectors[joint_action[j]]
                        pos_j = (r_j + dr_j, c_j + dc_j)
                        if pos_j == agent_positions[i]:
                            conflict = True
                            break
                            
            if conflict:
                # Agent i is forced to wait in its current cell
                # Check if its current cell conflicts with any already scheduled higher-priority agent
                stay_conflict = False
                for j in range(i):
                    if next_active_mask[j] and final_positions[j] == agent_positions[i]:
                        stay_conflict = True
                        break
                if stay_conflict:
                    # Cell is claimed, agent i crashes
                    rewards[i] = -1.0
                    next_active_mask[i] = False
                    final_positions[i] = agent_positions[i]
                else:
                    # Safe to stay
                    final_positions[i] = agent_positions[i]
                    rewards[i] = -0.05
            else:
                # No conflict: accept movement
                final_positions[i] = pos_i
                
                # Goal Check
                if pos_i == goal_positions[i]:
                    rewards[i] = 1.0
                    next_active_mask[i] = False
                else:
                    rewards[i] = -0.05

        # 3. Create next state
        next_state = (
            tuple(final_positions),
            goal_positions,
            obstacles_positions,
            tuple(next_active_mask)
        )
        
        # Done if all agents are inactive (either reached goal or crashed)
        done = not any(next_active_mask)
        
        return next_state, rewards, done, {}

    def get_agent_observation(self, state, agent_idx):
        """
        Translates state into a grid tensor from the perspective of agent_idx.
        Returns a (3, size, size) tensor:
        - Channel 0: Current agent's position (1.0) and other active agents' positions (0.5)
        - Channel 1: Obstacles (1.0)
        - Channel 2: Current agent's goal (1.0) and other active agents' goals (0.5)
        """
        agent_positions, goal_positions, _, active_mask = state
        grid = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Channel 1: Obstacles
        for r, c in self.obstacles:
            grid[1, r, c] = 1.0
            
        # Channel 0: Agents
        for idx in range(self.num_agents):
            if not active_mask[idx]:
                continue
            r, c = agent_positions[idx]
            if idx == agent_idx:
                grid[0, r, c] = 1.0
            else:
                grid[0, r, c] = 0.5
                
        # Channel 2: Goals
        for idx in range(self.num_agents):
            if not active_mask[idx]:
                continue
            r, c = goal_positions[idx]
            if idx == agent_idx:
                grid[2, r, c] = 1.0
            else:
                grid[2, r, c] = 0.5
                
        return grid

    def get_joint_observation(self, state):
        """
        Returns PyTorch tensor of shape (M, 3, size, size) for all agents.
        """
        obs_list = []
        for i in range(self.num_agents):
            obs_list.append(self.get_agent_observation(state, i))
        return torch.tensor(np.stack(obs_list, axis=0))

    def get_heuristic_policy(self, state, agent_idx):
        """
        Generates a collision-aware potential field heuristic policy.
        Attracts the agent to the goal while applying repulsive forces away from obstacles and other agents.
        """
        agent_positions, goal_positions, _, active_mask = state
        if not active_mask[agent_idx]:
            # Inactive agent: output dummy uniform policy
            return np.ones(8) / 8.0
            
        r, c = agent_positions[agent_idx]
        gr, gc = goal_positions[agent_idx]
        
        scores = []
        for i, (dr, dc) in enumerate(self.action_vectors):
            nr, nc = r + dr, c + dc
            
            # Check boundaries and static obstacles
            if not self.in_bounds(nr, nc) or (nr, nc) in self.obstacles:
                scores.append(-9999.0)
                continue
                
            # Attraction to Goal (Strengthened to 3.5x to dominate local repulsion loops)
            dist_goal = math.sqrt((nr - gr)**2 + (nc - gc)**2)
            score = -3.5 * dist_goal
            
            # Repulsion from other active agents
            for j in range(self.num_agents):
                if j == agent_idx or not active_mask[j]:
                    continue
                
                jr, jc = agent_positions[j]
                dist_agent = math.sqrt((nr - jr)**2 + (nc - jc)**2)
                
                if dist_agent < 1.1:
                    # Immediate collision risk (strengthened from -15 to -40)
                    score -= 40.0
                elif dist_agent < 2.5:
                    # Stronger repulsion to prevent diagonal collisions (strengthened coefficient from 3.0 to 12.0)
                    score -= 12.0 / (dist_agent + 0.1)
                    
            scores.append(score)
            
        scores = np.array(scores)
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / np.sum(exp_s)
        return probs

    def get_joint_heuristic_policies(self, state):
        """
        Returns heuristic policy distributions for all agents as a numpy array of shape (M, 8).
        """
        probs_list = []
        for i in range(self.num_agents):
            probs_list.append(self.get_heuristic_policy(state, i))
        return np.stack(probs_list, axis=0)

if __name__ == "__main__":
    env = MultiAgentNavigationEnv(num_agents=2)
    s = env.generate_initial_state()
    print("Initial Agent Pos:", s[0])
    print("Initial Goal Pos:", s[1])
    
    # Step where agent 0 goes Down (1) and agent 1 goes Left (2)
    next_s, rewards, done, _ = env.step(s, (1, 2))
    print("Next Agent Pos:", next_s[0])
    print("Rewards:", rewards)
    print("Active Mask:", next_s[3])
