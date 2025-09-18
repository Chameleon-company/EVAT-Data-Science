import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class EVATGamificationEnv(gym.Env):
    def __init__(self, user_action_data):
        super(EVATGamificationEnv, self).__init__()
        self.user_action_data = user_action_data
        self.action_map = [
            "check_in",
            "report_fault",
            "validate_ai_prediction",
            "discover_new_station_in_black_spot",
            "use_route_planner",
            "ask_chatbot_question"
        ]

        # Gym spaces
        self.action_space = spaces.Discrete(len(self.action_map))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.action_map),), dtype=np.float32)

        # State tracking 
        self.points_balance = 0
        self.action_counts = {a: 0 for a in self.action_map}
        self.last_action = None
        self.steps = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.points_balance = 0
        self.action_counts = {a: 0 for a in self.action_map}
        self.last_action = None
        self.steps = 0
        self.max_steps = np.random.randint(30, 80)  # vary length
        obs = np.zeros(len(self.action_map), dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        action_type = self.action_map[action]
        self.action_counts[action_type] += 1

        # --- Reward shaping ---
        base_rewards = {
            "check_in": 10,
            "report_fault": 25,
            "validate_ai_prediction": 35,
            "discover_new_station_in_black_spot": 120,
            "use_route_planner": 15,
            "ask_chatbot_question": 10,
        }
        reward = base_rewards[action_type]

        

        # Diminishing returns (do same action too many times → lower reward)
        count = self.action_counts[action_type]
        reward *= 1.0 / (1 + 0.1 * (count - 1))  # e.g. 10, 9, 8.3, …

        # Exploration bonus (encourage trying new actions)
        if count == 1:
            reward += 5

        # Diversity bonus (don’t repeat same action back-to-back)
        if self.last_action is not None and self.last_action != action_type:
            reward += 2

        # --- Normalize / clip reward ---
        reward = reward / 100.0       # scale down
        reward = np.clip(reward, -1, 1)  # optional: keep within [-1, 1]

        # Update
        self.points_balance += reward
        self.last_action = action_type

        # Observation = normalized action counts
        obs = np.array(list(self.action_counts.values()), dtype=np.float32)
        obs = obs / (1 + obs.sum())

        terminated = self.steps >= self.max_steps
        truncated = False
        info = {"points_balance": self.points_balance, "last_action": action_type}

        return obs, reward, terminated, truncated, info
