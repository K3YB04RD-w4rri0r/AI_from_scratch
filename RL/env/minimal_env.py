import gymnasium as gym
from gymnasium import spaces

class Corridor(gym.Env):
    """Start at cell 0, goal at cell n-1. Actions: 0 = left, 1 = right."""

    def __init__(self, n=5):
        self.n = n
        self.observation_space = spaces.Discrete(n)   # what obs look like
        self.action_space = spaces.Discrete(2)        # what actions are legal

    def reset(self, *, seed=None, options = None):
        super().reset(seed=seed)    # seeds self.np_random
        self.pos = 0
        return self.pos, {}         # (obs, info)

    def step(self, action):
        self.pos = min(max(self.pos + (1 if action == 1 else -1), 0), self.n - 1)
        terminated = self.pos == self.n - 1
        reward = 1.0 if terminated else 0.0
        return self.pos, reward, terminated, False, {}   # obs, r, terminated, truncated, info


if __name__=="__main__":
    env = gym.make("cartpole-v1")
    def run():
        for i in range(50):
            obs, _ = env.reset()
            while True:
                obs, r, terminated, truncated, _ = env.step(action=0)
                if terminated or truncated:
                    break
                    # log reward
        return None

    run()
