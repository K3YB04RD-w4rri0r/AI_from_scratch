import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

def run(policy, num_episodes=50):
    lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        # print(obs.shape)
        t = 0
        while True:
            action = policy(obs)
            print(action)
            obs, r, terminated, truncated, _ = env.step(action)
            t += 1
            if terminated or truncated:
                break
        lengths.append(t)
    return np.mean(lengths) # length as a proxy ? 


if __name__ == "__main__":
    print(run(lambda o: env.action_space.sample()))  # random:        ~19 steps
    print(run(lambda o: int(o[2] > 0)))              # push toward the lean: ~39
    print(run(lambda o: int(o[2] + o[3] > 0)))       # angle + angular velocity: ~497