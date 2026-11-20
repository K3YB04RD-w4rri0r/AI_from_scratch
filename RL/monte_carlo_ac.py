import gymnasium as gym
import numpy as np
import torch
from RL.models.networks import policy_NET, value_NET
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

def calculate_returns(rewards, gamma=0.99):
    returns = torch.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def get_action(probs : torch.Tensor):
    probs = probs
    distribution = torch.distributions.Categorical(probs=probs)
    action = distribution.sample()
    return action

def run(num_episodes = 50, device="cpu", num_policy_updates = 5, lr = 0.01, num_value_updates = 1):
    all_rewards = []
    policy_model = policy_NET(device=device)
    value_model = value_NET(device = device)
    # optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr = lr)
    value_optimizer =  torch.optim.Adam(value_model.parameters(), lr = lr)


    trajectories =  []

    for ep in range(1,num_episodes + 1):
        print(ep)
        if ep % (num_episodes//num_policy_updates)  == 0: # we update the trajectory
            print("UPDATING POLICY", ep)
            # need to calculate the total loss over trajectories
            # then backwards
            # we need to reset the tajectories since the policy updates

            policy_optimizer.zero_grad()


            # loss calculation
            logprobs = []
            s, values, rewards = [], [], []
            for trajectory_index in range(len(trajectories)): # (length_trajectory, 3)
                trajectory = trajectories[trajectory_index] # list of tuples

                log_probs = torch.stack([l for s,r,l in trajectory]).to(device=device)
                
                traj_rewards = torch.stack([r for s,r,l in trajectory]).to(device=device)
                discounted_returns = calculate_returns(traj_rewards)
                
                states = torch.stack([s for s,r,l in trajectory]).to(device=device)
                
                state_values = value_model(states).squeeze(-1)

                s.append(states)
                values.append(state_values)
                rewards.append(discounted_returns)
                logprobs.append(log_probs)

                

            s, values, rewards = torch.cat(s), torch.cat(values), torch.cat(rewards)
            logprobs, phi = torch.cat(logprobs), (rewards.detach() - values.detach())
            assert values.shape == rewards.shape == logprobs.shape, (values.shape, rewards.shape, logprobs.shape)



            policy_loss = - torch.sum(logprobs * phi)/len(trajectories)
            policy_loss.backward()
            policy_optimizer.step()


            for _ in range(num_value_updates):

                value_optimizer.zero_grad()
                preds = value_model(s).squeeze(-1)
                value_loss = torch.mean((preds-rewards) ** 2 )
                value_loss.backward()
                value_optimizer.step()

            trajectories = []

        # we start collecting again
        obs, _ = env.reset(seed = ep)
        obs = torch.tensor(obs, device=device).unsqueeze(0)
        timestep = 0
        episode_reward = 0
        current_trajetory = []

        while True: # samples one trajetory
            probs = policy_model(obs)
            action = get_action(probs)
            # print(probs, probs.shape, action.item())
            action_prob = (probs.squeeze())[action.item()]
            # print(action_prob, action.type())
            # print(int(action.squeeze().cpu().numpy()))
            st = obs
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            obs = torch.tensor(obs, device=device).unsqueeze(0)
            current_trajetory.append((st ,torch.tensor(reward, device=device), torch.log(action_prob)))
            timestep += 1

            episode_reward += float(reward)
            
            if terminated or truncated:
                all_rewards.append(episode_reward)
                trajectories.append(current_trajetory) # list(2 tensor elsement lists)
                break

    print("DONE")
    return all_rewards


if __name__=="__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # rewards = run(num_episodes=700, device=device, num_policy_updates=50, lr = 0.001) # noisy
    rewards = run(num_episodes=600, device=device, num_policy_updates=100, lr = 0.01) # collapses on sgd, noisy with adam but learned
    w = 50
    plt.plot(np.convolve(rewards, np.ones(w) / w, mode="valid"))
    plt.xlabel("episode"); plt.ylabel(f"return ({w}-episode average)")
    plt.show()