import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import random

# policy function
class Pi_net(nn.Module):
    def __init__(self, s_dim, h_dim): # a_dim = num_actions
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, 1)
        self.fc2_sigma = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state.T))
        h2 = self.relu(self.fc1(h1))
        mu = self.fc2_mu(h2)
        sigma = self.relu(self.fc2_sigma(h2)) + 1e-8
        return mu, sigma


# def eps_greedy_action(greedy_action, nA, eps):
#     action = greedy_action
#     r = np.random.uniform(0, 1)
#     if eps > r:
#         action = np.random.choice(nA)
#     return action


### main

# hyperparams
lr = 1e-3
df = 0.99
num_samples = 10
max_episodes = 1000
render_final_episodes = True
random_seed = 42

# set random seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# load environment
env = gym.make('Pendulum-v1')
nS = 3

# instantiate q net
policy = Pi_net(nS, nS)

# optimizer
optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)

# start episodes
ep_loss_list = []
ep_return_list = []
trajectories = []

for ep in range(max_episodes):

    ep_return = 0
    ep_steps = 0
    ep_trace = []

    state = env.reset()
    done = False

    while not done:

        if render_final_episodes and ep + 10 > max_episodes:
            env.render()

        mu, sigma = policy(state)
        mu = mu.detach().numpy()
        sigma = sigma.detach().numpy()
        action = np.random.normal(mu, sigma)
        # action = eps_greedy_action(greedy_action, nA, eps)
        next_state, reward, done, _ = env.step(action)
        ep_return += (df ** ep_steps) * reward
        ep_trace.append([state, action, reward])

        # for next step
        state = next_state
        ep_steps += 1

    # store ep stats
    ep_return_list.append(ep_return)
    trajectories.append([ep_trace, ep_return])

    if ep % (max_episodes//10) == 0:
        print('ep:{} \t ep_return:{}'.format(ep, ep_return))

    if len(trajectories) % num_samples == 0:
        # time to train policy
        loss = 0
        for m in range(len(trajectories)):
            trajectory, trajectory_return = trajectories[m]
            trajectory_loss = 0
            for state, action, reward in trajectory:
                trajectory_loss += -Normal(loc=policy(state)[0], scale=policy(state)[1]).log_prob(torch.tensor(action)) * trajectory_return
                trajectory_return = (trajectory_return - reward) / df # update to return from next state to be considered in the trajectory
            loss += trajectory_loss
            # print('trajectory_loss:{} \t trajectory_return:{}'.format(trajectory_loss.item(), trajectory_return.item()))
        loss = loss / m
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # flush for collecting trajectories from new policy
        trajectories = []



# plot results
plt.plot(ep_return_list, label='ep_return')
plt.legend()
plt.xlabel('episode')
plt.show()
