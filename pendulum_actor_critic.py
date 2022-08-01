import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

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


class Pi_net_small(nn.Module):
    def __init__(self, s_dim, h_dim): # a_dim = num_actions
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1_mu = nn.Linear(h_dim, 1)
        self.fc1_sigma = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state.T))
        mu = self.fc1_mu(h1)
        sigma = self.relu(self.fc1_sigma(h1)) + 1e-8
        return mu, sigma


# value function for baseline
class V_net(nn.Module):
    def __init__(self, s_dim, h_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        h2 = self.relu(self.fc1(h1))
        out = self.fc2(h2)
        return out



### main

# hyperparams
lr_policy = 1e-2
lr_vf = 1e-3
df = 0.99
max_episodes = 1000
render_final_episodes = True
random_seed = 42

# set random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# load environment
env = gym.make('Pendulum-v1')
nS = 3

# instantiate function approximators
policy = Pi_net(nS, nS*2)
vf = V_net(nS, nS*2)

# optimizer
optimizer_policy = torch.optim.Adam(params=policy.parameters(), lr=lr_policy)
optimizer_vf = torch.optim.Adam(params=vf.parameters(), lr=lr_vf)

# start episodes
ep_return_list = []

for ep in range(max_episodes):

    ep_return = 0
    ep_steps = 0

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

        # train
        delta = reward + df * vf(next_state) - vf(state)
        delta_scalar = delta.clone().detach()
        p_loss = -Normal(loc=policy(state)[0], scale=policy(state)[1]).log_prob(torch.tensor(action)) * delta_scalar
        v_loss = delta**2

        optimizer_policy.zero_grad()
        p_loss.backward()
        optimizer_policy.step()

        optimizer_vf.zero_grad()
        v_loss.backward()
        optimizer_vf.step()

        # for next step
        state = next_state
        ep_steps += 1

    # store ep stats
    ep_return_list.append(ep_return)

    if ep % (max_episodes//10) == 0:
        print('ep:{} \t ep_return:{}'.format(ep, ep_return))



# plot results
plt.plot(ep_return_list, label='ep_return')
plt.legend()
plt.xlabel('episode')
plt.show()
