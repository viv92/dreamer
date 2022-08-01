import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import random
import sys

# policy function
class Pi_net(nn.Module):
    def __init__(self, s_dim, h_dim, n_layers): # a_dim = num_actions
        super().__init__()
        self.n_layers = n_layers
        self.fc_in = nn.Linear(s_dim, h_dim)

        self.fc_n = nn.ModuleList()
        for  i in range(n_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.fc_out_mu = nn.Linear(h_dim, 1)
        self.fc_out_sigma = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state.T))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        mu = self.fc_out_mu(h)
        sigma = self.relu(self.fc_out_sigma(h)) + 1e-8
        return mu, sigma


# value function for baseline
class V_net(nn.Module):
    def __init__(self, s_dim, h_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.fc_in = nn.Linear(s_dim, h_dim)

        self.fc_n = nn.ModuleList()
        for i in range(n_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.fc_out = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        out = self.fc_out(h)
        return out



### main

# hyperparams
lr_policy = 1
lr_vf = 1
df = 0.99
num_samples = 2
max_episodes = 10000
render_final_episodes = False
random_seed = 42

# set random seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# load environment
env = gym.make('Pendulum-v1')
nS = 3
nH = 64
num_layers = 1

# instantiate function approximators
policy = Pi_net(nS, nH, num_layers)
vf = V_net(nS, nH, num_layers)

# optimizer
optimizer_policy = torch.optim.Adam(params=policy.parameters(), lr=lr_policy)
optimizer_vf = torch.optim.Adam(params=vf.parameters(), lr=lr_vf)

# start episodes
ep_loss_list = []
ep_return_list = []
trajectories = []

for ep in range(max_episodes):

    ep_return = 0
    ep_steps = 0
    ep_trace = []

    state = env.reset(seed=random_seed)
    done = False

    while not done:

        if render_final_episodes and ep + 5 > max_episodes:
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
        policy_loss = 0
        vf_loss = 0
        for m in range(len(trajectories)):
            trajectory, trajectory_return = trajectories[m]
            p_loss = 0
            v_loss = 0
            for state, action, reward in trajectory:
                delta = trajectory_return - vf(state)
                delta_scalar = delta.clone().detach()
                p_loss += -Normal(loc=policy(state)[0], scale=policy(state)[1]).log_prob(torch.tensor(action)) * delta_scalar
                v_loss += delta**2
                trajectory_return = (trajectory_return - reward) / df # update to return from next state to be considered
            policy_loss += p_loss
            vf_loss += v_loss

        policy_loss = policy_loss / m
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        vf_loss = vf_loss / m
        optimizer_vf.zero_grad()
        vf_loss.backward()
        optimizer_vf.step()

        # flush for collecting trajectories from new policy
        trajectories = []



ep_returns_moving_mean = [ep_return_list[0]]
n = 0
for ep_return in ep_return_list[1:]:
    n += 1
    n = n % (max_episodes//20)
    prev_mean = ep_returns_moving_mean[-1]
    new_mean = prev_mean + ((ep_return - prev_mean)/(n+1))
    ep_returns_moving_mean.append(new_mean)


# hyperparam dict
hyperparam_dict = {}
hyperparam_dict['env'] = 'Pendulum-v0'
hyperparam_dict['algo'] = 'reinforce-baseline'
hyperparam_dict['num-samples'] = str(num_samples)
hyperparam_dict['lr-policy'] = str(lr_policy)
hyperparam_dict['lr-vf'] = str(lr_vf)
hyperparam_dict['df'] = str(df)
hyperparam_dict['max-ep'] = str(max_episodes)
hyperparam_dict['nH'] = str(nH)
hyperparam_dict['num-layers'] = str(num_layers)
hyperparam_dict['random-seed'] = str(random_seed)

# hyperparam string
hyperstr = ""
for k,v in hyperparam_dict.items():
    hyperstr += k + ':' + v + "__"

# plot results
plt.plot(ep_returns_moving_mean, label='ep_return')
plt.legend()
plt.xlabel('episode')
plt.savefig('plots/' + hyperstr + '.png')
