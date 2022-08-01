import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

# policy function
class Pi_net(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim): # a_dim = num_actions
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        h2 = self.relu(self.fc1(h1))
        out = self.softmax(self.fc2(h2))
        return out


# value function for baseline
class V_net(nn.Module):
    def __init__(self, s_dim, h_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        out = self.fc1(h1)
        return out



def eps_greedy_action(greedy_action, nA, eps):
    action = greedy_action
    r = np.random.uniform(0, 1)
    if eps > r:
        action = np.random.choice(nA)
    return action


### main

# hyperparams
lr_policy = 1e-5
lr_vf = 1e-4
eps = .8
df = 0.99
max_episodes = 2000
render_final_episodes = True
random_seed = 42

# set random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# load environment
env = gym.make('MountainCar-v0')
nS = 2
nA = env.action_space.n

# instantiate function approximators
policy = Pi_net(nS, nS*2, nA)
vf = V_net(nS, nS*2)

# optimizer
optimizer_policy = torch.optim.Adam(params=policy.parameters(), lr=lr_policy)
optimizer_vf = torch.optim.Adam(params=vf.parameters(), lr=lr_vf)

# start episodes
ep_return_list = []

for ep in range(max_episodes):

    # eps decay
    if ep % (max_episodes//10) == 0:
        if eps > 0:
            eps -= 0.1

    ep_return = 0
    ep_steps = 0

    state = env.reset()
    done = False

    while not done:

        if render_final_episodes and ep + 10 > max_episodes:
            env.render()

        action_probs = policy(state).clone().detach().numpy()
        greedy_action = np.random.choice(nA, p=action_probs)
        action = eps_greedy_action(greedy_action, nA, eps)
        next_state, reward, done, _ = env.step(action)
        ep_return += (df ** ep_steps) * reward

        # train
        delta = reward + df * vf(next_state) - vf(state)
        delta_scalar = delta.clone().detach()
        p_loss = -torch.log(policy(state)[action]) * delta_scalar
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
