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
        self.fc1 = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        out = self.softmax(self.fc1(h1))
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
lr_policy = 1e-2
lr_vf = 1e-2
eps = 1.
df = 0.99
num_samples = 10
max_episodes = 2000
render_final_episodes = True
random_seed = 42

# load environment
env = gym.make('CartPole-v1')
nS = 4
nA = env.action_space.n

# set random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# instantiate function approximators
policy = Pi_net(nS, nS, nA)
vf = V_net(nS, nS)

# optimizer
optimizer_policy = torch.optim.Adam(params=policy.parameters(), lr=lr_policy)
optimizer_vf = torch.optim.Adam(params=vf.parameters(), lr=lr_vf)

# start episodes
ep_loss_list = []
ep_return_list = []
trajectories = []

for ep in range(max_episodes):

    # eps decay
    if ep % (max_episodes//10) == 0:
        eps -= (1/7)

    ep_return = 0
    ep_steps = 0
    ep_trace = []

    state = env.reset(seed=random_seed)
    done = False

    while not done:

        if render_final_episodes and ep + 10 > max_episodes:
            env.render()

        action_probs = policy(state).clone().detach().numpy()
        greedy_action = np.random.choice(nA, p=action_probs)
        action = eps_greedy_action(greedy_action, nA, eps)
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
                p_loss += -torch.log(policy(state)[action]) * delta_scalar
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

        # print('policy_loss:{} \t vf_loss:{}'.format(policy_loss.item(), vf_loss.item()))


ep_returns_100moving_mean = [ep_return_list[0]]
n = 0
for ep_return in ep_return_list[1:]:
    n += 1
    n = n % 100
    prev_mean = ep_returns_100moving_mean[-1]
    new_mean = prev_mean + ((ep_return - prev_mean)/(n+1))
    ep_returns_100moving_mean.append(new_mean)

# plot results
plt.plot(ep_returns_100moving_mean, label='ep_return')
plt.legend()
plt.xlabel('episode')
plt.show()
