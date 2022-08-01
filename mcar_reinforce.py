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
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        h2 = self.relu(self.fc1(h1))
        h3 = self.relu(self.fc2(h2))
        out = self.softmax(self.fc3(h3))
        return out


def eps_greedy_action(greedy_action, nA, eps):
    action = greedy_action
    r = np.random.uniform(0, 1)
    if eps > r:
        action = np.random.choice(nA)
    return action


### main

# hyperparams
lr = 1e-3
eps = .9
df = .9
num_samples = 2
max_episodes = 1000
render_final_episodes = True

# load environment
env = gym.make('MountainCar-v0')
nS = 2
nA = env.action_space.n

# instantiate q net
policy = Pi_net(nS, nS*2, nA)

# optimizer
optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)

# start episodes
ep_loss_list = []
ep_return_list = []
trajectories = []

for ep in range(max_episodes):

    # eps decay
    if ep % (max_episodes//10) == 0:
        if eps > 0:
            eps -= 0.1

    ep_return = 0
    ep_steps = 0
    ep_trace = []

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
                trajectory_loss += -torch.log(policy(state)[action]) * trajectory_return
                trajectory_return = (trajectory_return - reward) / df # update to return from next state to be considered
            loss += trajectory_loss
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
