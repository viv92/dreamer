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


def eps_greedy_action(greedy_action, nA, eps):
    action = greedy_action
    r = np.random.uniform(0, 1)
    if eps > r:
        action = np.random.choice(nA)
    return action


### main

# hyperparams
lr = 1e-2
eps = .8
df = 0.99
num_samples = 10
max_episodes = 2000
render_final_episodes = True

# load environment
env = gym.make('CartPole-v1')
nS = 4
nA = env.action_space.n

# instantiate q net
policy = Pi_net(nS, nS, nA)

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
            # calculate state-wise returns for the trajectory
            state_wise_returns = []
            for t in range(len(trajectory)):
                state_wise_returns.append(trajectory_return)
                reward = trajectory[t][2]
                trajectory_return = (trajectory_return - reward) / df
            # standdardize state wise returns - reduces variance of the reinforce gradients (and removes dependency on scale and sign of rewards)
            state_wise_returns -= np.mean(state_wise_returns)
            state_wise_returns /= np.std(state_wise_returns)
            # formulate the trajectory loss
            trajectory_loss = 0
            for t in range(len(trajectory)):
                state, action, state_return = trajectory[t][0], trajectory[t][1], state_wise_returns[t]
                trajectory_loss += -torch.log(policy(state)[action]) * state_return
            loss += trajectory_loss
        loss = loss / m
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # flush for collecting trajectories from new policy
        trajectories = []


ep_returns_moving_mean = [0]
for ep_return in ep_return_list:
    n = len(ep_returns_moving_mean)
    prev_mean = ep_returns_moving_mean[n-1]
    new_mean = prev_mean + ((ep_return - prev_mean)/n)
    ep_returns_moving_mean.append(new_mean)


# plot results
plt.plot(ep_returns_moving_mean[1:], label='ep_return')
plt.legend()
plt.xlabel('episode')
plt.show()
