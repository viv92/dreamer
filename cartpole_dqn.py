import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

# policy as q function
class Qnet(nn.Module):
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
lr = 1e-3
eps = 1.
df = 0.99
train_period = 5
max_episodes = 10000

# load environment
env = gym.make('CartPole-v1')
nS = 4
nA = env.action_space.n

# instantiate q net
qnet = Qnet(nS, nS, nA)

# optimizer
optimizer = torch.optim.Adam(params=qnet.parameters(), lr=lr)

# start episodes
ep_loss_list = []
ep_return_list = []

for ep in range(max_episodes):

    # eps decay
    if ep % (max_episodes//10) == 0:
        if eps > 0:
            eps -= 0.1

    ep_loss = 0
    train_loss = 0
    ep_return = 0
    ep_steps = 0
    state = env.reset()
    done = False
    while not done:
        # env.render()
        action_probs = qnet(state).clone().detach().numpy()
        greedy_action = np.random.choice(nA, p=action_probs)
        action = eps_greedy_action(greedy_action, nA, eps)
        next_state, reward, done, _ = env.step(action)
        ep_return += (df ** ep_steps) * reward

        # td loss
        q_pred = qnet(state)[action]
        q_target = reward + df * torch.max(qnet(next_state))
        train_loss += (q_pred - q_target) ** 2

        if ep_steps % train_period == 0:
            # time to train
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss = 0

        # for next step
        state = next_state
        ep_steps += 1
        ep_loss += train_loss

    # store ep stats
    ep_return_list.append(ep_return)
    ep_loss_list.append(ep_loss.item())

    if ep % (max_episodes//10) == 0:
        print('ep:{} \t ep_return:{} \t ep_loss:{}'.format(ep, ep_return, ep_loss))



# plot results
fig, ax = plt.subplots(2)
ax[0].plot(ep_return_list, color='green', label='ep_return')
ax[1].plot(ep_loss_list, color='red', label='ep_loss')
# ax[1,0].set_ylim([0,1000])
plt.legend()
plt.xlabel('episode')
plt.show()
