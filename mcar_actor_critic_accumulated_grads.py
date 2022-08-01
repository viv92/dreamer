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
lr_policy = 1e-3
lr_vf = 1e-3
eps = 0.8
beta = 0
df = 0.99
update_steps = 5
max_episodes = 1200
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
policy = Pi_net(nS, nS, nA)
vf = V_net(nS, nS)

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
    update_steps_trace = []

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
        update_steps_trace.append([state, action, reward])

        # # train
        # if done:
        #     delta = reward + df * 0 - vf(state)
        # else:
        #     delta = reward + df * vf(next_state) - vf(state)
        # delta_scalar = delta.clone().detach()
        # p_loss = -torch.log(policy(state)[action]) * delta_scalar
        # # entropy = -torch.dot(policy(state), torch.log(policy(state)))
        # # p_loss -= beta * entropy
        # v_loss = delta**2
        #
        # # accumulate grads
        # v_loss.backward()
        # p_loss.backward()

        ## time to train
        if done or (ep_steps % update_steps == 0):
            # calculate losses - have to do O(n^2) iterations to prevent circular graph in autograd
            N = len(update_steps_trace)
            for j in range(N-1, -1, -1):

                # calculate discounted return
                discounted_return = 0
                if not done:
                    discounted_return = vf(next_state)
                for k in range(N-1, j, -1):
                    discounted_return = update_steps_trace[k][2] + df * discounted_return

                # calculate loss
                state, action, reward = update_steps_trace[j]
                delta = discounted_return - vf(state)
                delta_scalar = delta.clone().detach()
                p_loss = -torch.log(policy(state)[action]) * delta_scalar
                v_loss = delta**2
                # accumulate gradients
                v_loss.backward()
                p_loss.backward()
            # take the gradient step
            optimizer_vf.step()
            optimizer_policy.step()
            # for next update step
            optimizer_vf.zero_grad()
            optimizer_policy.zero_grad()

        # for next step
        state = next_state
        ep_steps += 1

    # store ep stats
    ep_return_list.append(ep_return)

    if ep % (max_episodes//10) == 0:
        print('ep:{} \t ep_return:{}'.format(ep, ep_return))


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
