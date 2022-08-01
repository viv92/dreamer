import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

# policy function
class Pi_net(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim, num_layers):
        super().__init__()
        self.fc_in = nn.Linear(s_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, a_dim)

        self.n_layers = num_layers
        self.fc_n = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        out = self.softmax(self.fc_out(h))
        return out


# value function for baseline
class V_net(nn.Module):
    def __init__(self, s_dim, h_dim, num_layers):
        super().__init__()
        self.fc_in = nn.Linear(s_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, 1)

        self.n_layers = num_layers
        self.fc_n = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        out = self.fc_out(h)
        return out



def eps_greedy_action(greedy_action, nA, eps):
    action = greedy_action
    r = np.random.uniform(0, 1)
    if eps > r:
        action = np.random.choice(nA)
    return action


### main

# hyperparams
lr_policy = 1e-4
lr_vf = 1e-4
eps = 0.8
beta = 0
df = 0.99
update_steps = 2
max_episodes = 2000
render_final_episodes = True
random_seed = 42

# set random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# load environment
env = gym.make('CartPole-v1')
nS = 4
nA = env.action_space.n
nH = 24
num_layers = 2

# instantiate function approximators
policy = Pi_net(nS, nH, nA, num_layers)
vf = V_net(nS, nH, num_layers)

# optimizer
optimizer_policy = torch.optim.Adam(params=policy.parameters(), lr=lr_policy)
optimizer_vf = torch.optim.Adam(params=vf.parameters(), lr=lr_vf)

# start episodes
ep_return_list = []

for ep in range(max_episodes):

    # eps decay
    if ep % (max_episodes//10) == 0:
        eps -= (1/7)

    ep_return = 0
    ep_steps = 0
    update_steps_trace = []

    state = env.reset(seed=random_seed)
    done = False

    # initialize dict to accumulate param gradients - {key=param_name: value=param.grad}
    accumulated_gradients_policy = {}
    for param_name, param in policy.named_parameters():
        accumulated_gradients_policy[param_name] = None
    accumulated_gradients_vf = {}
    for param_name, param in vf.named_parameters():
        accumulated_gradients_vf[param_name] = None

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

                # accumulate gradients
                for param_name, param in policy.named_parameters():
                    if param.grad is not None:
                        if accumulated_gradients_policy[param_name] is None:
                            accumulated_gradients_policy[param_name] = param.grad.clone()
                        else:
                            accumulated_gradients_policy[param_name] += param.grad.clone()
                for param_name, param in vf.named_parameters():
                    if param.grad is not None:
                        if accumulated_gradients_vf[param_name] is None:
                            accumulated_gradients_vf[param_name] = param.grad.clone()
                        else:
                            accumulated_gradients_vf[param_name] += param.grad.clone()

                # clear gradients for next step evaluation
                optimizer_vf.zero_grad()
                optimizer_policy.zero_grad()

            # apply the accumulated gradients
            for param_name, param in policy.named_parameters():
                param._grad = accumulated_gradients_policy[param_name].clone()
            for param_name, param in vf.named_parameters():
                param._grad = accumulated_gradients_vf[param_name].clone()
            optimizer_vf.step()
            optimizer_policy.step()

            # clear gradients for next round of update step
            update_steps_trace = []
            optimizer_vf.zero_grad()
            optimizer_policy.zero_grad()
            for param_name, param in policy.named_parameters():
                accumulated_gradients_policy[param_name] = None
            for param_name, param in vf.named_parameters():
                accumulated_gradients_vf[param_name] = None


        # for next step
        state = next_state
        ep_steps += 1

    # store ep stats
    ep_return_list.append(ep_return)

    if ep % (max_episodes//10) == 0:
        print('ep:{} \t ep_return:{}'.format(ep, ep_return))


# resuts to plot
ep_returns_100moving_mean = [ep_return_list[0]]
n = 0
for ep_return in ep_return_list[1:]:
    n += 1
    n = n % 100
    prev_mean = ep_returns_100moving_mean[-1]
    new_mean = prev_mean + ((ep_return - prev_mean)/(n+1))
    ep_returns_100moving_mean.append(new_mean)

# hyperparam dict
hyperparam_dict = {}
hyperparam_dict['env'] = 'CartPole-v1'
hyperparam_dict['algo'] = 'actor-critic-accumulated-gradients-explicit-buffer'
hyperparam_dict['update-steps'] = str(update_steps)
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
plt.plot(ep_returns_100moving_mean, label='ep_return')
plt.legend()
plt.xlabel('episode')
plt.savefig('plots/' + hyperstr + '.png')
