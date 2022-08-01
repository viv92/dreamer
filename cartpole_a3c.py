import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


# local net - specific copy for each agent/worker
class LocalNet(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1_p = nn.Linear(h_dim, a_dim)
        self.fc1_v = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        out_p = self.softmax(self.fc1_p(h1))
        out_v = self.fc1_v(h1)
        return out_p, out_v


# global net - shared
class GlobalNet(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1_p = nn.Linear(h_dim, a_dim)
        self.fc1_v = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        out_p = self.softmax(self.fc1_p(h1))
        out_v = self.fc1_v(h1)
        return out_p, out_v


# shared adam optimizer
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# worker
class Worker(mp.Process):
    def __init__(self, env_name, nS, nA, nH, lr, eps, gnet, optimizer, ep_return_list, episode_counter, name):
        super().__init__()
        # local members
        self.env = gym.make(env_name)
        self.lnet = LocalNet(nS, nH, nA)
        self.l_optimizer = torch.optim.Adam(lr=lr, params=self.lnet.parameters())
        self.eps = eps
        self.name = str(name)

        # global members
        self.gnet = gnet
        self.g_optimizer = optimizer
        self.ep_return_list = ep_return_list
        self.episode_counter = episode_counter

    def run(self):
        while self.episode_counter.value < max_episodes:

            with self.episode_counter.get_lock():
                self.episode_counter.value += 1

            # eps decay
            if self.eps % (max_episodes//10) == 0:
                if self.eps > 0:
                    self.eps -= 0.1

            ep_return = 0
            ep_steps = 0

            state = self.env.reset()
            done = False

            # initialize dict to accumulate param gradients - {key=param_name: value=param.grad}
            # accumulated_gradients = {}
            # for param_name, param in self.lnet.named_parameters():
            #     accumulated_gradients[param_name] = None
            # self.l_optimizer.zero_grad()

            while not done:

                # if render_final_episodes and ep + 10 > max_episodes:
                #     env.render()

                action_probs = self.lnet(state)[0].clone().detach().numpy()
                greedy_action = np.random.choice(nA, p=action_probs)
                action = eps_greedy_action(greedy_action, nA, eps)
                next_state, reward, done, _ = self.env.step(action)
                ep_return += (df ** ep_steps) * reward

                # # train
                # delta = reward + df * self.lnet(next_state)[1] - self.lnet(state)[1]
                # delta_scalar = delta.clone().detach()
                # p_loss = -torch.log(self.lnet(state)[0][action]) * delta_scalar
                # v_loss = delta**2
                # p_entropy = torch.sum(self.lnet(state)[0] * torch.log(self.lnet(state)[0]))
                # total_loss = v_loss + p_loss + beta * p_entropy

                # self.l_optimizer.zero_grad()
                total_loss.backward()

                # accumulate gradients from local net
                # for param_name, param in self.lnet.named_parameters():
                #     if param.grad is not None:
                #         if accumulated_gradients[param_name] is None:
                #             accumulated_gradients[param_name] = param.grad.clone()
                #         else:
                #             accumulated_gradients[param_name] += param.grad.clone()

                if ep_steps % ep_update_steps == 0:
                    # time to apply the accumulated gradients to global net
                    # self.g_optimizer.zero_grad()

                    # for param_name, param in self.gnet.named_parameters():
                    #     param._grad = accumulated_gradients[param_name].clone()

                    # for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                    #     gp._grad = lp.grad

                    self.l_optimizer.step()
                    # self.g_optimizer.step()

                    # sync the local net params with the global net params
                    # self.lnet.load_state_dict(self.gnet.state_dict())

                    # clear accumulated gradient for next round
                    # for param_name, param in self.lnet.named_parameters():
                    #     accumulated_gradients[param_name] = None
                    self.l_optimizer.zero_grad()

                # for next step
                state = next_state
                ep_steps += 1

            # store ep stats
            self.ep_return_list.put(ep_return)
            if self.episode_counter.value % (max_episodes//10) == 0:
                print('ep:{} \t ep_return:{} \t worker:{}'.format(self.episode_counter.value, ep_return, self.name))




# function to get epsilon-greedy action
def eps_greedy_action(greedy_action, nA, eps):
    action = greedy_action
    r = np.random.uniform(0, 1)
    if eps > r:
        action = np.random.choice(nA)
    return action


## main

if __name__ == "__main__":
    # hyperparams
    lr = 1e-4
    eps = .8
    df = 0.99
    beta = 0 # entropy regularization
    ep_update_steps = 5
    max_episodes = 5000
    # render_final_episodes = True
    random_seed = 42

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    env_name = 'CartPole-v1'
    nS = 4
    nA = 2
    nH = 8

    gnet = GlobalNet(nS, nH, nA)
    gnet.share_memory()

    optimizer = SharedAdam(lr=lr, params=gnet.parameters())

    ep_return_list = mp.Queue()
    episode_counter = mp.Value('i', 0)

    # workers = [Worker(env_name, nS, nA, nH, lr, eps, gnet, optimizer, ep_return_list, episode_counter, i) for i in range(mp.cpu_count())]
    workers = [Worker(env_name, nS, nA, nH, lr, eps, gnet, optimizer, ep_return_list, episode_counter, i) for i in range(1)]
    [w.start() for w in workers]

    ep_returns_moving_mean = [0]
    while episode_counter.value < max_episodes:
        ep_return = ep_return_list.get()
        n = len(ep_returns_moving_mean)
        prev_mean = ep_returns_moving_mean[n-1]
        new_mean = prev_mean + ((ep_return - prev_mean)/n)
        ep_returns_moving_mean.append(new_mean)

    [w.join() for w in workers]


    # plot results
    plt.plot(ep_returns_moving_mean[1:], label='ep_return')
    plt.legend()
    plt.xlabel('episode')
    plt.show()
