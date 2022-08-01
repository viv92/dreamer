### program implementing SAC_v2 (Soft Actor Critic) on inverted pendulum environment

## key features of this implementation: same as SAC but with following changes:
# 1. no value function - use only Q functions - so two critic_Q nets and two target_critic_Q nets
# 2. learnable temperature parameter for trading off entropy maximization with return maximization - achieved by adding a temperature loss

## todos / questions:
# 1.

## important lessons:
# 1. when optimizing a distribution (e.g. the stochastic policy in this case), it is better to obtain the random variable sample as z ~ Normal(0, 1) followed by x = z * std + mean ; rather than directly sampling x ~ Normal(mean, std) -- because autograd is able to correctly backprop through the first case (correctly calculates gradients over learnable mean and std), but not through the second case (doesn't correctly calculates the gradients over learnable mean and std).
# 2. actions are clipped using tanh over the action sampled from gaussian policy pi. So actual policy pi' = tanh(pi). Thus when calculating log_prob(policy), use the change of variable formula: log_prob(pi') = log(pi) - log( det(jacobian(tanh)) - D * log(max_action)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import gym
from tqdm import tqdm


# actor network - parameterizing the stochastic poicy
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3_mean = nn.Linear(h_dim, a_dim)
        self.fc3_std = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()
        self.max_action = max_action
        self.tanh = nn.Tanh()
        self.a_dim = a_dim

        self.init_weights()

    def init_weights(self, init_w=3e-3):
        self.fc3_mean.weight.data.uniform_(-init_w, init_w)
        self.fc3_mean.bias.data.uniform_(-init_w, init_w)
        self.fc3_std.weight.data.uniform_(-init_w, init_w)
        self.fc3_std.bias.data.uniform_(-init_w, init_w)

    # note that this returns the mean and std of gaussian_policy
    # moreover, the actual policy used = tanh(gaussian_policy)
    def forward(self, state):
        h = self.relu(self.fc1(state))
        h = self.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        std = self.fc3_std(h)
        std = torch.clip(std, -20, 2)
        std = torch.exp(std)
        return mean, std

    # actual policy used pi' = tanh(pi) * max_action. Thus logprob(pi') = log(pi) - log( det(jacobian(tanh)) - D * log(max_action)
    def get_policy_logprob(self, state):
        mean, std = self.forward(state)
        z = tdist.Normal(0, 1).sample()
        gaussian_action = z * std + mean
        true_action = self.tanh(gaussian_action) * self.max_action

        gaussian_policy = tdist.Normal(mean, std)
        gaussian_policy_logprob = gaussian_policy.log_prob(gaussian_action)
        true_policy_logprob = gaussian_policy_logprob - torch.log( torch.abs(1 - torch.pow(self.tanh(gaussian_action), 2) + 1e-20) ) \
                              - self.a_dim * torch.log( torch.abs(torch.tensor(self.max_action)) )
        return true_policy_logprob, true_action


# critic network for parameterizing Q functions
class Critic_Q(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self, init_w=3e-3):
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        qval = self.fc3(h)
        return qval


# replay buffer
class ReplayBuffer:
    def __init__(self, buf_size, batch_size, s_dim, a_dim, device):
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.buf_state = np.zeros((buf_size, s_dim))
        self.buf_action = np.zeros((buf_size, a_dim))
        self.buf_next_state = np.zeros((buf_size, s_dim))
        self.buf_reward = np.zeros((buf_size, 1))
        self.buf_not_done = np.zeros((buf_size, 1))
        self.n_items = 0
        self.device = device

    def add(self, sars_tuple):
        state, action, reward, next_state, done = sars_tuple
        index = self.n_items % self.buf_size
        self.buf_state[index] = state
        self.buf_action[index] = action
        self.buf_reward[index] = reward
        self.buf_next_state[index] = next_state
        self.buf_not_done[index] = 1. - done
        self.n_items += 1

    def sample(self):
        limit = self.n_items
        if limit > self.buf_size:
            limit = self.buf_size
        idx = np.random.randint(0, limit, size=self.batch_size)
        state = torch.FloatTensor(self.buf_state[idx]).to(self.device)
        action = torch.FloatTensor(self.buf_action[idx]).to(self.device)
        reward = torch.FloatTensor(self.buf_reward[idx]).to(self.device)
        next_state = torch.FloatTensor(self.buf_next_state[idx]).to(self.device)
        not_done = torch.FloatTensor(self.buf_not_done[idx]).to(self.device)
        return (state, action, reward, next_state, not_done)


# SAC
class SAC(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim, max_action, tau, df, buf_size, batch_size, lr_actor, lr_critic, temperature_init, lr_temperature, device):
        super().__init__()
        self.actor = Actor(s_dim, a_dim, h_dim, max_action).to(device)
        self.critic_Q_1 = Critic_Q(s_dim, a_dim, h_dim).to(device)
        self.critic_Q_2 = Critic_Q(s_dim, a_dim, h_dim).to(device)
        self.target_critic_Q_1 = deepcopy(self.critic_Q_1)
        self.target_critic_Q_2 = deepcopy(self.critic_Q_2)
        self.temperature = torch.tensor(temperature_init).requires_grad_()
        self.replay_buffer = ReplayBuffer(buf_size, batch_size, s_dim, a_dim, device)
        self.optimizer_actor = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.optimizer_critc_Q_1 = torch.optim.Adam(params=self.critic_Q_1.parameters(), lr=lr_critic)
        self.optimizer_critc_Q_2 = torch.optim.Adam(params=self.critic_Q_2.parameters(), lr=lr_critic)
        self.optimizer_temperature = torch.optim.Adam(params=[self.temperature], lr=lr_temperature)
        self.df = df
        self.tau = tau
        self.max_action = max_action
        self.a_dim = a_dim
        self.device = device
        self.train_iters = 0
        self.tanh = nn.Tanh()
        self.target_entropy = -a_dim


    def get_action(self, state):
        mean, std = self.actor(state)
        z = tdist.Normal(0, 1).sample()
        gaussian_action = z * std + mean
        true_action = self.tanh(gaussian_action) * self.max_action
        return true_action


    def train(self):
        state, action, reward, next_state, not_done = self.replay_buffer.sample()

        # formulating Q value loss
        with torch.no_grad():
            policy_logprob_next_action, next_action = self.actor.get_policy_logprob(next_state)
            next_Q = torch.min(self.target_critic_Q_1(next_state, next_action), self.target_critic_Q_2(next_state, next_action))
            target_Q = reward + not_done * self.df * ( next_Q - self.temperature * policy_logprob_next_action )

        current_Q1 = self.critic_Q_1(state, action)
        current_Q2 = self.critic_Q_2(state, action)

        critic_Q1_loss = F.mse_loss(current_Q1, target_Q)
        self.optimizer_critc_Q_1.zero_grad()
        critic_Q1_loss.backward()
        self.optimizer_critc_Q_1.step()

        critic_Q2_loss = F.mse_loss(current_Q2, target_Q)
        self.optimizer_critc_Q_2.zero_grad()
        critic_Q2_loss.backward()
        self.optimizer_critc_Q_2.step()


        # formulating policy loss
        policy_logprob_current_action, current_action = self.actor.get_policy_logprob(state)

        target_Q1 = self.critic_Q_1(state, current_action)
        target_Q2 = self.critic_Q_2(state, current_action)
        target_Q = torch.min(target_Q1, target_Q2)

        policy_loss = (self.temperature.detach() * policy_logprob_current_action - target_Q).mean()
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()


        # formulating temperature loss
        with torch.no_grad():
            policy_logprob_current_action, current_action = self.actor.get_policy_logprob(state)

        temperature_loss = (-self.temperature * (policy_logprob_current_action + self.target_entropy)).mean()

        self.optimizer_temperature.zero_grad()
        temperature_loss.backward()
        self.optimizer_temperature.step()

        self.update_target_networks()



    def update_target_networks(self):
        for target_param, current_param in zip(self.target_critic_Q_1.parameters(), self.critic_Q_1.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)

        for target_param, current_param in zip(self.target_critic_Q_2.parameters(), self.critic_Q_2.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)



# main
if __name__ == '__main__':

    # hyperparams
    h_dim = 256
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_temperature = 3e-4
    tau = 0.005
    replay_buffer_size = 10**6
    df = 0.99
    batch_size = 256
    num_episodes = 250
    temperature_init = 1.
    random_seed = 0
    render_final_episodes = True
    init_random_episodes = 1
    num_train_calls = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('Pendulum-v1')
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    # init SAC agent
    agent = SAC(s_dim, a_dim, h_dim, max_action, tau, df, replay_buffer_size, batch_size, lr_actor, lr_critic, temperature_init, lr_temperature, device)

    # results and stats containers
    ep_return_list = []

    # start
    for ep in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        ep_return = 0
        ep_steps = 0

        while not done:
            if render_final_episodes and (ep > (num_episodes - 5)):
                env.render()

            if ep < init_random_episodes:
                # random action
                action = env.action_space.sample()
            else:
                action = agent.get_action(torch.tensor(state).to(device))
                action = action.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)

            # used to differentiate between goal_terminal_state and timeout_terminal_state
            done = float(done) if ep_steps < env._max_episode_steps else 0

            # add experience to replay buffer
            sars_tuple = [state, action, reward, next_state, done]
            agent.replay_buffer.add(sars_tuple)

            if ep >= init_random_episodes:
                # train agent
                for _ in range(num_train_calls):
                    agent.train()

            # for next step in episode
            state = next_state
            ep_return += (df ** ep_steps) * reward
            # ep_return += reward
            ep_steps += 1

        # store episode stats
        ep_return_list.append(ep_return)
        if ep % (num_episodes//10) == 0:
            print('ep:{} \t ep_return:{}'.format(ep, ep_return))


# hyperparam dict
hyperparam_dict = {}
hyperparam_dict['env'] = 'Pendulum-v1'
hyperparam_dict['algo'] = 'sac_v2'
hyperparam_dict['lr-actor'] = str(lr_actor)
hyperparam_dict['lr-critic'] = str(lr_critic)
hyperparam_dict['lr-temperature'] = str(lr_temperature)
hyperparam_dict['df'] = str(df)
hyperparam_dict['max-ep'] = str(num_episodes)
hyperparam_dict['tau'] = str(tau)
hyperparam_dict['batch-size'] = str(batch_size)
hyperparam_dict['temperature_init'] = str(temperature_init)
hyperparam_dict['num_train_calls'] = str(num_train_calls)
hyperparam_dict['Hdim'] = str(h_dim)
hyperparam_dict['random-seed'] = str(random_seed)
hyperparam_dict['init_random_episodes'] = str(init_random_episodes)

# hyperparam string
hyperstr = ""
for k,v in hyperparam_dict.items():
    hyperstr += k + ':' + v + "__"


# plot results
ep_returns_moving_mean = [ep_return_list[0]]
n = 0
for ep_return in ep_return_list[1:]:
    n += 1
    n = n % (num_episodes//20)
    prev_mean = ep_returns_moving_mean[-1]
    new_mean = prev_mean + ((ep_return - prev_mean)/(n+1))
    ep_returns_moving_mean.append(new_mean)

# plot results
plt.plot(ep_returns_moving_mean, label='ep_return')
plt.legend()
plt.xlabel('episode')
plt.savefig('plots/' + hyperstr + '.png')
