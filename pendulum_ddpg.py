### program implementing ddpg on inverted pendulum environment

## key features of this implementation:
# 1. determinsitic policy - actor loss according to dpg algo
# 2. batchnorm layers to handle different input (state) scales
# 3. noise infusion in actions to allow exploration
# 4. soft update (moving mean) of target networks
# 5. clipping action range using tanh

## todos / questions:
# 1.[resolved] sampling from buffer - sample contagious blocks or random indices of given batch size? [fix - random indices]
# 2.[resolved] sampling from replay buffer - sample with or without replacement? [fix - with replacement]
# 3.[resolved] adding multivariate_normal noise in actions - setting covariance_matrix of MultivariateNormal [fix - use np.random.normal with size to draw independent samples equal to action_dim]
# 4. add batchnorm layers in actor and critic networks
# 5.[resolved] soft update of target params - correct way to update weights [fix - use param.data.copy_()]
# 6.[resolved] sars_tuple for replay_buffer - when to get next_action [fix - according to dpg algo - get it from target_actor when training]
# 7.[resolved] get state_dim and max_action from environment [fix - env.observation_space.shape[0] and env.action_space.high[0]]

## important learning: using `not_done` to prevent timeout_terminal_state from being regarded as the goal_terminal_state

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import gym
from tqdm import tqdm


# actor network
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()
        self.max_action = max_action
        self.tanh = nn.Tanh()

    def forward(self, state):
        h = self.relu(self.fc1(state))
        h = self.relu(self.fc2(h))
        action = self.tanh(self.fc3(h)) * self.max_action
        return action


# critic network
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

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


# ddpg
class DDPG(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim, max_action, tau, df, buf_size, batch_size, lr_actor, lr_critic, action_noise_factor, device):
        super().__init__()
        self.actor = Actor(s_dim, a_dim, h_dim, max_action).to(device)
        self.critic = Critic(s_dim, a_dim, h_dim).to(device)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.replay_buffer = ReplayBuffer(buf_size, batch_size, s_dim, a_dim, device)
        self.optimizer_actor = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.optimizer_critc = torch.optim.Adam(params=self.critic.parameters(), lr=lr_critic)
        self.df = df
        self.tau = tau
        self.max_action = max_action
        self.a_dim = a_dim
        self.device = device

    def get_action(self, state, action_noise_factor):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        action = action.data.cpu().numpy().flatten()
        noise = np.random.normal(loc = 0., scale = self.max_action * action_noise_factor, size = self.a_dim)
        action += noise
        action = action.clip(-self.max_action, self.max_action)
        return action

    def train(self):
        state, action, reward, next_state, not_done = self.replay_buffer.sample()

        next_action = self.target_actor(next_state)
        target_Q = reward + not_done * self.df * self.target_critic(next_state, next_action)
        target_Q = target_Q.detach()
        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target_Q)
        self.optimizer_critc.zero_grad()
        critic_loss.backward()
        self.optimizer_critc.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.update_target_networks()


    def update_target_networks(self):
        for target_param, current_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)

        for target_param, current_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)


# main
if __name__ == '__main__':

    # hyperparams
    h_dim = 256
    lr_actor = 3e-4
    lr_critic = 3e-4
    tau = 0.005
    replay_buffer_size = 10**6
    df = 0.99
    batch_size = 256
    num_episodes = 250
    # action_noise_init = 0.1
    # action_noise_factor = np.linspace(start=action_noise_init, stop=1e-8, num=num_episodes) # decayed
    action_noise_factor = 0.1
    random_seed = 0
    render_final_episodes = True
    init_random_episodes = 125

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

    # init ddpg agent
    agent = DDPG(s_dim, a_dim, h_dim, max_action, tau, df, replay_buffer_size, batch_size, lr_actor, lr_critic, action_noise_factor, device)

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
                action = agent.get_action(state, action_noise_factor)

            next_state, reward, done, _ = env.step(action)

            # used to differentiate between goal_terminal_state and timeout_terminal_state
            done = float(done) if ep_steps < env._max_episode_steps else 0

            # add experience to replay buffer
            sars_tuple = [state, action, reward, next_state, done]
            agent.replay_buffer.add(sars_tuple)

            if ep >= init_random_episodes:
                # train agent
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
hyperparam_dict['algo'] = 'ddpg'
hyperparam_dict['lr-actor'] = str(lr_actor)
hyperparam_dict['lr-critic'] = str(lr_critic)
hyperparam_dict['df'] = str(df)
hyperparam_dict['max-ep'] = str(num_episodes)
hyperparam_dict['tau'] = str(tau)
hyperparam_dict['batch-size'] = str(batch_size)
hyperparam_dict['action_noise_factor'] = str(action_noise_factor)
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
