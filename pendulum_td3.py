### program implementing td3 on inverted pendulum environment

## key features of this implementation:
# 1. Two critics and one actor
# 2. Critic updates based on the smaller Q target value
# 3. noise infusion in target action when estimating Q target value
# 4. delayed policy updates
# 5. policy update based on dpg on critic Q value

## todos / questions:
# 1.[resolved] policy update - the dpg should be taken on which of the two critics ? [fix: according to algo, critic_1]
# 2. add batchnorm layers in actor and critic networks


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


# td3
class TD3(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim, max_action, tau, df, buf_size, batch_size, lr_actor, lr_critic, action_noise_factor, noise_clip, policy_update_freq, device):
        super().__init__()
        self.actor = Actor(s_dim, a_dim, h_dim, max_action).to(device)
        self.critic_1 = Critic(s_dim, a_dim, h_dim).to(device)
        self.critic_2 = Critic(s_dim, a_dim, h_dim).to(device)
        self.target_actor = deepcopy(self.actor)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.replay_buffer = ReplayBuffer(buf_size, batch_size, s_dim, a_dim, device)
        self.optimizer_actor = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.optimizer_critc_1 = torch.optim.Adam(params=self.critic_1.parameters(), lr=lr_critic)
        self.optimizer_critc_2 = torch.optim.Adam(params=self.critic_2.parameters(), lr=lr_critic)
        self.df = df
        self.tau = tau
        self.max_action = max_action
        self.a_dim = a_dim
        self.action_noise_factor = action_noise_factor
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq
        self.device = device
        self.train_iters = 0

    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        action = action.data.cpu().numpy().flatten()
        noise = np.random.normal(loc = 0., scale = self.max_action * self.action_noise_factor, size = self.a_dim)
        action += noise
        action = action.clip(-self.max_action, self.max_action)
        return action

    def get_target_action(self, state):
        action = self.target_actor(state)
        noise = np.random.normal(loc = 0., scale = self.max_action * self.action_noise_factor, size = self.a_dim)
        noise = noise.clip(-self.noise_clip, self.noise_clip)
        action += torch.tensor(noise).to(self.device)
        action = action.clip(-self.max_action, self.max_action)
        return action

    def train(self):
        state, action, reward, next_state, not_done = self.replay_buffer.sample()

        with torch.no_grad():
            next_action = self.get_target_action(next_state)
            target_Q1 = reward + not_done * self.df * self.target_critic_1(next_state, next_action)
            target_Q2 = reward + not_done * self.df * self.target_critic_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        self.optimizer_critc_1.zero_grad()
        critic_1_loss.backward()
        self.optimizer_critc_1.step()

        critic_2_loss = F.mse_loss(current_Q2, target_Q)
        self.optimizer_critc_2.zero_grad()
        critic_2_loss.backward()
        self.optimizer_critc_2.step()

        if self.train_iters % self.policy_update_freq == 0:

            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.update_target_networks()

        self.train_iters += 1


    def update_target_networks(self):
        for target_param, current_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)

        for target_param, current_param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)

        for target_param, current_param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
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
    action_noise_factor = 0.2
    noise_clip = 0.5
    policy_update_freq = 2
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

    # init td3 agent
    agent = TD3(s_dim, a_dim, h_dim, max_action, tau, df, replay_buffer_size, batch_size, lr_actor, lr_critic, action_noise_factor, noise_clip, policy_update_freq, device)

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
                action = agent.get_action(state)

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
hyperparam_dict['algo'] = 'td3'
hyperparam_dict['lr-actor'] = str(lr_actor)
hyperparam_dict['lr-critic'] = str(lr_critic)
hyperparam_dict['df'] = str(df)
hyperparam_dict['max-ep'] = str(num_episodes)
hyperparam_dict['tau'] = str(tau)
hyperparam_dict['batch-size'] = str(batch_size)
hyperparam_dict['action_noise_factor'] = str(action_noise_factor)
hyperparam_dict['noise_clip'] = str(noise_clip)
hyperparam_dict['policy_update_freq'] = str(policy_update_freq)
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
