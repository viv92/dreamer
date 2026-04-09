### Program implementing DreamerV2 on LunarLander-v2  environment
## Series of dreamerv2_5_1_12_x_cu.py programs involve fixes applied to cartpole_dreamerv2_5_1_12_cu after changelog analysis
# This implementation adds further fixes over dreamerv2_5_1_12_3_cu.py by removing skip update for reward and df loss (we don't need to skip loss update for reward and df loss at the first step of an episode since the explicitly added terminal state in replay buffer takes care of that) + learnable sigma for gaussian models (reward and observation model)


## Key features of implementation / algorithm (differences over dreamer)
# 1. Latent distribution is one-hot categorical instead of gaussian - for transition model and actor (policy)
# 2. Straight through gradient for backpropogating through samples from the categorical distribution
# 3. KL balancing instead of free nats
# 4. Actor loss - in addition to negative lambda return, use reinforce loss and entropy of policy
# 5. Target network for critic
# 6. Model for learning discount factor - bernouli distribution
# 7. Actor and Critic losses are weighted by discount factor

## todos / questions
# 1. use mean reward instead of sample from reward model? Same for observation and df models.
# 2. sampling from replay buffer - adjusting the sampling window to encounter more terminal states
# 3. weighing actor and critic loss by learnt discount factor (cummulative product) - but the learnt df is state dependent
# 4. handling terminal state with df learning
# 5. correct indexing of values when dynamics learning and behaviour learning
# 6. tdist.Bernoulli vs tdist.bernoulli.Bernoulli (same problme for OneHotCategorical)
# 7. clarify backpropagation path for actor loss - through lambda_return and reinforce objective
# 8. losses - sum axis and mean axis
# 9. [resolved] scheduled eps-greedy for exploration during interaction
# 10. OneHotCategorical sometimes yieds non-binary value (e.g. 0.9994)

## important lessons / takeaways
# 1. dis.probs.clone.detach() or dis.clone().detach().probs



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.bernoulli import Bernoulli
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import gymnasium as gym
from tqdm import tqdm
from torchviz import make_dot
from copy import deepcopy
from graphviz import Source
import imageio

# torch.autograd.set_detect_anomaly(True)


# RSSM used for both Representation model p(s_t | s_t-1, a_t-1, o_t) and Transition model q(s_t | s_t-1, a_t-1)
class RSSM(nn.Module):
    def __init__(self, in_dim, o_dim, belief_dim, h_dim, out_dim, batch_size, device):
        super().__init__()
        self.fc_embed = nn.Linear(in_dim, belief_dim) # layer to embed input (s_t, a_t)

        # init deterministic recurrent net (shared between p and q models)
        self.gru_cell = nn.GRUCell(belief_dim, belief_dim) # belief h_t = f(h_t-1, s_t-1, a_t-1)

        # init stochastic net (separate layers for p and q models)
        self.obs_encoder_fc1 = nn.Linear(o_dim, h_dim)
        self.obs_encoder_fc2 = nn.Linear(h_dim, h_dim)
        self.p_fc1 = nn.Linear(belief_dim + h_dim, h_dim)
        self.p_fc2_logits = nn.Linear(h_dim, out_dim)
        self.q_fc1 = nn.Linear(belief_dim, h_dim)
        self.q_fc2_logits = nn.Linear(h_dim, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.device = device


    # forward pass through RSSM
    def forward(self, prev_belief, x, o=None):
        x = self.elu(self.fc_embed(x))
        belief = self.gru_cell(x, prev_belief)
        if o is None:
            h = self.elu(self.q_fc1(belief))
            logits = self.q_fc2_logits(h)
        else:
            o = self.elu(self.obs_encoder_fc1(o))
            o = self.obs_encoder_fc2(o)
            h_o = torch.cat((belief, o), dim=1)
            h = self.elu(self.p_fc1(h_o))
            logits = self.p_fc2_logits(h)
        return logits, belief

    # to draw sample from the learnt probabilistic model
    def sample(self, prev_belief, x, o=None):
        logits, belief = self.forward(prev_belief, x, o)
        dis = OneHotCategorical(logits=logits)
        out = dis.sample()
        # for straight through gradient
        out = out + dis.probs - dis.probs.clone().detach()
        return out, belief

    # formulates the one_hot_categorical distribution from logits
    def get_dist(self, prev_belief, x, o=None):
        logits, belief = self.forward(prev_belief, x, o)
        dis = OneHotCategorical(logits=logits)
        return dis

    # detached version of get_dist - used for kl balancing
    def get_dist_detached(self, prev_belief, x, o=None):
        logits, belief = self.forward(prev_belief, x, o)
        logits = logits.detach()
        dis = OneHotCategorical(logits=logits)
        return dis

    # calculates log p(y|x)
    def log_prob(self, prev_belief, x, y, o=None):
        dis = self.get_dist(prev_belief, x, o)
        lp = dis.log_prob(y)
        return lp



# Stochastic net representing parameterized gaussian distribution - used for reward model and observation model
class StochasticNet_Gaussian(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3_mean = nn.Linear(h_dim, out_dim)
        self.fc3_std = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        logstd = self.fc3_std(h).clip(-4, 2)
        std = torch.exp(logstd)
        # std = .25
        return mean, std

    # to draw sample from the learnt probabilistic model
    def sample(self, state, belief):
        mean, std = self.forward(state, belief)
        eps = torch.randn_like(std)
        out = mean + eps * std
        return out

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        mean, std = self.forward(state, belief)
        lp = tdist.Normal(mean, std).log_prob(y)
        return lp


# Stochastic net representing parameterized bernoulli distribution - used for discount factor model
class StochasticNet_Bernoulli(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        # self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=1)
        h = self.relu(self.fc1(x))
        # h = self.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits

    # parameterized bernoulli distribution
    def get_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = Bernoulli(logits=logits)
        return dis

    # to draw sample from the learnt probabilistic model
    def sample(self, state, belief):
        dis = self.get_dist(state, belief)
        out = dis.sample()
        # for straight through gradient
        out = out + dis.probs - dis.probs.clone().detach()
        return out

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        dis = self.get_dist(state, belief)
        lp = dis.log_prob(y)
        return lp



# actor network - parameterizing the stochastic poicy
class Actor(nn.Module):
    def __init__(self, s_dim, belief_dim, a_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + belief_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, h_dim)
        self.fc5_logits = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()

    # returns the logits of one_hot_categorical distribution representing the policy
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        logits = self.fc5_logits(h)
        return logits

    # returns the policy as one_hot_categorical distribution
    def policy_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = OneHotCategorical(logits=logits)
        return dis

    # returns policy log_prob
    def policy_logprob(self, state, belief, y):
        policy_dis = self.policy_dist(state, belief)
        lp = policy_dis.log_prob(y)
        return lp



# critic network for parameterizing Value function
class Critic_V(nn.Module):
    def __init__(self, s_dim, belief_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + belief_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        val = self.fc4(h)
        return val



# replay buffer
class ReplayBuffer:
    def __init__(self, buf_size, sample_seq_len, ep_seq_len, batch_size, o_dim, a_dim, device):
        self.buf_size = buf_size
        self.sample_seq_len = sample_seq_len
        self.ep_seq_len = ep_seq_len 
        self.batch_size = batch_size
        self.buf_observation = np.zeros((buf_size, ep_seq_len, o_dim))
        self.buf_action = np.zeros((buf_size, ep_seq_len, a_dim))
        self.buf_reward = np.zeros((buf_size, ep_seq_len))
        self.buf_done = np.zeros((buf_size, ep_seq_len))
        self.n_items = 0
        self.device = device

    def add(self, ep_trace):
        observations, actions, rewards, dones = zip(*ep_trace)
        index = self.n_items % self.buf_size
        self.buf_observation[index] = observations
        self.buf_action[index] = actions
        self.buf_reward[index] = rewards

        # right shift dones 
        dones = list(dones)
        dones = [dones[0]] + dones[:-1]
        self.buf_done[index] = dones

        self.n_items += 1

    def sample(self):
        row_limit = self.n_items
        if row_limit > self.buf_size:
            row_limit = self.buf_size

        row_idx = np.random.randint(0, row_limit, self.batch_size)

        observation = torch.FloatTensor(self.buf_observation[row_idx])
        action = torch.FloatTensor(self.buf_action[row_idx])
        reward = torch.FloatTensor(self.buf_reward[row_idx]).unsqueeze(-1)
        done = torch.FloatTensor(self.buf_done[row_idx]).unsqueeze(-1)

        # permute time index to front
        observation = observation.permute(1,0,2).to(device)
        action = action.permute(1,0,2).to(device)
        reward = reward.permute(1,0,2).to(device)
        done = done.permute(1,0,2).to(device)
        return observation, action, reward, done



# DreamerV2
class DreamerV2(nn.Module):
    def __init__(self, o_dim, s_dim, a_dim, belief_dim, h_dim, sample_seq_len, ep_seq_len, imagination_horizon, df, buf_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, tau, rho, eta, device):
        super().__init__()
        self.actor = Actor(s_dim, belief_dim, a_dim, h_dim).to(device)
        self.critic_V = Critic_V(s_dim, belief_dim, h_dim).to(device)
        self.target_critic_V = deepcopy(self.critic_V)
        self.replay_buffer = ReplayBuffer(buf_size, sample_seq_len, ep_seq_len, batch_size, o_dim, a_dim, device)
        self.rssm = RSSM(s_dim + a_dim, o_dim, belief_dim, h_dim, s_dim, batch_size, device).to(device)
        self.df_model = StochasticNet_Bernoulli(s_dim + belief_dim, h_dim, 1).to(device)
        self.reward_model = StochasticNet_Gaussian(s_dim + belief_dim, h_dim, 1).to(device)
        self.observation_model = StochasticNet_Gaussian(s_dim + belief_dim, h_dim, o_dim).to(device)
        self.state_model = StochasticNet_Gaussian(o_dim, h_dim, s_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic_V = torch.optim.Adam(params=self.critic_V.parameters(), lr=lr_critic)
        self.optimizer_model = torch.optim.Adam(params=list(self.rssm.parameters()) + list(self.df_model.parameters()) + \
        list(self.reward_model.parameters()) + list(self.observation_model.parameters()) + list(self.state_model.parameters()), lr=lr_model)
        self.df = df
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.belief_dim = belief_dim
        self.o_dim = o_dim
        self.device = device
        self.train_iters = 0
        self.sample_seq_len = sample_seq_len
        self.ep_seq_len = ep_seq_len
        self.imagination_horizon = imagination_horizon
        self.vib_beta = vib_beta
        self._lambda = _lambda
        self.alpha = alpha # used for kl balancing
        self.tau = tau # used when updating target_critic_V
        self.rho = rho # used for weighing actor dynamics loss and actor reinforce loss
        self.eta = eta # used for weighing entropy regulaization in actor loss
        self.batch_size = batch_size

    def get_action(self, state, belief):
        policy = self.actor.policy_dist(state, belief)
        action = policy.sample()
        # for straight through gradient
        action = action + policy.probs - policy.probs.clone().detach()
        return action

    # epsilon greedy exploration
    def action_exploration(self, action, epsilon):
        if torch.rand(1) < epsilon:
            idx = torch.randint(low=0, high=self.a_dim, size=(1,))
            action = torch.zeros(1, self.a_dim)
            action[0][idx] = 1
        return action


    def freeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(False)

    def unfreeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(True)


    def calculate_lambda_return(self, rewards, state_values_target, discounts):
        """
        Input:
        # rewards obtained from reward model - r[0] : r[H-1]
        # state values obtained from target_critic model - v_t[0] : v_t[H-1]
        # df values obtained from df model - df[0] : df[H-1]

        Output:
        # V_lambda[0] : V_lambda[H-2]
        """
        lambda_returns = []
        accumulator = state_values_target[-1] # V_t[H-1]
        for t in range(len(rewards)-1, 0, -1): # t is just going from last element to first element, since all arrays are of length H-1
            accumulator = rewards[t] + discounts[t] * ( (1 - self._lambda)*state_values_target[t] + self._lambda*accumulator )
            # V_lambda[H-2] = reward[H-1] + df[H-1] * ( (1-lambda) * V_t[H-1] + lambda * V_lambda[H-1] )
            lambda_returns = [accumulator] + lambda_returns
        lambda_returns = torch.stack(lambda_returns, dim=0)
        return lambda_returns


    def train(self):

        # sample from replay buffer
        observation, action, reward, done = self.replay_buffer.sample() 

        # prepare done targets (clip to {0,1})
        done = done.float().clip(0, 1)

        #########################
        ## dynamics learning (using experience sampled from replay buffer)
        #########################

        # unfreeze dynamics model params
        self.unfreeze_model_params(self.rssm)
        self.unfreeze_model_params(self.df_model)
        self.unfreeze_model_params(self.reward_model)
        self.unfreeze_model_params(self.observation_model)
        self.unfreeze_model_params(self.state_model)

        # using reconstruction loss for now
        # todo - try NCE loss

        loss_dynamics = 0
        loss_reward = 0
        loss_obs = 0
        loss_df = 0
        loss_kl = 0

        # list tensors to store states and beliefs obtained from representation model - used later for imagination rollout (during behaviour learning)
        state_list = []
        belief_list = []

        ## prepare starting states

        prev_state = torch.zeros(self.batch_size, self.s_dim).to(self.device)
        prev_belief = torch.zeros(self.batch_size, self.belief_dim).to(self.device)
        prev_action = torch.zeros(self.batch_size, self.a_dim).to(self.device)

        start_time_idx = np.random.randint(0, self.ep_seq_len - self.sample_seq_len)
        
        for t in range(0, start_time_idx):
            curr_obs = observation[t]
            curr_action = action[t]

            rssm_input = torch.cat((prev_state, prev_action), dim=1)
            curr_state, curr_belief = self.rssm.sample(prev_belief, rssm_input, curr_obs)

            prev_state = curr_state.detach() 
            prev_belief = curr_belief.detach()
            prev_action = curr_action


        # start training steps
        for t in range(start_time_idx, start_time_idx + self.sample_seq_len):
            curr_obs = observation[t]
            curr_action = action[t]
            curr_reward = reward[t]
            curr_done = done[t] 

            rssm_input = torch.cat((prev_state, prev_action), dim=1)
            curr_state, curr_belief = self.rssm.sample(prev_belief, rssm_input, curr_obs)

            state_list.append(curr_state.clone().detach()) # s_t - store state for imagination rollout
            belief_list.append(curr_belief.clone().detach()) # h_t - store belief for imagination rollout

            # log prob q(r_t | s_t)
            if t > start_time_idx:
                lp_reward = self.reward_model.log_prob(curr_state, curr_belief, prev_reward)
                lp_reward = lp_reward.sum(dim=1).mean()
            else:
                lp_reward = 0

            # log prob q(o_t | s_t)
            lp_obs = self.observation_model.log_prob(curr_state, curr_belief, curr_obs)
            lp_obs = lp_obs.sum(dim=1).mean()

            # log prob q(df_t | s_t)
            df_target = 1. - curr_done
            lp_df = self.df_model.log_prob(curr_state, curr_belief, df_target)
            lp_df = lp_df.sum(dim=1).mean()

            # KL divergence - using KL balancing
            dist_p = self.rssm.get_dist(prev_belief, rssm_input, curr_obs)
            dist_q = self.rssm.get_dist(prev_belief, rssm_input)
            dist_p_detached = self.rssm.get_dist_detached(prev_belief, rssm_input, curr_obs)
            dist_q_detached = self.rssm.get_dist_detached(prev_belief, rssm_input)
            kl_pq = self.alpha * tdist.kl.kl_divergence(dist_p_detached, dist_q) + \
                    (1 - self.alpha) * tdist.kl.kl_divergence(dist_p, dist_q_detached)

            # free_nats = torch.full((1,), 3).to(self.device)
            # kl_pq = kl_pq * (1. - prev_done)
            kl_div = kl_pq.mean()

            # vib objective
            vib_objective = lp_reward + lp_obs + lp_df - self.vib_beta * kl_div
            # vib_objective = lp_reward + lp_df - self.vib_beta * kl_div # no obs loss

            # loss
            loss_dynamics -= vib_objective

            # loss accumulators for book keeping and plotting
            loss_reward -= lp_reward
            loss_obs -= lp_obs
            loss_df -= lp_df
            loss_kl += self.vib_beta * kl_div

            # next time step
            prev_state = curr_state.detach()
            prev_belief = curr_belief.detach()
            prev_action = curr_action
            prev_reward = curr_reward
            prev_done = curr_done


        # update dynamics model
        self.optimizer_model.zero_grad()
        loss_dynamics.backward()
        nn.utils.clip_grad_norm_(list(self.rssm.parameters()) + list(self.df_model.parameters()) + list(self.reward_model.parameters()) + \
                                list(self.observation_model.parameters()) + list(self.state_model.parameters()) , 100., norm_type=2)
        self.optimizer_model.step()

        state_list = torch.stack(state_list, dim=0)
        belief_list = torch.stack(belief_list, dim=0)


        ####################
        ## behaviour learning (using imagined rollouts over the learnt dynamics model)
        ####################

        # freeze dynamics model params
        self.freeze_model_params(self.rssm)
        self.freeze_model_params(self.df_model)
        self.freeze_model_params(self.reward_model)
        self.freeze_model_params(self.observation_model)
        self.freeze_model_params(self.state_model)

        # flatten time and batch dimension of belief_list into one - for parallel rollouts
        # and init belief state with these values
        im_curr_belief = torch.flatten(belief_list, start_dim=0, end_dim=1)
        im_curr_belief = im_curr_belief.clone().detach()

        # flatten time and batch dimension into one - for parallel rollouts
        im_curr_state = torch.flatten(state_list, start_dim=0, end_dim=1)
        im_curr_state = im_curr_state.clone().detach()

        # containers to keep required values from imagination rollouts
        state_values = [] # state values obtained from critic [v(s_0) : v(s_H-1)]
        state_values_target = [] # state values obtained from target_critic [ v(s_1) : v(s_H) ]
        rewards = [] # rewards obtained from reward model [ r(s_0) : r(s_H-1) ]
        discounts = [] # discount factors from df_model [ df(s_1) : df(s_H) ]
        log_pi = [] # [ log policy(a_0 | s_0) : log policy(a_H-1 | s_H-1) ] - used for reinforce loss
        entropy_pi = [] # [ entropy ( policy(.|s_0) ) : entropy ( policy(.|s_0) ) ] - policy entropy used for regularizing actor loss

        # start (parallel) rollout(s)
        for tau in range(self.imagination_horizon):
            im_curr_reward = self.reward_model.sample(im_curr_state, im_curr_belief) # reward obtained on transitioning into curr_state from (prev_state, prev_action)
            im_curr_df = self.df_model.sample(im_curr_state, im_curr_belief) # whether curr_state is terminal state or not ( so df[t] * s[t] )

            # im_curr_df = im_curr_df * self.df
            im_curr_df = torch.ones_like(im_curr_df) * self.df

            im_curr_action = self.get_action(im_curr_state, im_curr_belief)
            # im_curr_pi = self.actor.policy_dist(im_curr_state.detach(), im_curr_belief.detach())
            # im_curr_log_pi = im_curr_pi.log_prob(torch.round(im_curr_action.detach()))
            # im_curr_entropy_pi = im_curr_pi.entropy()

            # rollout step
            rssm_input = torch.cat((im_curr_state, im_curr_action), dim=1)
            im_next_state, im_next_belief = self.rssm.sample(im_curr_belief, rssm_input)

            # store required values
            state_values.append(self.critic_V( im_curr_state.detach(), im_curr_belief.detach() )) # [0 : H-1]

            state_values_target.append( self.target_critic_V(im_curr_state, im_curr_belief).detach() ) # [0 : H-1]

            rewards.append(im_curr_reward) # [0 : H-1]
            discounts.append(im_curr_df) # [0 : H-1]

            # log_pi.append(im_curr_log_pi) # [0 : H-1]
            # entropy_pi.append(im_curr_entropy_pi) # [0 : H-1]

            # for next step in rollout
            im_curr_state = im_next_state
            im_curr_belief = im_next_belief

        # end of rollout
        # calculate lambda return
        state_values = torch.stack(state_values[:-1], dim=0) # [0 : H-2]
        state_values_target = torch.stack(state_values_target, dim=0) # [0 : H-1]
        rewards = torch.stack(rewards, dim=0) # [0 : H-1]
        discounts = torch.stack(discounts, dim=0) # [0 : H-1]

        # log_pi = torch.stack(log_pi[:-1], dim=0) # [0 : H-2]
        # entropy_pi = torch.stack(entropy_pi[:-1], dim=0) # [0 : H-2]
        # log_pi = log_pi.unsqueeze(-1)
        # entropy_pi = entropy_pi.unsqueeze(-1)

        lambda_returns = self.calculate_lambda_return(rewards, state_values_target, discounts) # [0 : H-2]

        # cumulative product of discounts to weight the actor and critic losses
        discounts_cumprod = torch.cumprod(discounts, dim=0).detach() # [0 : H-1]
        discounts_cumprod = discounts_cumprod[:-1] # [0 : H-2]

        ## calculate critic loss - and weight by learnt discount factor
        critic_target = lambda_returns.clone().detach()
        loss_critic = F.mse_loss(state_values * torch.pow(discounts_cumprod, 0.5), critic_target * torch.pow(discounts_cumprod, 0.5), reduction='none')
        loss_critic = loss_critic.sum(dim=0).mean() # sum over horizon dim and then average over batch dim

        ## calculate actor loss - and weight by discount factor

        # loss through dynamics for actor
        loss_actor_dynamics = -lambda_returns * discounts_cumprod
        loss_actor_dynamics = loss_actor_dynamics.sum(dim=0).mean() # sum over horizon dim and then average over batch dim

        # # reinforce loss for actor
        # advantage = (lambda_returns - state_values).detach()
        # loss_actor_reinforce = (-log_pi * advantage) * discounts_cumprod
        # loss_actor_reinforce = loss_actor_reinforce.sum(dim=0).mean()

        # # policy entropy for regularization
        # policy_entropy = entropy_pi * discounts_cumprod
        # policy_entropy = policy_entropy.sum(dim=0).mean()

        # total loss for actor - weight by discount factor
        # loss_actor = (1 - self.rho) * loss_actor_dynamics + self.rho * loss_actor_reinforce - self.eta * policy_entropy
        loss_actor = loss_actor_dynamics

        # update actor
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters() , 100., norm_type=2)
        self.optimizer_actor.step()

        # update critic
        self.optimizer_critic_V.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic_V.parameters() , 100., norm_type=2)
        self.optimizer_critic_V.step()

        return loss_dynamics, loss_reward, loss_obs, loss_df, loss_kl, loss_actor, loss_critic


# main
if __name__ == '__main__':

    # hyperparams
    h_dim = 256
    s_dim = 32
    belief_dim = 128
    lr_actor = 4e-5 # 4e-5
    lr_critic = 1e-5 # 1e-5
    lr_model = 2e-4 # 2e-4
    sample_seq_len = 32 # 16 # 50 # length of contiguous sequence sampled from replay buffer (when training)
    imagination_horizon = 16 # 15 # length of imagined rollouts using the learnt dynamics model (when behaviour learning)
    vib_beta = 1. # beta - tradeoff hyperparam in vib objective
    _lambda = .95 # lambda - used to calculate lambda return
    alpha = .8 # used for kl balancing
    tau = 1e-4 # 1e-2 # used when updating target_critic_V
    rho = 0.75 # 0.25 #.95 # used for weighing actor dynamics loss and actor reinforce loss
    eta = 1e-3 # used for weighing entropy regulaization in actor loss
    df = 1 # 0.995
    batch_size = 256 # 50
    replay_buffer_size = 10**3
    num_episodes = 150
    random_seed = 1010
    record_episode = num_episodes // 5
    init_random_episodes = 32 # 5
    num_train_calls = 10 # 100
    train_episode = 1
    action_repeat = 2
    explore_minLimit = 0. 
    max_time_steps = 500
    min_reward = 0
    max_reward = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    a_dim = env.action_space.n
    o_dim = env.observation_space.shape[0]

    # hyperparam dict
    hyperparam_dict = {}
    hyperparam_dict['env'] = 'CartPole-v1'
    hyperparam_dict['algo'] = 'dreamerV2_6_10_noReinforce_constDF_FixTargetUpdate'
    # hyperparam_dict['Sdim'] = str(s_dim)
    # hyperparam_dict['Bdim'] = str(belief_dim)
    # hyperparam_dict['Hdim'] = str(h_dim)
    hyperparam_dict['lrActor'] = str(lr_actor)
    hyperparam_dict['lrCritic'] = str(lr_critic)
    hyperparam_dict['lrModel'] = str(lr_model)
    # hyperparam_dict['_lambda'] = str(_lambda)
    hyperparam_dict['L'] = str(sample_seq_len)
    hyperparam_dict['H'] = str(imagination_horizon)
    hyperparam_dict['beta'] = str(vib_beta)
    hyperparam_dict['rho'] = str(rho)
    hyperparam_dict['eta'] = str(eta)
    hyperparam_dict['df'] = str(df)
    hyperparam_dict['B'] = str(batch_size)
    hyperparam_dict['EP'] = str(num_episodes)
    hyperparam_dict['trainCalls'] = str(num_train_calls)
    hyperparam_dict['trainEP'] = str(train_episode)
    hyperparam_dict['initEP'] = str(init_random_episodes)
    # hyperparam_dict['random_seed'] = str(random_seed)
    # hyperparam_dict['actionRep'] = str(action_repeat)
    hyperparam_dict['explore_minLimit'] = str(explore_minLimit)

    # hyperparam string
    hyperstr = ""
    for k,v in hyperparam_dict.items():
        hyperstr += k + ':' + v + "_"

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # env.seed(random_seed)
    obs, info = env.reset(seed=random_seed)
    env.action_space.seed(random_seed)

    # init Dreamer agent
    agent = DreamerV2(o_dim, s_dim, a_dim, belief_dim, h_dim, sample_seq_len, max_time_steps+1, imagination_horizon, df, replay_buffer_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, tau, rho, eta, device)

    # results and stats containers
    ep_return_list = []
    loss_dynamics_list = []
    loss_reward_list = []
    loss_obs_list = []
    loss_df_list = []
    loss_kl_list = []
    loss_actor_list = []
    loss_critic_list = []

    # seed episodes
    for ep in range(init_random_episodes):
        obs, info = env.reset()
        done = 0
        ep_steps = 0
        ep_trace = []

        while done == 0:
            action_scalar = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action_scalar)
            if terminated:
                done = 1
            if truncated:
                done = 2

            action = np.zeros(a_dim)
            action[action_scalar] = 1

            # normalize reward in [0, 1]
            reward_normalized = (reward - min_reward) / (max_reward - min_reward)

            # add experience to ep trace
            ep_trace.append([obs, action, reward_normalized, done]) 

            # handle terminal state and add experience to replay buffer
            if done:
                # add terminal transitions to ep_trace
                while len(ep_trace) < (max_time_steps + 1):
                    terminal_observation = next_obs # terminal state is absorbing
                    if done == 1: # terminated
                        terminal_reward = 0 # since termination in cartpole means failure (learnable df handles this anyway)
                    else:
                        terminal_reward = 1 # since truncation in cartpole means success (learnable df handles this anyway)
                    terminal_action_scalar = env.action_space.sample() # all further actions can be random
                    terminal_action = np.zeros(a_dim)
                    terminal_action[terminal_action_scalar] = 1
                    terminal_done = done 
                    ep_trace.append([terminal_observation, terminal_action, terminal_reward, terminal_done])
                    
                # add ep_trace to replay buffer
                agent.replay_buffer.add(ep_trace)

            obs = next_obs
            ep_steps += 1


    # epsilon schedule
    epsilon_schedule = np.ones(num_episodes) * explore_minLimit
    epsilon_schedule[:int(num_episodes * 0.9)] = np.linspace(1., explore_minLimit, int(num_episodes * 0.9))


    # interactive episodes
    for ep in tqdm(range(num_episodes)):
        done = 0
        ep_return = 0
        ep_steps = 0
        ep_trace = []
        frames = []

        with torch.no_grad():

            # first observation of the episode
            observation, info = env.reset()

            # init state and action
            prev_state = torch.zeros(1, s_dim)
            prev_action = torch.zeros(1, a_dim)
            prev_belief = torch.zeros(1, belief_dim)

            while done == 0:

                # record episode
                if (ep+1) % record_episode == 0: 
                    frames.append(env.render())

                # infer state from observation using representation model
                rssm_input = torch.cat( (prev_state, prev_action), dim=1 ).to(device)
                obs_input = torch.FloatTensor(observation).unsqueeze(0).to(device)
                prev_belief = prev_belief.to(device)
                state, belief = agent.rssm.sample(prev_belief, rssm_input, obs_input)

                if ep_steps % action_repeat == 0: # action repeat
                    # sample action from the (stochastic) policy
                    action = agent.get_action(state, belief) # NOTE: this is one hot
                    # eps-greedy exploration
                    epsilon = epsilon_schedule[ep]
                    action = agent.action_exploration(action, epsilon)
                else:
                    action = prev_action

                action_numpy = action.squeeze(0).detach().cpu().numpy()
                action_scalar = np.argmax(action_numpy).squeeze() 
                next_observation, reward, terminated, truncated, _ = env.step(action_scalar)
                if terminated:
                    done = 1
                if truncated:
                    done = 2

                # normalize reward in [0, 1]
                reward_normalized = (reward - min_reward) / (max_reward - min_reward)

                # add experience to ep trace
                ep_trace.append([observation, action_numpy, reward_normalized, done]) 

                # handle terminal state and add experience to replay buffer
                if done:
                    # add terminal transitions to ep_trace
                    while len(ep_trace) < (max_time_steps + 1):
                        terminal_observation = next_observation # terminal state is absorbing
                        if done == 1: # terminated
                            terminal_reward = 0 # since termination in cartpole means failure (learnable df handles this anyway)
                        else:
                            terminal_reward = 1 # since truncation in cartpole means success (learnable df handles this anyway)
                        terminal_action_scalar = env.action_space.sample() # all further actions can be random
                        terminal_action = np.zeros(a_dim)
                        terminal_action[terminal_action_scalar] = 1
                        terminal_done = done
                        ep_trace.append([terminal_observation, terminal_action, terminal_reward, terminal_done])
                        
                    # add ep_trace to replay buffer
                    agent.replay_buffer.add(ep_trace)

                # for next step in episode
                observation = next_observation
                prev_state = state.detach().cpu()
                prev_action = action.detach().cpu()
                prev_belief = belief.detach().cpu()

                ep_return += (df ** ep_steps) * reward
                ep_steps += 1
                

        ## episode ended
        # train agent
        if ep % train_episode == 0:
            for _ in range(num_train_calls):
                l_dyn, l_rew, l_obs, l_df, l_kl, l_act, l_cri = agent.train()
                loss_dynamics_list.append(l_dyn.item())
                loss_reward_list.append(l_rew.item())
                loss_obs_list.append(l_obs.item())
                loss_df_list.append(l_df.item())
                loss_kl_list.append(l_kl.item())
                loss_actor_list.append(l_act.item())
                loss_critic_list.append(l_cri.item())

            # update critic target net
            for target_param, current_param in zip(agent.target_critic_V.parameters(), agent.critic_V.parameters()):
                target_param.data.copy_(agent.tau * current_param.data + (1 - agent.tau) * target_param.data)

        # store episode stats
        ep_return_list.append(ep_return)
        if ep % (num_episodes//10) == 0:
            print('ep:{} \t ep_return:{}'.format(ep, ep_return))

        # save episode recording 
        if (ep+1) % record_episode == 0:
            imageio.mimsave('plots/' + hyperstr + '_' + str(ep) + '.gif', frames, fps=30)



# get moving mean lists
def get_moving_mean_list(a):
    mmlist = [a[0]]
    n = 0
    st = len(a)
    for i in range(1, st):
        n += 1
        n = n % (st//20)
        prev_mean = mmlist[-1]
        new_mean = prev_mean + ((a[i] - prev_mean)/(n+1))
        mmlist.append(new_mean)
    return mmlist

# ep_returns_moving_mean = get_moving_mean_list(ep_return_list)
ep_returns_moving_mean = ep_return_list
loss_dynamics_moving_mean = get_moving_mean_list(loss_dynamics_list)
loss_reward_moving_mean = get_moving_mean_list(loss_reward_list)
loss_obs_moving_mean = get_moving_mean_list(loss_obs_list)
loss_df_moving_mean = get_moving_mean_list(loss_df_list)
loss_kl_moving_mean = get_moving_mean_list(loss_kl_list)
loss_actor_moving_mean = get_moving_mean_list(loss_actor_list)
loss_critic_moving_mean = get_moving_mean_list(loss_critic_list)


# plot results
fig, ax = plt.subplots(2,3, figsize=(15,10))

ax[0,0].plot(ep_returns_moving_mean, color='green', label='ep_return')
ax[0,0].legend()
ax[0,0].set(xlabel='episode')
ax[0,0].grid()

ax[0,1].plot(loss_dynamics_moving_mean, color='red', label='dynamics_loss')
ax[0,1].plot(loss_reward_moving_mean, color='lime', label='reward_loss')
ax[0,1].plot(loss_obs_moving_mean, color='blue', label='obs_loss')
ax[0,1].plot(loss_df_moving_mean, color='magenta', label='df_loss')
ax[0,1].plot(loss_kl_moving_mean, color='black', label='kl_loss')
ax[0,1].legend()
ax[0,1].set(xlabel='steps')
ax[0,1].grid()
ax[0,1].set_ylim([-500,500])

ax[1,0].plot(loss_actor_moving_mean, color='blue', label='actor_loss')
ax[1,0].legend()
ax[1,0].set(xlabel='steps')
ax[1,0].grid()

ax[1,1].plot(loss_critic_moving_mean, color='gray', label='critic_loss')
ax[1,1].legend()
ax[1,1].set(xlabel='steps')
ax[1,1].grid()

ax[0,2].plot(loss_df_moving_mean, color='magenta', label='df_loss')
ax[0,2].legend()
ax[0,2].set(xlabel='steps')
ax[0,2].grid()

ax[1,2].plot(loss_dynamics_moving_mean, color='red', label='dynamics_loss')
ax[1,2].legend()
ax[1,2].set(xlabel='steps')
ax[1,2].grid()
ax[1,2].set_ylim([-500,500])

plt.savefig('plots/' + hyperstr + '.png')
