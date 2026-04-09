'''
Differences in guhare implementation (from our implementation of dreamerv2_6_10):

1. n_latents = 20, n_classes = 20, so state_dim = 20*20
2. belief_dim = 200, so model_state = cat(state_dim, belief_dim) = 600, hidden_dim = 200
3. replay_buffers are flat [step_idx] (not 2d array: [episode_idx, step_idx])
4. ObsEncoder is outside the RSSM
5. Actor network has 4 linear layers with h_dim=100 and act_fn=elu. Output is categorical logits for n_actions
6. action exploration is eps_greedy with eps schedule 0.4 -> 0.05 (but its decayed at a constant rate of iter/7000)
7. Reward network has 4 linear layers with h_dim=100 and act_fn=elu. Output is just the mean of gaussian with constant std=1
8. Critic and target Critic networks are same as reward network. Note that their value output is also interpreted as mean of gaussian with constant std=1.
9. Discount network is same as reward network but output is bernouli logit
10. ObsEncoder has a convolutional backbone of 3 conv_2d layers followed by a final linear projection layer. The conv filters grow [16, 32, 64] with fixed kernel_size=3 (padding=0 but why). All act_fn=elu. Obs_input.shape = [b, c=3, h=10, w=10] with all values either 0 or 1. ObsEncoder.output.shape = [b, belief_dim]
11. ObsDecoder network has a linear proj layer followed by a deconv backbone of 3 ConvTranspose2d layers. The output is the mean of guassian with constant std=1.
12. seed episodes are collected as 4000 seed steps of type [obs, action, reward, done] (instead of episode wise). No special terminal state handling, no entry for terminal state as [terminal_obs=next_obs, terminal_action=random_action, terminal_reward, terminal_done].

13. on starting interactive episodes:
13.1 Initial state and action are set as:
        prev_rssm_state.logit = torch.zeros(1, state_dim)
        prev_rssm_state.state = torch.zeros(1, state_dim)
        prev_rssm_state.belief = torch.zeros(1, belief_dim)
        prev_action = torch.zeros(1, a_dim=num_actions)
        done = False
        obs = env.reset()
13.2 A train call is made after every 50 episode steps (note: steps, not episodes)
13.3 The target_critic is updated after every 100 episode steps with tau=1 (target_critic params are made equal to critic params in one update)
13.4 For each interaction step, we have (prev_state, prev_belief, prev_action) and o_t. We calculate (curr_state, curr_belief) as follows:
        sa_embed = fc_embed(torch.cat([prev_state, prev_action], dim=-1)) - note that only prev_state is multiplied with (1-d_t)
        curr_belief = self.rnn(sa_embed, prev_belief)
        logit_prior = self.prior_net(curr_belief)
        curr_state_prior = prior_dist.rsample(logit_prior.reshape(num_latents, num_classes)).flatten() # with straight-through
        bo = torch.cat([curr_belief, ObsEncoder(o_t)], dim=-1)
        logit_posterior = self.posterior_net(bo)
        curr_state_posterior = posterior_dist.rsample(logit_posterior.reshape(num_latents, num_classes)).flatten() # with straight-through 
        (both prior and posterior states are calculated but only posterior state is used during interaction)
13.5 Rest is as usual:
        action a_t is selected using actor distribution followed with eps_greedy exploration
        next_obs, r_t, d_t, _ = env.step(a_t)
        replay_buffer.add(o_t, a_t, r_t, d_t)
        if done:
            reinit prev_state=0, prev_belief=0, prev_action=0, done=False and start new episode by calling env.reset()
        else:
            prev_state=curr_state_posterior, prev_belief=curr_belief, prev_action=a_t, obs=next_obs

            
14. Dynamics model training:
14.1 sample from replay buffer - exactly as I was doing in dreamerv2_5_1_12_cu. Then samples are shifted. Finally samples are: 
        obs = obs[1:]
        actions = actions[:1]
        rewards = rewards[:1]
        dones = dones[:1]
            
'''




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import matplotlib.pyplot as plt
from copy import deepcopy
import gymnasium as gym
from tqdm import tqdm
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
        # self.obs_encoder_fc2 = nn.Linear(h_dim, h_dim)
        self.p_fc1 = nn.Linear(belief_dim + h_dim, h_dim)
        self.p_fc2_logits = nn.Linear(h_dim, out_dim)
        self.q_fc1 = nn.Linear(belief_dim, h_dim)
        self.q_fc2_logits = nn.Linear(h_dim, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.device = device

    # forward pass through RSSM
    def forward(self, prev_belief, x, o=None):
        logits_prior, logits_posterior = None, None

        x = self.elu(self.fc_embed(x))
        belief = self.gru_cell(x, prev_belief)
        
        h = self.elu(self.q_fc1(belief))
        logits_prior = self.q_fc2_logits(h)

        if o is not None:
            # o = self.relu(self.obs_encoder_fc1(o))
            o = self.obs_encoder_fc1(o)
            h_o = torch.cat((belief, o), dim=1)
            h = self.elu(self.p_fc1(h_o))
            logits_posterior = self.p_fc2_logits(h)

        return belief, logits_prior, logits_posterior

    # to draw sample from the learnt probabilistic model
    def sample(self, prev_belief, x, o=None):
        state_prior, state_posterior = None, None
        belief, logits_prior, logits_posterior = self.forward(prev_belief, x, o)

        def rsample(logits):
            batch_shape, item_shape = logits.shape[:-1], logits.shape[-1]

            logits = logits.reshape(*batch_shape, n_latents, n_classes)
            dis = tdist.OneHotCategorical(logits=logits)
            state = dis.sample()
            state = state + dis.probs - dis.probs.clone().detach() # for straight through gradient

            state = state.flatten(start_dim=-2, end_dim=-1)
            state = state.reshape(*batch_shape, item_shape)
            return state 

        if logits_prior is not None:
            state_prior = rsample(logits_prior)

        if logits_posterior is not None:
            state_posterior = rsample(logits_posterior)

        return belief, state_prior, state_posterior, logits_prior, logits_posterior


    # formulates the one_hot_categorical distribution from logits
    def get_dist(self, logits, detach=False):
        if detach:
            logits = logits.detach()
        logits = logits.flatten(start_dim=0, end_dim=-2)
        logits = logits.reshape(logits.shape[0], n_latents, n_classes)
        dis = tdist.Independent(tdist.OneHotCategoricalStraightThrough(logits=logits), 1)
        return dis



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
        x = torch.cat((state, belief), dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        # logstd = self.fc3_std(h).clip(minClip, maxClip)
        # std = torch.exp(logstd)
        std = torch.ones_like(mean) * STD
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
        dis = tdist.independent.Independent(tdist.Normal(mean, std), 1)
        lp = dis.log_prob(y)
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
        x = torch.cat((state, belief), dim=-1)
        h = self.relu(self.fc1(x))
        # h = self.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits

    # to draw sample from the learnt probabilistic model
    def sample(self, state, belief):
        logits = self.forward(state, belief)
        dis = tdist.Bernoulli(logits=logits)
        out = dis.sample()
        # for straight through gradient
        out = out + dis.probs - dis.probs.clone().detach()
        return out

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        logits = self.forward(state, belief)
        dis = tdist.independent.Independent(tdist.Bernoulli(logits=logits), 1)
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
        x = torch.cat((state, belief), dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        logits = self.fc5_logits(h)
        return logits

    # returns the policy as one_hot_categorical distribution
    def policy_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = tdist.OneHotCategorical(logits=logits)
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
        x = torch.cat((state, belief), dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        val = self.fc4(h)
        return val



# replay buffer
class ReplayBuffer:
    def __init__(self, buf_size, seq_len, batch_size, o_dim, a_dim, max_ep_steps, frac, device):
        self.buf_size = buf_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buf_observation = np.zeros((buf_size, o_dim))
        self.buf_action = np.zeros((buf_size, a_dim))
        self.buf_reward = np.zeros((buf_size, 1))
        self.buf_done = np.zeros((buf_size, 1))
        self.n_items = 0
        self.device = device
        self.max_ep_steps = max_ep_steps
        self.frac = frac

    def add(self, oar_tuple):
        observation, action, reward, done = oar_tuple
        index = self.n_items % self.buf_size
        self.buf_observation[index] = observation
        self.buf_action[index] = action
        self.buf_reward[index] = reward
        self.buf_done[index] = done
        self.n_items += 1

    def sample(self):
        limit = self.n_items
        if limit > self.buf_size:
            limit = self.buf_size

        idx_list = []
        while len(idx_list) < self.batch_size:
            idx_start = np.random.randint(0, limit)
            idx_chunk = np.arange(idx_start, idx_start+self.seq_len) % self.buf_size
            curr_idx = self.n_items % self.buf_size
            # don't append sample chunks that are part old and part new
            if not (curr_idx in idx_chunk):
                 # ensure atleast frac% samples have a learn_done (premature termination states) - to encourage experiencing premature terminal states (required for learning df_model)
                if len(idx_list) < (self.batch_size * self.frac):
                     if (1 in self.buf_done[idx_chunk]):
                         idx_list.append(idx_chunk)
                else:
                    idx_list.append(idx_chunk)

        idx = np.array(idx_list)
        idx = idx.T # first dimension should be time_step and second dimension should be batch
        observation = torch.FloatTensor(self.buf_observation[idx]).to(self.device)
        action = torch.FloatTensor(self.buf_action[idx]).to(self.device)
        reward = torch.FloatTensor(self.buf_reward[idx]).to(self.device)
        done = torch.FloatTensor(self.buf_done[idx]).to(self.device)
        return (observation, action, reward, done)



# DreamerV2
class DreamerV2(nn.Module):
    def __init__(self, o_dim, s_dim, a_dim, belief_dim, h_dim, seq_len, imagination_horizon, df, buf_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, tau, rho, eta, max_ep_steps, frac, device):
        super().__init__()
        self.actor = Actor(s_dim, belief_dim, a_dim, h_dim).to(device)
        self.critic_V = Critic_V(s_dim, belief_dim, h_dim).to(device)
        self.target_critic_V = deepcopy(self.critic_V)
        self.replay_buffer = ReplayBuffer(buf_size, seq_len, batch_size, o_dim, a_dim, max_ep_steps, frac, device)
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
        self.tanh = nn.Tanh()
        self.seq_len = seq_len
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
        return action, policy

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


    def calculate_lambda_return(self, rewards, discounts, state_values_target):
        """
        Input:
        # rewards obtained from reward model - r[t+1 : t+H+1]
        # df values obtained from df model - df[t+1 : t+H+1]
        # state values obtained from target_critic model - v[t+1 : t+H+1]

        Output:
        # V_lambda[t+1 : t+H]
        """
        lambda_returns = []

        # prepare accumulator bootstrap value 
        accumulator = state_values_target[-1] # v[t+H+1]

        # adjust tensors
        rewards = rewards[:-1] # r[t+1 : t+H]
        discounts = discounts[:-1] # df[t+1 : t+H]
        state_values_target = state_values_target[1:] # v[t+2 : t+H+1]

        for t in range(len(rewards)-1, -1, -1): # t is just going from last element to first element, since all arrays are of length H
            accumulator = rewards[t] + discounts[t] * ( (1 - self._lambda)*state_values_target[t] + self._lambda*accumulator )
            # V_lambda[j] = reward[j] + df[j] * ( (1-lambda) * V_t[j+1] + lambda * V_lambda[j+1] )
            lambda_returns = [accumulator] + lambda_returns
        lambda_returns = torch.stack(lambda_returns, dim=0) # V_lambda[t+1 : t+H]
        return lambda_returns


    def train(self):

        # sample from replay buffer
        observation, action, reward, done = self.replay_buffer.sample() 

        # shift samples
        observation = observation[1:] # o[t : t+H]
        reward = reward[:-1] # r[t-1 : t+H-1]
        action = action[:-1] # a[t-1 : t+H-1]
        done = done[:-1] # d[t-1 : t+H-1]


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

        # list tensors to store 
        belief_list = []
        state_posterior_list = []
        logits_posterior_list = []
        logits_prior_list = []

        # fixed prev_state and prev_action (to intialize the first state)
        prev_state = torch.zeros(self.batch_size, self.s_dim).to(self.device)
        prev_belief = torch.zeros(self.batch_size, self.belief_dim).to(self.device)

        # rssm rollout 
        for t in range(observation.shape[0]): 

            # reset if done
            prev_state = prev_state * (1. - done[t])
            prev_belief = prev_belief * (1. - done[t])
            prev_action = action[t] * (1. - done[t])

            # get curr state
            rssm_input = torch.cat((prev_state, prev_action), dim=-1)
            curr_belief, state_prior, state_posterior, logits_prior, logits_posterior = self.rssm.sample(prev_belief, rssm_input, observation[t])

            belief_list.append(curr_belief)
            state_posterior_list.append(state_posterior)
            logits_posterior_list.append(logits_posterior)
            logits_prior_list.append(logits_prior)

            # for next step (no detach)
            prev_state = state_posterior 
            prev_belief = curr_belief


        # rollout ended - stack lists into tensors
        beliefs = torch.stack(belief_list[:-1], dim=0) # h[t : t+H-1]
        states = torch.stack(state_posterior_list[:-1], dim=0) # s[t : t+H-1]
        logits_posterior = torch.stack(logits_posterior_list[:-1], dim=0) 
        logits_prior = torch.stack(logits_prior_list[:-1], dim=0) 


        ## calculate loss terms 

        # reward loss
        lp_reward = self.reward_model.log_prob(states, beliefs, reward[1:]) # logp( r[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_reward = lp_reward.mean()

        # observation loss (reconstruction)
        lp_obs = self.observation_model.log_prob(states, beliefs, observation[:-1]) # logp( o[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_obs = lp_obs.mean() * obs_loss_scale

        # df loss
        lp_df = self.df_model.log_prob(states, beliefs, (1. - done[1:])) # logp( d[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_df = lp_df.mean() * df_loss_scale
            
        # KL divergence - using KL balancing
        dist_p = self.rssm.get_dist(logits_posterior)
        dist_q = self.rssm.get_dist(logits_prior)
        dist_p_detached = self.rssm.get_dist(logits_posterior, detach=True)
        dist_q_detached = self.rssm.get_dist(logits_prior, detach=True)
        kl_pq = self.alpha * tdist.kl.kl_divergence(dist_p_detached, dist_q) + \
                (1 - self.alpha) * tdist.kl.kl_divergence(dist_p, dist_q_detached)
        kl_div = kl_pq.mean()

        # vib objective
        vib_objective = lp_reward + lp_obs + lp_df - self.vib_beta * kl_div

        # loss
        loss_dynamics = -vib_objective

        # update dynamics model
        self.optimizer_model.zero_grad()
        loss_dynamics.backward()
        nn.utils.clip_grad_norm_(list(self.rssm.parameters()) + list(self.df_model.parameters()) + list(self.reward_model.parameters()) + \
                                list(self.observation_model.parameters()) + list(self.state_model.parameters()) , 100., norm_type=2)
        self.optimizer_model.step()

        # loss accumulators for book keeping and plotting
        loss_reward = -lp_reward
        loss_obs = -lp_obs
        loss_df = -lp_df
        loss_kl = self.vib_beta * kl_div


        ####################
        ## behaviour learning (using imagined rollouts over the learnt dynamics model)
        ####################

        # freeze dynamics model params
        self.freeze_model_params(self.rssm)
        self.freeze_model_params(self.df_model)
        self.freeze_model_params(self.reward_model)
        self.freeze_model_params(self.observation_model)
        self.freeze_model_params(self.state_model)

        # list tensors to store 
        im_belief_list = []
        im_state_list = []
        log_pi_list = []
        entropy_pi_list = []

        # flatten time and batch dimension into one - for parallel imagination rollouts
        # and init imagination states with these values
        im_curr_belief = torch.flatten(beliefs, start_dim=0, end_dim=1).clone().detach()
        im_curr_state = torch.flatten(states, start_dim=0, end_dim=1).clone().detach()

        # imagination rollout 
        for tau in range(self.imagination_horizon):

            # get actor action, also get policy logprob and entropy
            im_curr_action, policy = self.get_action(im_curr_state, im_curr_belief)
            curr_log_pi = policy.log_prob(torch.round(im_curr_action.detach()))
            curr_entropy_pi = policy.entropy()

            # rssm step
            rssm_input = torch.cat((im_curr_state, im_curr_action), dim=1)
            im_next_belief, im_next_state_prior, _, _, _ = self.rssm.sample(im_curr_belief, rssm_input)

            # store
            im_belief_list.append(im_next_belief)
            im_state_list.append(im_next_state_prior)
            log_pi_list.append(curr_log_pi)
            entropy_pi_list.append(curr_entropy_pi)

            # for next step (no detach)
            im_curr_belief = im_next_belief 
            im_curr_state = im_next_state_prior

        # rollout ended - stack lists to tensors
        im_beliefs = torch.stack(im_belief_list, dim=0) # h[t+1 : t+1+H]
        im_states = torch.stack(im_state_list, dim=0) # s[t+1 : t+1+H]
        log_pi = torch.stack(log_pi_list, dim=0) # logp[t : t+H]
        entropy_pi = torch.stack(entropy_pi_list, dim=0) # entropy[t : t+H]

        # calculate rewards, discounts, state_values_target and bootstrap_value for imagined rollout 
        rewards = self.reward_model.sample(im_states, im_beliefs) # r[t+1 : t+H+1]
        discounts = self.df * self.df_model.sample(im_states, im_beliefs) # df[t+1 : t+H+1]
        state_values_target = self.target_critic_V(im_states, im_beliefs).detach()  # v[t+1 : t+H+1]

        # calculate lambda returns 
        lambda_returns = self.calculate_lambda_return(rewards, discounts, state_values_target) # v_lambda[t+1 : t+H]

        ## calculate actor loss

        # create discount cumprods
        discounts_shifted = torch.cat((torch.ones_like(discounts[:1]), discounts[1:]), dim=0) # [df[t+1] = 1] + df[t+2 : t+H+1]
        discounts_cumprod = torch.cumprod(discounts_shifted[:-1], dim=0).detach() # df[t+1 : t+H]

        # loss through dynamics 
        loss_actor_dynamics = -lambda_returns * discounts_cumprod
        loss_actor_dynamics = loss_actor_dynamics.mean(dim=1).sum() # sum over horizon dim and mean over batch dim

        # reinforce loss
        advantage = (lambda_returns - state_values_target[:-1]).detach() # advantage[t+1 : t+H]
        loss_actor_reinforce = (-log_pi[1:].unsqueeze(-1) * advantage) * discounts_cumprod 
        loss_actor_reinforce = loss_actor_reinforce.mean(dim=1).sum()

        # policy entropy for regularization (and encourage exploration)
        policy_entropy = entropy_pi[1:].unsqueeze(-1) * discounts_cumprod
        policy_entropy = policy_entropy.mean(dim=1).sum()

        # total loss for actor - weight by discount factor
        loss_actor = (1 - self.rho) * loss_actor_dynamics + self.rho * loss_actor_reinforce - self.eta * policy_entropy

        ## calculate critic loss 

        state_values = self.critic_V( im_states.detach(), im_beliefs.detach() )[:-1] # v[t+1 : t+H+1]
        critic_target = lambda_returns.clone().detach()
        loss_critic = F.mse_loss(state_values * torch.pow(discounts_cumprod, 0.5), critic_target * torch.pow(discounts_cumprod, 0.5))

        ## update actor and critic

        self.optimizer_actor.zero_grad()
        self.optimizer_critic_V.zero_grad()

        loss_actor.backward()
        loss_critic.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters() , 100., norm_type=2)
        nn.utils.clip_grad_norm_(self.critic_V.parameters() , 100., norm_type=2)

        self.optimizer_actor.step()
        self.optimizer_critic_V.step()        

        return loss_dynamics, loss_reward, loss_obs, loss_df, loss_kl, loss_actor, loss_critic


# main
if __name__ == '__main__':

    # hyperparams
    h_dim = 200
    n_latents = 20
    n_classes = 20
    s_dim = n_latents * n_classes 
    belief_dim = 200
    lr_actor = 4e-5 # 4e-5
    lr_critic = 1e-4 # 1e-4
    lr_model = 2e-4 # 2e-4
    sample_seq_len = 50 # length of contiguous sequence sampled from replay buffer (when training)
    imagination_horizon = 15 # length of imagined rollouts using the learnt dynamics model (when behaviour learning)
    vib_beta = 1 # beta - tradeoff hyperparam in vib objective
    _lambda = .95 # lambda - used to calculate lambda return
    alpha = .8 # used for kl balancing
    tau = 1 # 1e-2 # used when updating target_critic_V
    target_critic_update_step = 100 # 1
    rho = 0.05 # used for weighing actor dynamics loss and actor reinforce loss
    eta = 1e-3 # used for weighing entropy regulaization in actor loss
    df = 0.995
    df_loss_scale = 1
    obs_loss_scale = 1 # 1e-1
    frac = 0.5 # 0.25
    STD = 1
    minClip, maxClip = -2, 2
    random_seed = 10
    batch_size = 256 # 512
    replay_buffer_size = 3000 # 10**6
    num_episodes = 200
    init_random_episodes = 5
    num_train_calls = 15 # 10 # 50  
    train_episode = 1
    record_episode = num_episodes // 5
    action_repeat = 2
    explore_minLimit, explore_maxLimit = 0.05, 0.4
    max_ep_steps = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    a_dim = env.action_space.n
    o_dim = env.observation_space.shape[0]

    # hyperparam dict
    hyperparam_dict = {}
    hyperparam_dict['env'] = 'CartPole-v1'
    hyperparam_dict['algo'] = 'dreamerV2_6_10_guhare'
    hyperparam_dict['Sdim'] = str(s_dim)
    hyperparam_dict['Bdim'] = str(belief_dim)
    hyperparam_dict['Hdim'] = str(h_dim)
    # hyperparam_dict['lrActor'] = str(lr_actor)
    # hyperparam_dict['lrCritic'] = str(lr_critic)
    # hyperparam_dict['lrModel'] = str(lr_model)
    # hyperparam_dict['_lambda'] = str(_lambda)
    # hyperparam_dict['L'] = str(sample_seq_len)
    # hyperparam_dict['H'] = str(imagination_horizon)
    hyperparam_dict['beta'] = str(vib_beta)
    hyperparam_dict['rho'] = str(rho)
    hyperparam_dict['eta'] = str(eta)
    hyperparam_dict['tau'] = str(tau)
    # hyperparam_dict['df'] = str(df)
    # hyperparam_dict['dfScale'] = str(df_loss_scale)
    hyperparam_dict['obsScale'] = str(obs_loss_scale)
    hyperparam_dict['B'] = str(batch_size)
    hyperparam_dict['EP'] = str(num_episodes)
    hyperparam_dict['trCalls'] = str(num_train_calls)
    hyperparam_dict['trEP'] = str(train_episode)
    hyperparam_dict['initEP'] = str(init_random_episodes)
    # hyperparam_dict['VTupdateSt'] = str(target_critic_update_step)
    # hyperparam_dict['maxSt'] = str(max_time_steps)
    # hyperparam_dict['random_seed'] = str(random_seed)
    # hyperparam_dict['actionRep'] = str(action_repeat)
    hyperparam_dict['buffSz'] = str(replay_buffer_size)
    # hyperparam_dict['xploreMin'] = str(explore_minLimit)
    # hyperparam_dict['xploreMax'] = str(explore_maxLimit)
    hyperparam_dict['frac'] = str(frac)
    # hyperparam_dict['minClip'] = str(minClip)
    # hyperparam_dict['maxClip'] = str(maxClip)
    hyperparam_dict['std'] = str(STD)

    # hyperparam string
    hyperstr = ""
    for k,v in hyperparam_dict.items():
        hyperstr += k + ':' + v + "_"

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    obs, info = env.reset(seed=random_seed)
    env.action_space.seed(random_seed)

    # init Dreamer agent
    agent = DreamerV2(o_dim, s_dim, a_dim, belief_dim, h_dim, sample_seq_len, imagination_horizon, df, replay_buffer_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, tau, rho, eta, max_ep_steps, frac, device)

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
        done = False
        ep_steps = 0

        while not done:
            action_scalar = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action_scalar)
            done = terminated or truncated

            action = np.zeros(a_dim)
            action[action_scalar] = 1

            oar_tuple = [obs, action, reward, done]
            agent.replay_buffer.add(oar_tuple)

            obs = next_obs
            ep_steps += 1


    # epsilon schedule
    epsilon_schedule = np.ones(num_episodes) * explore_minLimit
    epsilon_schedule[:int(num_episodes * 0.9)] = np.linspace(explore_maxLimit, explore_minLimit, int(num_episodes * 0.9))


    # interactive episodes
    total_steps = 0
    for ep in tqdm(range(num_episodes)):
        done = False
        ep_return = 0
        ep_steps = 0
        frames = []

        # first observation of the episode
        observation, info = env.reset()

        # init state and action
        prev_state = torch.zeros(1, s_dim)
        prev_action = torch.zeros(1, a_dim)
        prev_belief = torch.zeros(1, belief_dim)

        while not done:

            # record episode
            if (ep+1) % record_episode == 0: 
                frames.append(env.render())

            # infer state from observation using representation model
            rssm_input = torch.cat( (prev_state, prev_action), dim=1 ).to(device)
            obs_input = torch.FloatTensor(observation).unsqueeze(0).to(device)
            prev_belief = prev_belief.to(device)
            belief, _, state, _, _ = agent.rssm.sample(prev_belief, rssm_input, obs_input)

            if ep_steps % action_repeat == 0: # action repeat
                # sample action from the (stochastic) policy
                action, policy = agent.get_action(state, belief)
                # eps-greedy exploration
                epsilon = epsilon_schedule[ep]
                action = agent.action_exploration(action, epsilon)
            else:
                action = prev_action

            action_numpy = action.squeeze(0).detach().cpu().numpy()
            action_scalar = np.argwhere(action_numpy > 0.9).squeeze()
            next_observation, reward, terminated, truncated, _ = env.step(action_scalar)
            done = terminated or truncated

            oar_tuple = [observation, action_numpy, reward, done]
            agent.replay_buffer.add(oar_tuple)

            # for next step in episode
            observation = next_observation
            prev_state = state.detach().cpu()
            prev_action = action.detach().cpu()
            prev_belief = belief.detach().cpu()

            ep_return += (df ** ep_steps) * reward
            ep_steps += 1
            total_steps += 1

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

        # update critic
        if total_steps % target_critic_update_step == 0:
            with torch.no_grad():
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
# ax[0,1].set_ylim([-500,500])

ax[1,0].plot(loss_actor_moving_mean, color='blue', label='actor_loss')
ax[1,0].legend()
ax[1,0].set(xlabel='steps')
ax[1,0].grid()

ax[1,1].plot(loss_critic_moving_mean, color='gray', label='critic_loss')
ax[1,1].legend()
ax[1,1].set(xlabel='steps')
ax[1,1].grid()

ax[0,2].plot(loss_df_moving_mean, color='magenta', label='df_loss')
ax[0,2].plot(loss_kl_moving_mean, color='black', label='kl_loss')
ax[0,2].legend()
ax[0,2].set(xlabel='steps')
ax[0,2].grid()

ax[1,2].plot(loss_dynamics_moving_mean, color='red', label='dynamics_loss')
ax[1,2].plot(loss_reward_moving_mean, color='lime', label='reward_loss')
ax[1,2].plot(loss_obs_moving_mean, color='blue', label='obs_loss')
ax[1,2].legend()
ax[1,2].set(xlabel='steps')
ax[1,2].grid()
# ax[1,2].set_ylim([-500,500])

plt.savefig('plots/' + hyperstr + '.png')