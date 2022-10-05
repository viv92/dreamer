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
import gym
from tqdm import tqdm
from torchviz import make_dot
from copy import deepcopy
from graphviz import Source

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
        x = self.elu(self.fc_embed(x))
        belief = self.gru_cell(x, prev_belief)
        if o is None:
            h = self.elu(self.q_fc1(belief))
            logits = self.q_fc2_logits(h)
        else:
            # o = self.relu(self.obs_encoder_fc1(o))
            o = self.obs_encoder_fc1(o)
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
        logstd = self.fc3_std(h).clip(-2, 2)
        std = torch.exp(logstd)
        # std = .25
        return mean, std

    # to draw sample from the learnt probabilistic model
    def sample(self, state, belief):
        mean, std = self.forward(state, belief)
        eps = tdist.Normal(0, 1).sample()
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
        logits = self.relu(self.fc3(h))
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
                 # ensure atleast 50% samples have a done (termination state) - to encourage experiencing terminal state (required for learning df_model)
                if len(idx_list) < (self.batch_size * self.frac):
                     if (1 in self.buf_done[idx_chunk]):
                         idx_list.append(idx_chunk)
                else:
                    idx_list.append(idx_chunk)

        # idx = np.array([np.arange(x, x + self.seq_len) for x in idx_start_list])
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
        self.optimizer_model = torch.optim.Adam(params=list(self.rssm.parameters()) + list(self.df_model.parameters()) + list(self.reward_model.parameters()) + \
                                                       list(self.observation_model.parameters()) + list(self.state_model.parameters()), lr=lr_model)
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


    def get_action(self, state, belief, epsilon):
        policy = self.actor.policy_dist(state, belief)
        action = policy.sample()
        # for straight through gradient
        action = action + policy.probs - policy.probs.clone().detach()
        # exploration
        batch_size = action.shape[0]
        random_actions = torch.randint(0, self.a_dim, (batch_size,)).to(device)
        random_actions_onehot = F.one_hot(random_actions, num_classes=self.a_dim)
        if torch.rand(1) < epsilon:
            action = random_actions_onehot.float()
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
        # rewards obtained from reward model - r[1] : r[H-1]
        # state values obtained from target_critic model - v_t[1] : v_t[H-1]
        # df values obtained from df model - df[1] : df[H-1]

        Output:
        # V_lambda[0] : V_lambda[H-2]
        """
        lambda_returns = []
        accumulator = state_values_target[-1] # V_t[H-1]
        for t in range(self.imagination_horizon-2, -1, -1): # t is just going from last element to first element, since all arrays are of length H-1
            accumulator = rewards[t] + discounts[t] * ( (1 - self._lambda)*state_values_target[t] + self._lambda*accumulator )
            # V_lambda[H-2] = reward[H-1] + df[H-1] * ( (1-lambda) * V_t[H-1] + lambda * V_lambda[H-1] )
            lambda_returns = [accumulator] + lambda_returns
        lambda_returns = torch.stack(lambda_returns, dim=0)
        return lambda_returns


    def train(self, epsilon):
        observation, action, reward, done = self.replay_buffer.sample() # sample = [o_t-1, a_t-1, r_t-1, d_t-1]
        observation = observation[1:] # observation index is shifted
        # rest indices are not shifted, just clipped (though this clipping is not needed as its taken care by iteration limit of seq_len - 1)
        reward = reward[:-1]
        action = action[:-1]
        done = done[:-1]

        # shifted sample (aligns with theory) = [o_t, a_t-1, r_t-1, d_t-1]

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

        ## fixed prev_state and prev_action (to intialize the first state)
        prev_state = torch.zeros(self.batch_size, self.s_dim).to(self.device)
        prev_action = torch.zeros(self.batch_size, self.a_dim).to(self.device) # action should also be reset to zero - as the decided dummy value
        prev_belief = torch.zeros(self.batch_size, self.belief_dim).to(self.device)
        prev_prev_done = torch.ones(self.batch_size, 1).to(self.device) # prev_prev_done only usesd for reseting prev_state for new episode

        # start training steps
        for t in range(self.seq_len-1):
            curr_obs = observation[t]
            prev_action = action[t]
            prev_reward = reward[t] # prev_reward is reward obtained on transitioning into curr_state from (prev_state, prev_action)
            prev_done = done[t] # prev_done tells whether curr_state is terminal or not (also, this implies using df[t] * V(s_t) when discounting state values )

            # reset if curr_obs is from new episode (implied by prev_prev_done =  1)
            prev_state = prev_state * (1. - prev_prev_done)
            prev_action = prev_action * (1. - prev_prev_done)
            prev_belief = prev_belief * (1. - prev_prev_done)

            rssm_input = torch.cat((prev_state, prev_action), dim=1)
            curr_state, curr_belief = self.rssm.sample(prev_belief, rssm_input, curr_obs)

            state_list.append(curr_state.clone().detach()) # s_t - store state for imagination rollout
            belief_list.append(curr_belief.clone().detach()) # h_t - store belief for imagination rollout

            # log prob q(r_t | s_t)
            lp_reward = self.reward_model.log_prob(curr_state, curr_belief, prev_reward)  # no need to skip reward loss when curr state is first state of new episode (taken care by the explicitly added terminal state)
            # print('lp_reward.shape: ', lp_reward.shape)
            lp_reward = lp_reward.sum(dim=1).mean()

            # log prob q(o_t | s_t)
            lp_obs = self.observation_model.log_prob(curr_state, curr_belief, curr_obs)
            # print('lp_obs.shape: ', lp_obs.shape)
            lp_obs = lp_obs.sum(dim=1).mean()

            # log prob q(df_t | s_t)
            df_target = 1. - prev_done
            lp_df = self.df_model.log_prob(curr_state, curr_belief, df_target)
            # print('lp_df.shape: ', lp_df.shape)
            lp_df = lp_df.sum(dim=1).mean()

            # KL divergence - using KL balancing
            dist_p = self.rssm.get_dist(prev_belief, rssm_input, curr_obs)
            dist_q = self.rssm.get_dist(prev_belief, rssm_input)
            dist_p_detached = self.rssm.get_dist_detached(prev_belief, rssm_input, curr_obs)
            dist_q_detached = self.rssm.get_dist_detached(prev_belief, rssm_input)
            kl_pq = self.alpha * tdist.kl.kl_divergence(dist_p_detached, dist_q) + \
                    (1 - self.alpha) * tdist.kl.kl_divergence(dist_p, dist_q_detached)


            # free_nats = torch.full((1,), 3).to(self.device)
            kl_pq = kl_pq # * (1. - prev_done)
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
            prev_state = curr_state
            # prev_action = curr_action
            prev_belief = curr_belief
            prev_prev_done = prev_done


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
        curr_belief = torch.flatten(belief_list, start_dim=0, end_dim=1)
        im_curr_belief = curr_belief.clone().detach()

        # flatten time and batch dimension into one - for parallel rollouts
        curr_state = torch.flatten(state_list, start_dim=0, end_dim=1)
        im_curr_state = curr_state.clone().detach()

        # # get the start state for the imagination rollout
        # im_curr_action = self.get_action(im_curr_state)
        # rssm_input = torch.cat((im_curr_state, im_curr_action), dim=1)
        # im_next_state, im_next_belief = self.rssm.sample(im_curr_belief, rssm_input)
        # im_curr_state = im_next_state
        # im_curr_belief = im_next_belief

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
            im_curr_df = im_curr_df * self.df
            # im_curr_df = self.df_model.sample(im_curr_state, im_curr_belief) * self.df

            im_curr_action = self.get_action(im_curr_state, im_curr_belief, epsilon)
            im_curr_pi = self.actor.policy_dist(im_curr_state, im_curr_belief)
            im_curr_log_pi = im_curr_pi.log_prob(torch.round(im_curr_action.detach()))
            im_curr_entropy_pi = im_curr_pi.entropy()
            # rollout step
            rssm_input = torch.cat((im_curr_state, im_curr_action), dim=1)
            im_next_state, im_next_belief = self.rssm.sample(im_curr_belief, rssm_input)
            # store required values
            state_values.append(self.critic_V( im_curr_state.clone().detach(), im_curr_belief.clone().detach() )) # [0 : H-1]
            state_values_target.append( self.target_critic_V(im_next_state, im_next_belief).detach() ) # [1 : H]
            rewards.append(im_curr_reward) # [0 : H-1]
            discounts.append(im_curr_df) # [0 : H-1]
            log_pi.append(im_curr_log_pi) # [0 : H-1]
            entropy_pi.append(im_curr_entropy_pi) # [0 : H-1]
            # for next step in rollout
            im_curr_state = im_next_state
            im_curr_belief = im_next_belief

        # end of rollout
        # calculate lambda return
        state_values = torch.stack(state_values[:-1], dim=0) # [0 : H-2]
        state_values_target = torch.stack(state_values_target[:-1], dim=0) # [1 : H-1]
        rewards = torch.stack(rewards[1:], dim=0) # [1 : H-1]
        discounts = torch.stack(discounts[1:], dim=0) # [1 : H-1]
        log_pi = torch.stack(log_pi[:-1], dim=0) # [0 : H-2]
        entropy_pi = torch.stack(entropy_pi[:-1], dim=0) # [0 : H-2]
        log_pi = log_pi.unsqueeze(-1)
        entropy_pi = entropy_pi.unsqueeze(-1)

        lambda_returns = self.calculate_lambda_return(rewards, state_values_target, discounts) # [0 : H-2]

        # cumulative product of discounts to weight the actor and critic losses
        discounts_shifted = torch.cat((torch.ones_like(discounts[:1]), discounts[:-1]), dim=0) # since discounts[t] should be multiplied to V(s_t)
        discounts_cumprod = torch.cumprod(discounts_shifted, dim=0).detach()

        ## calculate critic loss - and weight by learnt discount factor
        critic_target = lambda_returns.clone().detach()
        loss_critic = F.mse_loss(state_values * torch.pow(discounts_cumprod, 0.5), critic_target * torch.pow(discounts_cumprod, 0.5))

        ## calculate actor loss - and weight by discount factor

        # loss through dynamics for actor
        loss_actor_dynamics = -lambda_returns * discounts_cumprod
        loss_actor_dynamics = loss_actor_dynamics.sum(dim=0).mean()

        # reinforce loss for actor
        advantage = (lambda_returns - state_values).detach()
        loss_actor_reinforce = (-log_pi * advantage) * discounts_cumprod
        loss_actor_reinforce = loss_actor_reinforce.sum(dim=0).mean()

        # policy entropy for regularization
        policy_entropy = entropy_pi * discounts_cumprod
        policy_entropy = policy_entropy.sum(dim=0).mean()

        # total loss for actor - weight by discount factor
        loss_actor = (1 - self.rho) * loss_actor_dynamics + self.rho * loss_actor_reinforce - self.eta * policy_entropy

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

        # update critic target net
        for target_param, current_param in zip(self.target_critic_V.parameters(), self.critic_V.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)

        return loss_dynamics, loss_reward, loss_obs, loss_df, loss_kl, loss_actor, loss_critic


# main
if __name__ == '__main__':

    # hyperparams
    h_dim = 256
    s_dim = 30
    belief_dim = 200
    lr_actor = 4e-5 # 4e-5
    lr_critic = 1e-5 # 1e-5
    lr_model = 2e-4 # 2e-4
    sample_seq_len = 50 # length of contiguous sequence sampled from replay buffer (when training)
    imagination_horizon = 15 # length of imagined rollouts using the learnt dynamics model (when behaviour learning)
    vib_beta = 1. # beta - tradeoff hyperparam in vib objective
    _lambda = .95 # lambda - used to calculate lambda return
    alpha = .8 # used for kl balancing
    tau = 1e-2 # used when updating target_critic_V
    rho = 0.25 #.95 # used for weighing actor dynamics loss and actor reinforce loss
    eta = 1e-3 # used for weighing entropy regulaization in actor loss
    df = 0.995
    frac = 0.75
    batch_size = 50
    replay_buffer_size = 10**6
    num_episodes = 200
    random_seed = 0
    render_final_episodes = True
    init_random_episodes = 5
    num_train_calls = 10 # 100
    action_repeat = 2
    explore20 = 0.03

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('LunarLander-v2')
    a_dim = env.action_space.n
    o_dim = env.observation_space.shape[0]
    max_ep_steps = env._max_episode_steps

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
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
        obs = env.reset()
        done = False
        ep_steps = 0

        while not done:
            action_scalar = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action_scalar)
            # used to differentiate between goal_terminal_state and timeout_terminal_state - not used in dreamer
            # done = float(done) if ep_steps < env._max_episode_steps else 0

            # add experience to replay buffer
            action = np.zeros(a_dim)
            action[action_scalar] = 1
            oar_tuple = [obs, action, reward, done] # done used to denote the break in dynamics between episodes
            agent.replay_buffer.add(oar_tuple)

            if done:
                # add terminal transition to the replay buffer
                terminal_observation = next_obs
                terminal_reward = reward * 0 # dummy value (not used for reward loss since we skip)
                terminal_action = action * 0 # dummy value / action reset value
                terminal_done = 0 # this is done flag for first observation of next episode
                oar_tuple = [terminal_observation, terminal_action, terminal_reward, terminal_done]
                agent.replay_buffer.add(oar_tuple)

            obs = next_obs
            ep_steps += 1


    # epsilon schedule
    epsilon_schedule = np.ones(num_episodes) * explore20
    epsilon_schedule[:int(num_episodes * 0.8)] = np.linspace(1., explore20, int(num_episodes * 0.8))

    # interactive episodes
    for ep in tqdm(range(num_episodes)):
        done = False
        ep_return = 0
        ep_steps = 0
        # for eps-greedy exploration
        epsilon = epsilon_schedule[ep]

        # first observation of the episode
        observation = env.reset()

        # init state and action
        prev_state = torch.zeros(1, s_dim)
        prev_action = torch.zeros(1, a_dim)
        prev_belief = torch.zeros(1, belief_dim)

        while not done:
            if render_final_episodes and (ep > (num_episodes - 10)):
            # if render_final_episodes and (ep % 1 == 0):
                env.render()

            # infer state from observation using representation model
            rssm_input = torch.cat( (prev_state, prev_action), dim=1 ).to(device)
            obs_input = torch.FloatTensor(observation).unsqueeze(0).to(device)
            prev_belief = prev_belief.to(device)
            state, belief = agent.rssm.sample(prev_belief, rssm_input, obs_input)

            if ep_steps % action_repeat == 0: # action repeat
                # sample action from the (stochastic) policy
                action = agent.get_action(state, belief, epsilon)
            else:
                action = prev_action

            action_numpy = action.squeeze(0).detach().cpu().numpy()
            action_scalar = np.argwhere(action_numpy > 0.9).squeeze()
            next_observation, reward, done, _ = env.step(action_scalar)

            # used to differentiate between goal_terminal_state and timeout_terminal_state - not used in dreamer
            # done = float(done) if ep_steps < env._max_episode_steps else 0

            # add experience to replay buffer
            oar_tuple = [observation, action_numpy, reward, done] # done used to denote the break in dynamics between episodes
            agent.replay_buffer.add(oar_tuple)

            if done:
                # add terminal transition to the replay buffer
                terminal_observation = next_observation
                terminal_reward = reward * 0 # dummy value (not used for reward loss since we skip)
                terminal_action = action_numpy * 0 # dummy value / action reset value
                terminal_done = 0 # this is done flag for first observation of next episode
                oar_tuple = [terminal_observation, terminal_action, terminal_reward, terminal_done]
                agent.replay_buffer.add(oar_tuple)

            # for next step in episode
            observation = next_observation
            prev_state = state.detach().cpu()
            prev_action = action.detach().cpu()
            prev_belief = belief.detach().cpu()

            ep_return += (df ** ep_steps) * reward
            ep_steps += 1

        ## episode ended
        # train agent
        for _ in range(num_train_calls):
            l_dyn, l_rew, l_obs, l_df, l_kl, l_act, l_cri = agent.train(epsilon)
            loss_dynamics_list.append(l_dyn.item())
            loss_reward_list.append(l_rew.item())
            loss_obs_list.append(l_obs.item())
            loss_df_list.append(l_df.item())
            loss_kl_list.append(l_kl.item())
            loss_actor_list.append(l_act.item())
            loss_critic_list.append(l_cri.item())

        # store episode stats
        ep_return_list.append(ep_return)
        if ep % (num_episodes//10) == 0:
            print('ep:{} \t ep_return:{}'.format(ep, ep_return))


# hyperparam dict
hyperparam_dict = {}
hyperparam_dict['env'] = 'LunarLander-v2'
hyperparam_dict['algo'] = 'dreamerV2_5_1_12_7_dfexp4_exploreImagination'
hyperparam_dict['lr_actor'] = str(lr_actor)
hyperparam_dict['lr_critic'] = str(lr_critic)
hyperparam_dict['lr_model'] = str(lr_model)
# hyperparam_dict['_lambda'] = str(_lambda)
# hyperparam_dict['STD'] = str(.25)
hyperparam_dict['num_train_calls'] = str(num_train_calls)
hyperparam_dict['rho'] = str(rho)
hyperparam_dict['num_episodes'] = str(num_episodes)
hyperparam_dict['random_seed'] = str(random_seed)
hyperparam_dict['frac'] = str(frac)
hyperparam_dict['explore20'] = str(explore20)


# hyperparam string
hyperstr = ""
for k,v in hyperparam_dict.items():
    hyperstr += k + ':' + v + "__"


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

ep_returns_moving_mean = get_moving_mean_list(ep_return_list)
loss_dynamics_moving_mean = get_moving_mean_list(loss_dynamics_list)
loss_reward_moving_mean = get_moving_mean_list(loss_reward_list)
loss_obs_moving_mean = get_moving_mean_list(loss_obs_list)
loss_df_moving_mean = get_moving_mean_list(loss_df_list)
loss_kl_moving_mean = get_moving_mean_list(loss_kl_list)
loss_actor_moving_mean = get_moving_mean_list(loss_actor_list)
loss_critic_moving_mean = get_moving_mean_list(loss_critic_list)


# plot results
fig, ax = plt.subplots(2,2, figsize=(15,10))

ax[0,0].plot(ep_returns_moving_mean, color='green', label='ep_return')
ax[0,0].legend()
ax[0,0].set(xlabel='episode')

ax[0,1].plot(loss_dynamics_moving_mean, color='red', label='dynamics_loss')
ax[0,1].plot(loss_reward_moving_mean, color='lime', label='reward_loss')
ax[0,1].plot(loss_obs_moving_mean, color='blue', label='obs_loss')
ax[0,1].plot(loss_df_moving_mean, color='magenta', label='df_loss')
ax[0,1].plot(loss_kl_moving_mean, color='black', label='kl_loss')
ax[0,1].legend()
ax[0,1].set(xlabel='steps')
# ax[0,1].set_ylim([-500,1000])

ax[1,0].plot(loss_actor_moving_mean, color='blue', label='actor_loss')
ax[1,0].legend()
ax[1,0].set(xlabel='steps')


ax[1,1].plot(loss_critic_moving_mean, color='gray', label='critic_loss')
ax[1,1].legend()
ax[1,1].set(xlabel='steps')

plt.savefig('plots/' + hyperstr + '.png')
