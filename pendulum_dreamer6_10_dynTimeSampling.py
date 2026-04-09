### Program implementing Dreamer on Pendulum environment

## Key features of implementation / algorithm
# 1.1. behaviour learning - value function (implemented as fully connected net) : expected lambda return as value target with mse loss to learn the value function (using rollouts based on current policy over the currently learnt latent dynamics)
# 1.2. behaviour learning - policy (implemented as fully connected net parameterizing a gaussian followed by a tanh non-linearity) : objective is to maximize the expected lambda return. Learnt via gradient of the objective backproped through the leart dynamics (the dynamics functions are kept frozen)
# 2.1. dynamics learning - representation model p(s_t | s_t-1, a_t-1, o_t) - implemented as recurrent state space model with a CNN input to handle observation images (CNN not required for pendulum)
# 2.2. dynamics learning - transition model q(s_t | s_t-1, a_t-1) - implemented as a recurrent state space model
# 2.3. dynamics learning - reward model q(r_t | s_t) - implemented as a fully connected net parameterizing mean of a gaussian with unit variance
# 2.4. dynamics learning - observation model q(o_t | s_t) - implemented as a transposed CNN - (for pendulum, implemented as fully connected net)
# 2.5. dynamics learning - state model q(s_t | o_t) - implemented as a CNN - (for pendulum, implemented as fully connected net)
# 2.6. dynamics learning - all the dynamics models are learnt using two possible / separate loss objectives (given by equation 10 and 12 in the paper). Both loss objectives are based on the variational information bottleneck (VIB) principle.
# 3.1. interaction with environment - infer state from observation using the representation model, obtain action from policy, store experience in replay buffer

## todos / questions
# 1. [resolved] Activation function : ELU or ReLU? [fix: ELU]
# 2. in dynamics learning loss - how is the expectation in the KL term (with respect to representation model) calculated? [guess / possible fix: taking expectation over the representation model is equivalent to taking sample mean of quantities sampled from representation model. So if all the terms inside the KL are formulated from quantities sampled from the representation model, then the expectation can be approximated by sample mean of the terms inside the KL. The first term in the KL is the representation model logprob over states sampled from the representation model. The second term is the transition model logprob over states sampled from the representation model. So both the terms satisfy the criteria]
# 3. how is s_t-1 and a_t-1 obtained for the first sample / step?
# 4. should rssm state be reset (state = zero, grad = zero) at start of episode?
# 5. check equivalency between torch inbuilt function for kl div and our formulation
# 5.1 dynamics loss - batch mean at each time slice or at the end?
# 6. dynamics learning - reconstruction loss vs. NCE loss
# 7. model parameters passed to the model optimizer - should state model params included for reconstruction loss?
# 8. freezing model parameters during behaviour learning and freezing actor-critic parameters when dynamics learning
# 9. behaviour learning - parallel rollouts
# 10. behaviour learning - do we need mean (over the batch dimension) for the lambda return and the critic loss?
# 11. interaction - action repeat and exploration noise
# 12. model to learn discount_factor
# 13. value for belief_dim according to the paper

## important lessons / takeaways
# 1. For GRUCell, after loss.backward(), the computation graph is cleared but the hidden_state is sustained. Thus its necessary to correctly reset the hidden_state, else the hidden_state tensor will be interpreted as being at a different version when doing the next loss.backward().
# 2. The "done" flag here is used differently. In most cases, we use a "not_done" flag to differentiate between timeout_terminal_state and goal_terminal_state. However, in dreamer, we are learning behaviour using rollouts over a dynamics model, which doesn't have any terminal state - just imagination_horizon. So the learnt critic will take care of terminal states. But we do use "done" flag in dynamics learning - to denote the break in dynamics when going from the terminal state of previous episode and the start state of the next episode. We avoid these breaks in the contiguous samples from replay buffer.



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import gymnasium as gym
from tqdm import tqdm
from torchviz import make_dot
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
        # self.register_buffer('h_deterministic', torch.zeros(batch_size, belief_dim))

        # init stochastic net (separate layers for p and q models)
        self.obs_encoder_fc1 = nn.Linear(o_dim, h_dim)
        self.obs_encoder_fc2 = nn.Linear(h_dim, h_dim)
        self.p_fc0 = nn.Linear(belief_dim + h_dim, h_dim)
        self.p_fc1 = nn.Linear(h_dim, h_dim)
        self.p_fc2_mean = nn.Linear(h_dim, out_dim)
        self.p_fc2_std = nn.Linear(h_dim, out_dim)
        self.q_fc0 = nn.Linear(belief_dim, h_dim)
        self.q_fc1 = nn.Linear(h_dim, h_dim)
        self.q_fc2_mean = nn.Linear(h_dim, out_dim)
        self.q_fc2_std = nn.Linear(h_dim, out_dim)
        self.elu = nn.ELU()
        self.device = device

    # used to reset hidden state of gru - used at start of episode
    # def reset_h_deterministic(self, batch_size):
    #     self.h_deterministic = torch.zeros(batch_size, belief_dim).to(device)

    # forward pass through RSSM
    def forward(self, prev_belief, x, o=None):
        x = self.elu(self.fc_embed(x)) # x_t-1 = [s_t-1, a_t-1]
        belief = self.gru_cell(x, prev_belief) # h_t = gru(x_t-1, h_t-1)
        if o is None:
            h = self.elu(self.q_fc0(belief))
            h = self.elu(self.q_fc1(h))
            mean = self.q_fc2_mean(h) # q(s_t | h_t)
            logstd = self.q_fc2_std(h).clip(-4, 2)
            std = torch.exp(logstd)
        else:
            o = self.elu(self.obs_encoder_fc1(o))
            o = self.obs_encoder_fc2(o)
            h_o = torch.cat((belief, o), dim=1)
            h = self.elu(self.p_fc0(h_o))
            h = self.elu(self.p_fc1(h))
            mean = self.p_fc2_mean(h) # p(s_t | h_t, o_t)
            logstd = self.p_fc2_std(h).clip(-4, 2)
            std = torch.exp(logstd)
        return mean, std, belief

    # to draw sample from the learnt probabilistic model
    def sample(self, prev_belief, x, o=None):
        mean, std, belief = self.forward(prev_belief, x, o)
        eps = torch.randn_like(std)
        out = mean + eps * std
        return out, belief

    # formulates the guassina distribution from mean and std
    def get_dist(self, prev_belief, x, o=None):
        mean, std, belief = self.forward(prev_belief, x, o)
        dis = tdist.Normal(mean, std)
        return dis
    
    # formulates the guassina distribution from mean and std - in detached state for KL balancing
    def get_dist_detached(self, prev_belief, x, o=None):
        mean, std, belief = self.forward(prev_belief, x, o)
        mean, std = mean.detach(), std.detach()
        dis = tdist.Normal(mean, std)
        return dis

    # calculates log p(y|x)
    def log_prob(self, prev_belief, x, y, o=None):
        dis = self.get_dist(prev_belief, x, o)
        lp = dis.log_prob(y)
        return lp



# Stochastic net base
class StochasticNet_Base(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2_mean = nn.Linear(h_dim, out_dim)
        self.fc2_std = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=-1)
        h = self.relu(self.fc0(x))
        h = self.relu(self.fc1(h))
        mean = self.fc2_mean(h)
        logstd = self.fc2_std(h).clip(-4, 2)
        std = torch.exp(logstd)
        # std = 1.
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



# actor network - parameterizing the stochastic poicy
class Actor(nn.Module):
    def __init__(self, s_dim, belief_dim, a_dim, h_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + belief_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, h_dim)
        self.fc5_mean = nn.Linear(h_dim, a_dim)
        self.fc5_std = nn.Linear(h_dim, a_dim)
        self.relu = nn.ReLU()
        self.max_action = max_action
        self.tanh = nn.Tanh()
        self.a_dim = a_dim

    # note that this returns the mean and std of gaussian_policy
    # moreover, the actual policy used = tanh(gaussian_policy)
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        mean = self.fc5_mean(h)
        std = self.fc5_std(h)
        std = torch.clip(std, -4, 2)
        std = torch.exp(std)
        return mean, std

    # actual policy used pi' = tanh(pi) * max_action. Thus logprob(pi') = log(pi) - log( det(jacobian(tanh)) - D * log(max_action)
    def get_policy_logprob(self, state, belief):
        mean, std = self.forward(state, belief)
        z = torch.randn_like(std)
        gaussian_action = z * std + mean
        true_action = self.tanh(gaussian_action) * self.max_action

        gaussian_policy = tdist.Normal(mean, std)
        gaussian_policy_logprob = gaussian_policy.log_prob(gaussian_action)
        true_policy_logprob = gaussian_policy_logprob - torch.log( torch.abs(1 - torch.pow(self.tanh(gaussian_action), 2) + 1e-20) ) \
                              - self.a_dim * torch.log( torch.abs(torch.tensor(self.max_action)) )
        return true_policy_logprob, true_action



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



# Dreamer
class Dreamer(nn.Module):
    def __init__(self, o_dim, s_dim, a_dim, belief_dim, h_dim, max_action, sample_seq_len, ep_seq_len, imagination_horizon, df, buf_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, device):
        super().__init__()
        self.actor = Actor(s_dim, belief_dim, a_dim, h_dim, max_action).to(device)
        self.critic_V = Critic_V(s_dim, belief_dim, h_dim).to(device)
        self.replay_buffer = ReplayBuffer(buf_size, sample_seq_len, ep_seq_len, batch_size, o_dim, a_dim, device)
        self.rssm = RSSM(s_dim + a_dim, o_dim, belief_dim, h_dim, s_dim, batch_size, device).to(device)
        self.reward_model = StochasticNet_Base(s_dim + belief_dim, h_dim, 1).to(device)
        self.observation_model = StochasticNet_Base(s_dim + belief_dim, h_dim, o_dim).to(device)
        self.state_model = StochasticNet_Base(o_dim, h_dim, s_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic_V = torch.optim.Adam(params=self.critic_V.parameters(), lr=lr_critic)
        self.optimizer_model = torch.optim.Adam(params=list(self.rssm.parameters()) + list(self.reward_model.parameters()) + list(self.observation_model.parameters()) + list(self.state_model.parameters()), lr=lr_model)
        self.df = df
        self.max_action = max_action
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.belief_dim = belief_dim
        self.o_dim = o_dim
        self.device = device
        self.sample_seq_len = sample_seq_len
        self.ep_seq_len = ep_seq_len
        self.train_iters = 0
        self.tanh = nn.Tanh()
        self.imagination_horizon = imagination_horizon
        self.vib_beta = vib_beta
        self._lambda = _lambda
        self.batch_size = batch_size
        self.alpha = alpha

    def get_action(self, state, belief, explore_std):
        mean, std = self.actor(state, belief)
        z = torch.randn_like(std)
        gaussian_action = z * std + mean
        action = self.tanh(gaussian_action) * self.max_action

        # add exploration noise
        if explore_std > 0:
            exploration_noise = torch.randn_like(action) * explore_std 
            action += exploration_noise

        return action


    def freeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(False)

    def unfreeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(True)

    def calculate_lambda_return(self, rewards, state_values):
        """
        Input:
        # rewards obtained from reward model - r[0] : r[H-1]
        # state values obtained from critic model - v_t[0] : v_t[H-1]

        Output:
        # V_lambda[0] : V_lambda[H-2]
        """
        lambda_returns = []
        accumulator = state_values[-1].detach() # V_t[H-1]
        for t in range(len(rewards)-1, 0, -1): # t is just going from last element to first element, since all arrays are of length H-1
            accumulator = rewards[t] + df * ( (1 - self._lambda)*state_values[t].detach() + self._lambda*accumulator )
            # V_lambda[H-2] = reward[H-1] + df * ( (1-lambda) * V_t[H-1] + lambda * V_lambda[H-1] )
            lambda_returns = [accumulator] + lambda_returns
        lambda_returns = torch.stack(lambda_returns, dim=0)
        return lambda_returns


    def train(self, explore_std):
        observation, action, reward, done = self.replay_buffer.sample()

        #########################
        ## dynamics learning (using experience sampled from replay buffer)
        #########################

        # unfreeze dynamics model params
        self.unfreeze_model_params(self.rssm)
        self.unfreeze_model_params(self.reward_model)
        self.unfreeze_model_params(self.observation_model)
        self.unfreeze_model_params(self.state_model)

        # using reconstruction loss for now
        # todo - try NCE loss

        loss_dynamics = 0
        loss_reward = 0
        loss_obs = 0
        loss_kl = 0

        # list tensors to store states and beliefs obtained from representation model - used later for imagination rollout (during behaviour learning)
        prev_state_list = []
        prev_belief_list = []
        prev_action_list = []
        prev_reward_list = []
        im_state_list = []
        im_belief_list = []

        ## prepare starting states

        prev_state = torch.zeros(self.batch_size, self.s_dim).to(self.device)
        prev_belief = torch.zeros(self.batch_size, self.belief_dim).to(self.device)
        prev_action = torch.zeros(self.batch_size, self.a_dim).to(self.device)
        prev_reward = torch.zeros(self.batch_size, 1).to(self.device)
        
        for t in range(0, self.ep_seq_len):
            curr_obs = observation[t]
            curr_action = action[t]
            curr_reward = reward[t]

            rssm_input = torch.cat((prev_state, prev_action), dim=1)
            curr_state, curr_belief = self.rssm.sample(prev_belief, rssm_input, curr_obs)

            prev_state_list.append(prev_state.detach())
            prev_belief_list.append(prev_belief.detach())
            prev_action_list.append(prev_action)
            prev_reward_list.append(prev_reward)

            prev_state = curr_state.detach() 
            prev_belief = curr_belief.detach()
            prev_action = curr_action
            prev_reward = curr_reward


        # stack list to tensor for batched indexing
        prev_state_list = torch.stack(prev_state_list, dim=0)
        prev_belief_list = torch.stack(prev_belief_list, dim=0)
        prev_action_list = torch.stack(prev_action_list, dim=0)
        prev_reward_list = torch.stack(prev_reward_list, dim=0)

        # select rollout start time steps
        start_time_idx = np.random.randint(0, self.ep_seq_len - self.sample_seq_len, self.batch_size)

        ## start training steps
        for i in range(self.sample_seq_len):
            # prepare time indices
            time_idx = start_time_idx + i
            batch_idx = np.arange(self.batch_size)

            curr_obs = observation[time_idx, batch_idx]

            prev_state = prev_state_list[time_idx, batch_idx]
            prev_belief = prev_belief_list[time_idx, batch_idx]
            prev_action = prev_action_list[time_idx, batch_idx]
            prev_reward = prev_reward_list[time_idx, batch_idx]

            rssm_input = torch.cat((prev_state, prev_action), dim=1)
            curr_state, curr_belief = self.rssm.sample(prev_belief, rssm_input, curr_obs)

            im_state_list.append(curr_state.clone().detach()) # s_t - store state for imagination rollout
            im_belief_list.append(curr_belief.clone().detach()) # h_t - store belief for imagination rollout

            # log prob q(r_t | s_t)
            lp_reward = self.reward_model.log_prob(curr_state, curr_belief, prev_reward)
            lp_reward = lp_reward.sum(dim=1).mean()

            # log prob q(o_t | s_t)
            lp_obs = self.observation_model.log_prob(curr_state, curr_belief, curr_obs)
            lp_obs = lp_obs.sum(dim=1).mean()

            # KL divergence between gaussians
            dist_p = self.rssm.get_dist(prev_belief, rssm_input, curr_obs)
            dist_q = self.rssm.get_dist(prev_belief, rssm_input)

            dist_p_detached = self.rssm.get_dist_detached(prev_belief, rssm_input, curr_obs)
            dist_q_detached = self.rssm.get_dist_detached(prev_belief, rssm_input)
            kl_pq = self.alpha * tdist.kl.kl_divergence(dist_p_detached, dist_q) + \
                    (1 - self.alpha) * tdist.kl.kl_divergence(dist_p, dist_q_detached)

            # kl_pq = tdist.kl.kl_divergence(dist_p, dist_q)

            # free_nats = torch.full((1,), 3).to(self.device)
            # kl_div = torch.max(kl_pq, free_nats).mean()
            kl_div = kl_pq.mean()

            # vib objective
            vib_objective = lp_reward + lp_obs - self.vib_beta * kl_div
            # loss
            loss_dynamics -= vib_objective

            # loss accumulators for book keeping and plotting
            loss_reward -= lp_reward
            loss_obs -= lp_obs
            loss_kl += self.vib_beta * kl_div

            # # next time step
            # prev_state = curr_state.detach()
            # prev_belief = curr_belief.detach()
            # prev_action = curr_action
            # prev_reward = curr_reward


        # update dynamics model
        self.optimizer_model.zero_grad()
        loss_dynamics.backward()
        nn.utils.clip_grad_norm_(list(self.rssm.parameters()) + list(self.reward_model.parameters()) + list(self.observation_model.parameters()) + list(self.state_model.parameters()) , 100., norm_type=2)
        self.optimizer_model.step()


        ####################
        ## behaviour learning (using imagined rollouts over the learnt dynamics model)
        ####################

        # freeze dynamics model params
        self.freeze_model_params(self.rssm)
        self.freeze_model_params(self.reward_model)
        self.freeze_model_params(self.observation_model)
        self.freeze_model_params(self.state_model)

        im_state_list = torch.stack(im_state_list, dim=0)
        im_belief_list = torch.stack(im_belief_list, dim=0)

        # flatten time and batch dimension of belief_list into one - for parallel rollouts
        # and init belief state with these values
        im_curr_belief = torch.flatten(im_belief_list, start_dim=0, end_dim=1)

        # flatten time and batch dimension into one - for parallel rollouts
        im_curr_state = torch.flatten(im_state_list, start_dim=0, end_dim=1)

        # containers to keep required values from imagination rollouts
        im_rewards = [] # rewards obtained from reward model [ r(s_0) : r(s_H-1) ]
        im_state_values_actor = [] # state values obtained from critic [v(s_0) : v(s_H-1)] - used for actor training (no detach in state transitions)
        im_state_values_critic = [] # state values obtained from critic [v(s_0) : v(s_H-1)] - used for critic training (detach in state transitions)

        # start (parallel) rollout(s)
        for tau in range(self.imagination_horizon):
            im_curr_reward = self.reward_model.sample(im_curr_state, im_curr_belief)
            im_curr_action = self.get_action(im_curr_state, im_curr_belief, explore_std=0)
            # rollout step
            rssm_input = torch.cat((im_curr_state, im_curr_action), dim=1)
            im_next_state, im_next_belief = self.rssm.sample(im_curr_belief, rssm_input)
            # store values to calculate lambda return
            im_rewards.append(im_curr_reward) 
            im_state_values_actor.append( self.critic_V(im_curr_state, im_curr_belief).detach() )
            im_state_values_critic.append( self.critic_V(im_curr_state.clone().detach(), im_curr_belief.clone().detach()) )
            # next step 
            im_curr_state = im_next_state
            im_curr_belief = im_next_belief

        # end of rollout 
        # calculate lambda returns
        lambda_returns = self.calculate_lambda_return(im_rewards, im_state_values_actor) # [0 : H-2]
        im_state_values_critic = torch.stack(im_state_values_critic[:-1], dim=0) # [0 : H-2]

        # calculate loss_actor and loss_critic
        loss_actor = -lambda_returns.sum(dim=0).mean()
        critic_target = lambda_returns.clone().detach()
        loss_critic = F.mse_loss(im_state_values_critic, critic_target, reduction='none')
        loss_critic = loss_critic.sum(dim=0).mean()

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

        return loss_dynamics, loss_reward, loss_obs, loss_kl, loss_actor, loss_critic




# main
if __name__ == '__main__':

    # hyperparams
    h_dim = 256
    s_dim = 32 # 30
    belief_dim = 128 # 200
    lr_actor = 8e-5
    lr_critic = 8e-5
    lr_model = 6e-4
    sample_seq_len = 32 # 64 # 50 # length of contiguous sequence sampled from replay buffer (when training)
    imagination_horizon = 16 # length of imagined rollouts using the learnt dynamics model (when behaviour learning)
    vib_beta = 1. # beta - tradeoff hyperparam in vib objective
    _lambda = .95 # lambda - used to calculate lambda return
    alpha = .8 # used for kl balancing
    replay_buffer_size = 10**3
    df = 0.99
    batch_size = 256 
    num_episodes = 200 # 1000
    train_episode = 1 # 25
    random_seed = 1010
    init_random_episodes = 5 # 64
    num_train_calls = 10 # 30 # 50
    action_repeat = 2 # TODO: should we add action repeat during imagination?
    max_time_steps = 200
    min_reward = -16.2736044
    max_reward = 0
    record_episode = num_episodes // 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('Pendulum-v1', render_mode="rgb_array")
    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    explore_std_init = 2.0 # max_action # 10.
    explore_std_end = 0.03

    # hyperparam dict
    hyperparam_dict = {}
    hyperparam_dict['env'] = 'Pendulum-v1'
    hyperparam_dict['algo'] = 'dreamer6_10_dynTimeSamp'
    hyperparam_dict['lr_actor'] = str(lr_actor)
    hyperparam_dict['lr_critic'] = str(lr_critic)
    hyperparam_dict['lr_model'] = str(lr_model)
    # hyperparam_dict['lambda'] = str(_lambda)
    # hyperparam_dict['vib_beta'] = str(vib_beta)
    hyperparam_dict['train_calls'] = str(num_train_calls)
    hyperparam_dict['action_repeat'] = str(action_repeat)
    hyperparam_dict['trainEP'] = str(train_episode)
    hyperparam_dict['EP'] = str(num_episodes)
    hyperparam_dict['initEP'] = str(init_random_episodes)
    # hyperparam_dict['seed'] = str(random_seed)
    hyperparam_dict['xploreSt'] = str(explore_std_init)
    hyperparam_dict['xploreEnd'] = str(explore_std_end)
    # hyperparam_dict['belief_dim'] = 'belief_dim'
    # hyperparam_dict['s_dim'] = 's_dim'
    # hyperparam_dict['h_dim'] = 'h_dim'


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
    agent = Dreamer(o_dim, s_dim, a_dim, belief_dim, h_dim, max_action, sample_seq_len, max_time_steps+1, imagination_horizon, df, replay_buffer_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, device)

    # results and stats containers
    ep_return_list = []
    loss_dynamics_list = []
    loss_reward_list = []
    loss_obs_list = []
    loss_kl_list = []
    loss_actor_list = []
    loss_critic_list = []

    # seed episodes
    for ep in range(init_random_episodes):
        done = 0
        ep_steps = 0
        ep_trace = []

        obs, info = env.reset()

        while done == 0:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                done = 1
            if truncated:
                done = 2

            # normalize reward in [0, 1]
            reward_normalized = (reward - min_reward) / (max_reward - min_reward)

            # add experience to ep trace
            ep_trace.append([obs, action, reward_normalized, done]) 

            if done:
                # add terminal transitions to ep_trace
                while len(ep_trace) < (max_time_steps + 1):
                    terminal_observation = next_obs # terminal state is absorbing
                    terminal_reward = reward_normalized # terminal state should repeat the reward
                    terminal_action = env.action_space.sample() # all further actions can be random
                    terminal_done = 0 # this is not really used
                    ep_trace.append([terminal_observation, terminal_action, terminal_reward, terminal_done])
                    
                # add ep_trace to replay buffer
                agent.replay_buffer.add(ep_trace)

            obs = next_obs
            ep_steps += 1

    
    # explore std deviation schedule
    explore_std_schedule = np.ones(num_episodes) * explore_std_end
    # note that in linspace, the boundaries are inclusive
    explore_std_schedule[:int(num_episodes * 0.9)] = np.linspace(explore_std_init, explore_std_end, int(num_episodes * 0.9))

    # interactive episodes
    for ep in tqdm(range(num_episodes)):
        done = 0
        ep_return = 0
        ep_steps = 0
        ep_trace = []
        frames = []
        # for exploration noise
        explore_std = explore_std_schedule[ep]

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
                    action = agent.get_action(state, belief, explore_std)
                else:
                    action = prev_action

                action_numpy = action.squeeze(0).detach().cpu().numpy()

                next_observation, reward, terminated, truncated, _ = env.step(action_numpy)
                if terminated:
                    done = 1
                if truncated:
                    done = 2

                # normalize reward in [0, 1]
                reward_normalized = (reward - min_reward) / (max_reward - min_reward)

                # add experience to ep trace
                ep_trace.append([obs, action_numpy, reward_normalized, done]) 

                if done:
                    # add terminal transitions to ep_trace
                    while len(ep_trace) < (max_time_steps + 1):
                        terminal_observation = next_observation # terminal state is absorbing
                        terminal_reward = reward_normalized # terminal state should repeat the reward
                        terminal_action = env.action_space.sample() # all further actions can be random
                        terminal_done = 0 # this is not really used
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
                l_dyn, l_rew, l_obs, l_kl, l_act, l_cri = agent.train(explore_std)
                loss_dynamics_list.append(l_dyn.item())
                loss_reward_list.append(l_rew.item())
                loss_obs_list.append(l_obs.item())
                loss_kl_list.append(l_kl.item())
                loss_actor_list.append(l_act.item())
                loss_critic_list.append(l_cri.item())

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

ep_returns_moving_mean = get_moving_mean_list(ep_return_list)
loss_dynamics_moving_mean = get_moving_mean_list(loss_dynamics_list)
loss_reward_moving_mean = get_moving_mean_list(loss_reward_list)
loss_obs_moving_mean = get_moving_mean_list(loss_obs_list)
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
ax[0,1].plot(loss_kl_moving_mean, color='black', label='kl_loss')
ax[0,1].legend()
ax[0,1].set(xlabel='steps')
ax[0,1].set_ylim([-500, 500])

ax[1,0].plot(loss_actor_moving_mean, color='blue', label='actor_loss')
ax[1,0].legend()
ax[1,0].set(xlabel='steps')


ax[1,1].plot(loss_critic_moving_mean, color='gray', label='critic_loss')
ax[1,1].legend()
ax[1,1].set(xlabel='steps')

plt.savefig('plots/' + hyperstr + '.png')