### Program implementing Dreamer on BipedalWalker-v3 environment

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
        self.relu = nn.ReLU()
        self.device = device
        # self.min_std = 0.1

    # used to reset hidden state of gru - used at start of episode
    # def reset_h_deterministic(self, batch_size):
    #     self.h_deterministic = torch.zeros(batch_size, belief_dim).to(device)

    # forward pass through RSSM
    def forward(self, prev_belief, x, o=None):
        x = self.elu(self.fc_embed(x))
        belief = self.gru_cell(x, prev_belief)
        if o is None:
            h = self.elu(self.q_fc0(belief))
            h = self.elu(self.q_fc1(h))
            mean = self.q_fc2_mean(h)
            logstd = self.q_fc2_std(h).clip(-2, 2)
            std = torch.exp(logstd)
        else:
            o = self.elu(self.obs_encoder_fc1(o))
            o = self.obs_encoder_fc2(o)
            h_o = torch.cat((belief, o), dim=1)
            h = self.elu(self.p_fc0(h_o))
            h = self.elu(self.p_fc1(h))
            mean = self.p_fc2_mean(h)
            logstd = self.p_fc2_std(h).clip(-2, 2)
            std = torch.exp(logstd)
        return mean, std, belief

    # to draw sample from the learnt probabilistic model
    def sample(self, prev_belief, x, o=None):
        mean, std, belief = self.forward(prev_belief, x, o)
        eps = tdist.Normal(0, 1).sample()
        out = mean + eps * std
        return out, belief

    # formulates the guassina distribution from mean and std
    def get_dist(self, prev_belief, x, o=None):
        mean, std, belief = self.forward(prev_belief, x, o)
        dis = tdist.Normal(mean, std)
        return dis

    def get_dist_detached(self, prev_belief, x, o=None):
        mean, std, belief = self.forward(prev_belief, x, o)
        mean = mean.detach()
        std = std.detach()
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
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3_mean = nn.Linear(h_dim, out_dim)
        # self.fc3_std = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        # logstd = self.fc3_std(h).clip(-2, 2)
        # std = torch.exp(logstd)
        std = 1.
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
        x = torch.cat((state, belief), dim=1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        mean = self.fc5_mean(h)
        logstd = self.fc5_std(h).clip(-2, 2)
        std = torch.exp(logstd)
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
    def __init__(self, buf_size, seq_len, batch_size, o_dim, a_dim, frac, device):
        self.buf_size = buf_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buf_observation = np.zeros((buf_size, o_dim))
        self.buf_action = np.zeros((buf_size, a_dim))
        self.buf_reward = np.zeros((buf_size, 1))
        self.buf_done = np.zeros((buf_size, 1)) # timeout termination
        self.buf_pt = np.zeros((buf_size, 1)) # premature termination
        self.n_items = 0
        self.device = device
        self.frac = frac

    def add(self, oar_tuple, pt=0):
        observation, action, reward, done = oar_tuple
        index = self.n_items % self.buf_size
        self.buf_observation[index] = observation
        self.buf_action[index] = action
        self.buf_reward[index] = reward
        self.buf_done[index] = done
        self.buf_pt[index] = pt
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

                # if pt==1 for an index in idx_chunk, repeat that index for rest of the chunk
                fix_j = -1
                for j in range(len(idx_chunk)):
                    if self.buf_pt[idx_chunk[j]] == 1:
                        fix_j = j
                        break
                if fix_j > -1:
                    idx_chunk[fix_j:] = idx_chunk[fix_j]

                # ensure atleast frac% samples have pt flag
                if len(idx_list) < (self.batch_size * self.frac):
                    if (1 in self.buf_pt[idx_chunk]):
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



# Dreamer
class Dreamer(nn.Module):
    def __init__(self, o_dim, s_dim, a_dim, belief_dim, h_dim, max_action, seq_len, imagination_horizon, df, buf_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, frac, device):
        super().__init__()
        self.actor = Actor(s_dim, belief_dim, a_dim, h_dim, max_action).to(device)
        self.critic_V = Critic_V(s_dim, belief_dim, h_dim).to(device)
        self.replay_buffer = ReplayBuffer(buf_size, seq_len, batch_size, o_dim, a_dim, frac, device)
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
        self.train_iters = 0
        self.tanh = nn.Tanh()
        self.seq_len = seq_len
        self.imagination_horizon = imagination_horizon
        self.vib_beta = vib_beta
        self._lambda = _lambda
        self.batch_size = batch_size
        self.alpha = alpha

    def get_action(self, state, belief):
        mean, std = self.actor(state, belief)
        z = tdist.Normal(0, 1).sample()
        gaussian_action = z * std + mean
        true_action = self.tanh(gaussian_action) * self.max_action
        return true_action


    def freeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(False)

    def unfreeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad_(True)


    def calculate_lambda_return(self, rewards, state_values_target):
        """
        # rewards obtained from reward model [ r(s_0) : r(s_H-1) ]
        # state values obtained from target_critic [ v(s_1) : v(s_H) ]
        """
        lambda_returns = []
        accumulator = state_values_target[-1]
        for t in range(len(rewards)-1, -1, -1):
            accumulator = rewards[t] + self.df * ( (1 - self._lambda)*state_values_target[t] + self._lambda*accumulator )
            lambda_returns = [accumulator] + lambda_returns
        lambda_returns = torch.stack(lambda_returns, dim=0)
        return lambda_returns


    def train(self):
        observation, action, reward, done = self.replay_buffer.sample()
        observation = observation[1:]
        reward = reward[:-1]
        action = action[:-1]
        done = done[:-1]

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
        state_list = []
        belief_list = []

        ## fixed prev_state and prev_action (to intialize the first state)
        prev_state = torch.zeros(self.batch_size, self.s_dim).to(self.device)
        prev_action = torch.zeros(self.batch_size, self.a_dim).to(self.device) # action should also be reset to zero - as the decided dummy value
        prev_belief = torch.zeros(self.batch_size, self.belief_dim).to(self.device)
        prev_prev_done = torch.ones(self.batch_size, 1).to(self.device)

        # start training steps
        for t in range(self.seq_len-1):
            curr_obs = observation[t]
            prev_action = action[t]
            prev_reward = reward[t]
            prev_done = done[t]

            # reset if curr_obs is from new episode (implied by prev_prev_done =  1)
            prev_state = prev_state * (1. - prev_done)
            prev_action = prev_action * (1. - prev_done)
            prev_belief = prev_belief * (1. - prev_done)

            rssm_input = torch.cat((prev_state, prev_action), dim=1)
            curr_state, curr_belief = self.rssm.sample(prev_belief, rssm_input, curr_obs)

            state_list.append(curr_state.clone().detach()) # s_t - store state for imagination rollout
            belief_list.append(curr_belief.clone().detach()) # h_t - store belief for imagination rollout

            # log prob q(r_t | s_t)
            lp_reward = self.reward_model.log_prob(curr_state, curr_belief, prev_reward)
            lp_reward = lp_reward * (1. - prev_done) # skip reward loss for first state (since no reward for spawning into the first state)
            lp_reward = lp_reward.sum(dim=1).mean()

            # log prob q(o_t | s_t)
            lp_obs = self.observation_model.log_prob(curr_state, curr_belief, curr_obs).sum(dim=1).mean()

            # KL divergence - using KL balancing
            dist_p = self.rssm.get_dist(prev_belief, rssm_input, curr_obs)
            dist_q = self.rssm.get_dist(prev_belief, rssm_input)
            dist_p_detached = self.rssm.get_dist_detached(prev_belief, rssm_input, curr_obs)
            dist_q_detached = self.rssm.get_dist_detached(prev_belief, rssm_input)
            kl_pq = self.alpha * tdist.kl.kl_divergence(dist_p_detached, dist_q) + \
                    (1 - self.alpha) * tdist.kl.kl_divergence(dist_p, dist_q_detached)

            # free_nats = torch.full((1,), 3).to(self.device)
            # kl_div = torch.max(kl_pq, free_nats)
            kl_div = kl_pq.mean()

            # vib objective
            vib_objective = lp_reward + lp_obs - self.vib_beta * kl_div

            # loss
            loss_dynamics -= vib_objective

            # loss accumulators for book keeping and plotting
            loss_reward -= lp_reward
            loss_obs -= lp_obs
            loss_kl += self.vib_beta * kl_div

            # next time step
            prev_state = curr_state
            prev_belief = curr_belief
            prev_prev_done = prev_done


        # update dynamics model
        self.optimizer_model.zero_grad()
        loss_dynamics.backward()
        nn.utils.clip_grad_norm_(list(self.rssm.parameters()) + list(self.reward_model.parameters()) + list(self.observation_model.parameters()) + list(self.state_model.parameters()) , 100., norm_type=2)
        self.optimizer_model.step()

        state_list = torch.stack(state_list, dim=0)
        belief_list = torch.stack(belief_list, dim=0)


        ####################
        ## behaviour learning (using imagined rollouts over the learnt dynamics model)
        ####################

        # freeze dynamics model params
        self.freeze_model_params(self.rssm)
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

        # start (parallel) rollout(s)
        for tau in range(self.imagination_horizon):
            im_curr_reward = self.reward_model.sample(im_curr_state, im_curr_belief)
            im_curr_action = self.get_action(im_curr_state.clone().detach(), im_curr_belief.clone().detach())
            # rollout step
            rssm_input = torch.cat((im_curr_state, im_curr_action), dim=1)
            im_next_state, im_next_belief = self.rssm.sample(im_curr_belief, rssm_input)
            # store required values
            state_values.append(self.critic_V( im_curr_state.clone().detach(), im_curr_belief.clone().detach() ))
            state_values_target.append( self.critic_V(im_next_state, im_next_belief).detach() )
            rewards.append(im_curr_reward)
            # for next step in rollout
            im_curr_state = im_next_state
            im_curr_belief = im_next_belief

        # end of rollout
        # calculate lambda return
        state_values = torch.stack(state_values[:-1], dim=0) # [0 : H-2]
        state_values_target = torch.stack(state_values_target[:-1], dim=0) # [1 : H-1]
        rewards = torch.stack(rewards[1:], dim=0) # [1 : H-1]
        lambda_returns = self.calculate_lambda_return(rewards, state_values_target)  # [0 : H-2]

        # calculate loss_actor and loss_critic
        loss_actor = -lambda_returns.mean()
        critic_targets = lambda_returns.clone().detach()
        loss_critic = F.mse_loss(state_values, critic_targets)

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
    s_dim = 30
    belief_dim = 200
    lr_actor = 8e-5 # 8e-5
    lr_critic = 8e-5 # 8e-5
    lr_model = 1e-4 # 6e-4
    sample_seq_len = 50 # length of contiguous sequence sampled from replay buffer (when training)
    imagination_horizon = 15 # length of imagined rollouts using the learnt dynamics model (when behaviour learning)
    vib_beta = 1. # beta - tradeoff hyperparam in vib objective
    _lambda = .95 # lambda - used to calculate lambda return
    alpha = .8 # used for kl balancing
    replay_buffer_size = 10**6
    df = 0.99
    batch_size = 50
    num_episodes = 50
    random_seed = 0
    render_final_episodes = True
    init_random_episodes = 5
    num_train_calls = 100 # 100
    action_repeat = 2
    frac = 0.
    explore_std_init = 1.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('BipedalWalker-v3')
    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])


    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    # init Dreamer agent
    agent = Dreamer(o_dim, s_dim, a_dim, belief_dim, h_dim, max_action, sample_seq_len, imagination_horizon, df, replay_buffer_size, batch_size, lr_actor, lr_critic, lr_model, vib_beta, _lambda, alpha, frac, device)

    # results and stats containers
    ep_return_list = []
    loss_dynamics_list = []
    loss_reward_list = []
    loss_obs_list = []
    loss_kl_list = []
    loss_actor_list = []
    loss_critic_list = []

    # seed episodes
    seed_pt = 0
    for ep in range(init_random_episodes): # continue seed episodes until we get a premature termination
        obs = env.reset()
        done = False
        ep_steps = 0

        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)


            # add experience to replay buffer
            if done: # terminal transition
                # differentiate between actual_terminal_state and timeout_terminal_state
                if (ep_steps < env._max_episode_steps-1): # actual terminal state and not timeout termination
                    buf_done = 0 # so that rssm doesn't reset and understands that actual terminal state is to be followed by first state during rollout
                    seed_pt = 0
                    # add terminal transition to replay buffer
                    oar_tuple = [obs, action, reward, buf_done]
                    agent.replay_buffer.add(oar_tuple, seed_pt)
                    # add obs_next as the explicit terminal state
                    terminal_observation = next_obs
                    terminal_reward = reward
                    terminal_action = action
                    buf_done = 0 # since rssm not to be reset
                    seed_pt = 1 # premature termination
                    terminal_oar_tuple = [terminal_observation, terminal_action, terminal_reward, buf_done]
                    agent.replay_buffer.add(terminal_oar_tuple, seed_pt)
                else: # timeout termination
                    buf_done = 1
                    seed_pt = 0
                    # add terminal transition to replay buffer
                    oar_tuple = [obs, action, reward, buf_done]
                    agent.replay_buffer.add(oar_tuple, seed_pt)
            else: # intermediate transition
                # add intermediate transition  to replay buffer
                oar_tuple = [obs, action, reward, done] # done used to denote the break in dynamics between episodes
                agent.replay_buffer.add(oar_tuple)


            # if done:
            #     # add terminal transition to the replay buffer
            #     terminal_observation = next_obs
            #     terminal_reward = reward * 0 # dummy value (not used for reward loss since we skip)
            #     terminal_action = action * 0 # dummy value / action reset value
            #     terminal_done = 0 # this is done flag for first observation of next episode
            #     oar_tuple = [terminal_observation, terminal_action, terminal_reward, terminal_done]
            #     agent.replay_buffer.add(oar_tuple)

            obs = next_obs
            ep_steps += 1

    # explore std deviation schedule
    explore_std_schedule = np.ones(num_episodes) * 0.03
    explore_std_schedule[:int(num_episodes * 0.8)] = np.linspace(explore_std_init, 0.03, int(num_episodes * 0.8))

    # interactive episodes
    success_count = 0
    for ep in tqdm(range(num_episodes)):
        done = False
        ep_return = 0
        ep_steps = 0

        # first observation of the episode
        observation = env.reset()

        # init state and action
        prev_state = torch.zeros(1, s_dim)
        prev_action = torch.zeros(1, a_dim)
        prev_belief = torch.zeros(1, belief_dim)

        while not done:
            if render_final_episodes and ( (ep % 1 == 0) or (ep > (num_episodes - 10)) ):
            # if render_final_episodes and (ep > (num_episodes - 5)):
                env.render()

            # infer state from observation using representation model
            rssm_input = torch.cat( (prev_state, prev_action), dim=1 ).to(device)
            obs_input = torch.FloatTensor(observation).unsqueeze(0).to(device)
            prev_belief = prev_belief.to(device)
            state, belief = agent.rssm.sample(prev_belief, rssm_input, obs_input)

            if ep_steps % action_repeat == 0: # action repeat
                # sample action from the (stochastic) policy
                action = agent.get_action(state, belief)
                # add exploration noise
                explore_std = explore_std_schedule[ep] + 1e-5
                exploration_noise = tdist.Normal(0, explore_std).sample()
                action += exploration_noise
            else:
                action = prev_action

            action_numpy = action.squeeze(0).detach().cpu().numpy()

            next_observation, reward, done, _ = env.step(action_numpy)

            # add experience to replay buffer
            if done: # terminal transition
                # differentiate between actual_terminal_state and timeout_terminal_state
                if (ep_steps < env._max_episode_steps-1): # actual terminal state and not timeout termination
                    buf_done = 0 # so that rssm doesn't reset and understands that actual terminal state is to be followed by first state during rollout
                    pt = 0 # premature termination - turned on in the explicit terminal state
                    # add terminal transition to replay buffer
                    oar_tuple = [observation, action_numpy, reward, buf_done]
                    agent.replay_buffer.add(oar_tuple, pt)
                    # add obs_next as the explicit terminal state
                    terminal_observation = next_observation
                    terminal_reward = reward
                    terminal_action = action_numpy
                    buf_done = 0 # since rssm not to be reset
                    terminal_pt = 1 # premature termination
                    terminal_oar_tuple = [terminal_observation, terminal_action, terminal_reward, buf_done]
                    agent.replay_buffer.add(terminal_oar_tuple, terminal_pt)
                else: # timeout termination
                    success_count += 1
                    buf_done = 1
                    pt = 0
                    # add terminal transition to replay buffer
                    oar_tuple = [observation, action_numpy, reward, buf_done]
                    agent.replay_buffer.add(oar_tuple, pt)
            else: # intermediate transition
                # add intermediate transition  to replay buffer
                oar_tuple = [observation, action_numpy, reward, done] # done used to denote the break in dynamics between episodes
                agent.replay_buffer.add(oar_tuple)


            # if done:
            #     # add terminal transition to the replay buffer
            #     terminal_observation = next_observation
            #     terminal_reward = reward * 0 # dummy value (not used for reward loss since we skip)
            #     terminal_action = action_numpy * 0 # dummy value / action reset value
            #     terminal_done = 0 # this is done flag for first observation of next episode
            #     oar_tuple = [terminal_observation, terminal_action, terminal_reward, terminal_done]
            #     agent.replay_buffer.add(oar_tuple)


            # for next step in episode
            observation = next_observation
            prev_state = state.detach().cpu()
            prev_action = action.detach().cpu()
            prev_belief = belief.detach().cpu()

            ep_return += reward
            ep_steps += 1

        ## episode ended
        # train agent
        for _ in range(num_train_calls):
            l_dyn, l_rew, l_obs, l_kl, l_act, l_cri = agent.train()
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


# hyperparam dict
hyperparam_dict = {}
hyperparam_dict['env'] = 'BipedalWalker-v3'
hyperparam_dict['algo'] = 'dreamer5_dfexp25_fixedSigma'
hyperparam_dict['lr_actor'] = str(lr_actor)
hyperparam_dict['lr_critic'] = str(lr_critic)
hyperparam_dict['lr_model'] = str(lr_model)
# hyperparam_dict['_lambda'] = str(_lambda)
# hyperparam_dict['vib_beta'] = str(vib_beta)
hyperparam_dict['num_train_calls'] = str(num_train_calls)
# hyperparam_dict['action_repeat'] = str(action_repeat)
hyperparam_dict['num_episodes'] = str(num_episodes)
# hyperparam_dict['random_seed'] = str(random_seed)
# hyperparam_dict['belief_dim'] = 'belief_dim'
hyperparam_dict['frac'] = str(frac)
hyperparam_dict['explore_std_init'] = str(explore_std_init)


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

ep_returns_moving_mean = ep_return_list
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
# ax[0,1].set_ylim([-500, 500])

ax[1,0].plot(loss_actor_moving_mean, color='blue', label='actor_loss')
ax[1,0].legend()
ax[1,0].set(xlabel='steps')


ax[1,1].plot(loss_critic_moving_mean, color='gray', label='critic_loss')
ax[1,1].legend()
ax[1,1].set(xlabel='steps')

plt.title('success_count: ' + str(success_count))
plt.savefig('plots/' + hyperstr + '.png')
