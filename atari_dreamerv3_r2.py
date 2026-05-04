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

-----------------------------------------------------------------

Dreamer-v3 changes:

*1. symlog transform applied to reward targets, obsDecoder targets and critic targets. In practice, they are implemented as symexp_twohot distributions, such that logits are in symlog space and outputs are in symexp space. The paper also mentions applying symlog to encoder inputs p(s_t|h_t, o_t) but I'm not sure if they mean to apply symlog on both h_t and o_t or just one of them (skipping this for now). 
2. new weighing factors (betas) for dynamics losses
3. free nats for KL values via clipping
4. The categorical distributions of both prior q and posterior p of the rssm are smoothed by making them a mixture of 1% uniform and 99% output of neural nets. This can be implemented by smoothing the prior and posterior logits.
*5. The reward and the critic distributions are two-hot categorical and their output values are the expected values of the respective distributions (not samples from the distribution). The suggested number of discrete buckets = 255 but how do we choose the range of values represented by the discrete buckets (it will depend on the environment though symlog can somewhat limit the range) ? Also check how to implement two-hot encoding as it seems weird to implement than one-hot encoding.
6. Init the weights of output layers of reward model and critic model to zeros.
7. Rescaling the lambda_return value for actor loss: Actor uses the lambda_return value for both dynamics loss and reinforce loss. Authors suggest to rescale this lambda_return value for both the actor losses by dividing it with a scale = ema(95th percentile - 5th percentile)
8. Network architecture change: act_fn=SiLU() and added layernorms
9. Is unimix used only for RSSM or every one-hot dist ?
10. Calculation of advantage for actor loss should use critic or target_critic for the baseline value ? Some implementations (e.g. R2Dreamer) use critic (not target) even for calculating lambda_return (check what dreamer-v3 paper suggests). The dreamer-v3 paper suggest always using the critic for calculating both the lambda return and the advantage baseline. The target_critic is used only to regularize the critic via the critic loss: - ( log_prob(lambda_return) + log_prob(target_critic_value) )

Some more things that can be tried:
1. In RSSM sampling, use gumbel-softmax-straight-through instead of just straight-through
2. Try bigger networks as they improve efficiency in dreamer-v3
3. Store the states and beliefs obtained during interaction in the replay buffer 
4. Take note of when to do dist.rsample() versus dist.mode 
    4.1. During imagination rollout, we use reward_model.mode(), critic_model.mode(), target_critic_model.mode() and df_model.mean() 
5. Debug tip: visualize dreams using the observation decoder
6. R2Dreamer implementation calculates the critic_loss for the representation_rollout (used to train the dynamics model) along with the standard imagination rollouts - not sure if dreamer-v3 paper suggests doing this. R2Dreamer implementation also sets the lambda_return for the final imagination step = 0 for critic_loss.
            
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import matplotlib.pyplot as plt
from copy import deepcopy
import ale_py
import gymnasium as gym
from tqdm import tqdm
import imageio
import cv2

# torch.autograd.set_detect_anomaly(True)

# --------------------- distribution utils: oneHot and twoHot ------------------ #

def to_f32(x):
    return x.to(dtype=torch.float32)


def to_i32(x):
    return x.to(dtype=torch.int32)


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


# overwrite oneHotDist allowing for unimix
class OneHotDist(tdist.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        # (..., K)
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        # (..., K)
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=(), temperature=1.0):
        # (..., K)
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError


class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        # (..., N_bins), (N_bins,)
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        # (..., N_bins), (N_bins,) -> (..., 1)
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + ((p1 * b1).flip(dims=(-1,)) + (p3 * b3)).sum(
                dim=-1, keepdim=True
            )
            return self.unsquash(wavg)
        p1 = self.probs[..., : n // 2]
        p2 = self.probs[..., n // 2 :]
        b1 = self.bins[..., : n // 2]
        b2 = self.bins[..., n // 2 :]
        wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target):
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)


def onehot(mean, unimix_ratio, **kwargs):
    return OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)


def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    return TwoHot(to_f32(logits), bins)


# --------------------- define networks ------------------ #


# observation encoder network
class ObsEncoder(nn.Module):

    def __init__(self, h_dim, dropout, input_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # adaptive pooling fixes everything
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        flattened_size = 64 * 16 * 16

        self.fc1 = nn.Linear(flattened_size, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)

        self.norm = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()

    def forward(self, x):
        batch_shape = x.shape[:-3]
        x = x.flatten(start_dim=0, end_dim=-4)

        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        # x = self.silu(self.conv3(x))
        # x = self.silu(self.conv4(x))

        x = self.pool(x)

        x = x.reshape(*batch_shape, -1)

        x = self.silu(self.fc1(x))
        x = self.dropout(x)
        # x = self.norm(x)
        x = self.fc2(x)

        return x


class ObsEncoder_Strided(nn.Module):
    def __init__(self, h_dim, input_channels, dropout):
        super().__init__()
        # strided convs reduce spatial dims while preserving gradient flow
        self.conv1 = nn.Conv2d(input_channels, 8, 4, stride=2, padding=1)  # 32x32
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)              # 16x16
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=1)             # 8x8
        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1)            # 4x4
        
        self.fc = nn.Linear(64 * 4 * 4, h_dim)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_shape = x.shape[:-3]
        x = x.flatten(start_dim=0, end_dim=-4)
        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        x = self.silu(self.conv3(x))
        x = self.silu(self.conv4(x))
        x = x.reshape(*batch_shape, -1)
        x = self.fc(x)
        return x
    


# Observation model (Gaussian with deconv layers)
class Observation_Model_Deconv_Gaussian(nn.Module):
    def __init__(self, state_dim, belief_dim, h_dim, n_classes):
        super().__init__()

        self.n_classes = n_classes

        self.fc = nn.Linear(state_dim + belief_dim, 64 * 4 * 4)
        self.silu = nn.SiLU()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 8x8
            nn.SiLU(),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 16x16
            nn.SiLU(),

            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),    # 32x32
            nn.SiLU(),

            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),    # 64x64
            nn.SiLU(),

            # final conv (no stride change)
            nn.Conv2d(4, n_classes * 2, kernel_size=3, padding=1)
        )

    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=-1)
        h = self.fc(x)

        batch_shape = h.shape[:-1]
        h = h.flatten(start_dim=0, end_dim=-2)
        h = h.view(h.shape[0], 64, 4, 4) # [B, 64, 4, 4]

        out = self.net(h)  # [B, c*2, h, w]

        mean, logstd = out[:, :self.n_classes], out[:, self.n_classes:] # [B, c, h, w]
        logstd = logstd.clip(minClip, maxClip)
        std = torch.exp(logstd)
        # std = torch.ones_like(mean) * STD

        return mean, std
    

    # to draw sample from the learnt probabilistic model
    def sample(self, state, belief):
        batch_shape = state.shape[:-1]
        mean, std = self.forward(state, belief)
        eps = torch.randn_like(std)
        out = mean + eps * std # [B, c, h, w]
        out = out.permute(0,2,3,1) # [B, h, w, c]
        # restore batch dims 
        out = out.reshape(*batch_shape, *out.shape[-3:]) # [b_dims, h, w, c]
        return out

    # calculates log p(y|x)
    def log_prob(self, state, belief, y): # y.shape = [b_dims, c, h, w]
        batch_shape = state.shape[:-1]
        y = y.flatten(-3, -1) # [b_dims, c*h*w]

        mean, std = self.forward(state, belief) # [B, c, h, w]

        # restore batch_shape
        mean = mean.reshape(*batch_shape, *mean.shape[-3:]) # [b_dims, c, h, w]
        std = std.reshape(*batch_shape, *std.shape[-3:])

        # flatten c, h, w
        mean = mean.flatten(-3, -1) # [b_dims, c*h*w]
        std = std.flatten(-3, -1)

        dis = tdist.independent.Independent(tdist.Normal(mean, std), 1)
        lp = dis.log_prob(y) # [b_dims]

        # lp =  -0.5 * ((mean - y) ** 2) # [b_dims, c*h*w]
        # lp = lp.sum(-1) # [b_dims]
        
        return lp
    


# RSSM used for both Representation model p(s_t | s_t-1, a_t-1, o_t) and Transition model q(s_t | s_t-1, a_t-1)
class RSSM(nn.Module):
    def __init__(self, in_dim, o_dim, belief_dim, h_dim, out_dim, batch_size, n_channels, device):
        super().__init__()
        self.fc1_embed = nn.Linear(in_dim, h_dim) # layer to embed input (s_t, a_t)
        self.fc2_embed = nn.Linear(h_dim, belief_dim)
        self.norm_gru = nn.LayerNorm(belief_dim)

        # init deterministic recurrent net (shared between p and q models)
        self.gru_cell = nn.GRUCell(belief_dim, belief_dim) # belief h_t = f(h_t-1, s_t-1, a_t-1)

        # init stochastic net (separate layers for p and q models)
        self.obs_encoder = ObsEncoder_Strided(h_dim, n_channels, dropout=0.1)
        
        self.p_fc1 = nn.Linear(belief_dim + h_dim, h_dim)
        self.p_fc2 = nn.Linear(h_dim, h_dim)
        self.norm_plogit = nn.LayerNorm(h_dim)
        self.p_fc3_logits = nn.Linear(h_dim, out_dim)
        
        self.q_fc1 = nn.Linear(belief_dim, h_dim)
        self.q_fc2 = nn.Linear(h_dim, h_dim)
        self.norm_qlogit = nn.LayerNorm(h_dim)
        self.q_fc3_logits = nn.Linear(h_dim, out_dim)
        self.silu = nn.SiLU()
        self.device = device

    # forward pass through RSSM
    def forward(self, prev_belief, x, o=None):
        logits_prior, logits_posterior = None, None

        x = self.silu(self.fc1_embed(x))
        x = self.fc2_embed(x)
        x = self.norm_gru(x)
        belief = self.gru_cell(x, prev_belief)
        
        h = self.silu(self.q_fc1(belief))
        h = self.silu(self.norm_qlogit(self.q_fc2(h)))
        logits_prior = self.q_fc3_logits(h)

        if o is not None:
            o = self.obs_encoder(o)
            h_o = torch.cat((belief, o), dim=1)
            h = self.silu(self.p_fc1(h_o))
            h = self.silu(self.norm_plogit(self.p_fc2(h)))
            logits_posterior = self.p_fc3_logits(h)

        return belief, logits_prior, logits_posterior

    # to draw sample from the learnt probabilistic model
    def sample(self, prev_belief, x, o=None):
        state_prior, state_posterior = None, None
        belief, logits_prior, logits_posterior = self.forward(prev_belief, x, o)

        def rsample(logits):
            batch_shape, item_shape = logits.shape[:-1], logits.shape[-1]

            logits = logits.reshape(*batch_shape, n_latents, n_classes)
            dis = OneHotDist(logits=logits, unimix_ratio=0.01)
            state = dis.rsample() # gumbel-softmax sample is differentiable
            # state = state + dis.probs - dis.probs.clone().detach() # for straight through gradient

            state = state.flatten(start_dim=-2, end_dim=-1)
            state = state.reshape(*batch_shape, item_shape)
            return state 

        if logits_prior is not None:
            state_prior = rsample(logits_prior)

        if logits_posterior is not None:
            state_posterior = rsample(logits_posterior)

        return belief, state_prior, state_posterior, logits_prior, logits_posterior

    # function to calculate KL divergence
    def kl(self, logits_left, logits_right):

        def reshape_logits(logits):
            batch_shape = logits.shape[:-1]
            logits = logits.reshape(*batch_shape, n_latents, n_classes)
            return logits 

        logits_left = reshape_logits(logits_left)
        logits_right = reshape_logits(logits_right)

        # (..., K), (..., K)
        logprob_left = torch.log_softmax(logits_left, -1)
        logprob_right = torch.log_softmax(logits_right, -1)
        prob = torch.softmax(logits_left, -1)
        return (prob * (logprob_left - logprob_right)).sum(-1)  # sum over n_classes. So out.shape = [horizon, batch, n_latents]



# Discount Model - parameterized bernouli dist
class Discount_Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.silu = nn.SiLU()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
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
    
    # calculate mean - used during imagination rollout
    def mean(self, state, belief):
        logits = self.forward(state, belief)
        dis = tdist.independent.Independent(tdist.Bernoulli(logits=logits), 1)
        return dis.mean 
    

# Reward model - parameterized symexp_twohot
class Reward_Model(nn.Module):
    def __init__(self, in_dim, h_dim, n_bins):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, n_bins)
        self.silu = nn.SiLU()
        self.n_bins = n_bins

        # init last layer weights to zero
        with torch.no_grad():
            self.fc3.weight.zero_()
            self.fc3.bias.zero_()

    # forward pass through the stochastic net
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
        logits = self.fc3(h)
        return logits
    
    # prepare symexp_twohot dist
    def get_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = symexp_twohot(logits, bin_num=self.n_bins)
        return dis

    # get mode - used during imagination rollout
    def mode(self, state, belief):
        dis = self.get_dist(state, belief)
        mode = dis.mode()
        return mode

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        dis = self.get_dist(state, belief)
        lp = dis.log_prob(y)
        return lp


# Critic model - parameterized symexp_twohot
class Critic(nn.Module):
    def __init__(self, s_dim, belief_dim, h_dim, n_bins):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + belief_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, n_bins)
        self.silu = nn.SiLU()
        self.n_bins = n_bins

        # init last layer weights to zero
        with torch.no_grad():
            self.fc4.weight.zero_()
            self.fc4.bias.zero_()

    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
        h = self.silu(self.fc3(h))
        val = self.fc4(h)
        return val
    
    # prepare symexp_twohot dist
    def get_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = symexp_twohot(logits, bin_num=self.n_bins)
        return dis

    # get mode - used during imagination rollout
    def mode(self, state, belief):
        dis = self.get_dist(state, belief)
        mode = dis.mode()
        return mode

    # calculates log p(y|x)
    def log_prob(self, state, belief, y):
        dis = self.get_dist(state, belief)
        lp = dis.log_prob(y)
        return lp
    


# actor network - parameterizing one-hot categorical
class Actor(nn.Module):
    def __init__(self, s_dim, belief_dim, a_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + belief_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, h_dim)
        self.fc5_logits = nn.Linear(h_dim, a_dim)
        self.silu = nn.SiLU()

    # returns the logits of one_hot_categorical distribution representing the policy
    def forward(self, state, belief):
        x = torch.cat((state, belief), dim=-1)
        h = self.silu(self.fc1(x))
        h = self.silu(self.fc2(h))
        h = self.silu(self.fc3(h))
        h = self.silu(self.fc4(h))
        logits = self.fc5_logits(h)
        return logits

    # returns the policy as one_hot_categorical distribution
    def policy_dist(self, state, belief):
        logits = self.forward(state, belief)
        dis = OneHotDist(logits=logits, unimix_ratio=0.01) 
        return dis

    # returns policy log_prob
    def policy_logprob(self, state, belief, y):
        policy_dis = self.policy_dist(state, belief)
        lp = policy_dis.log_prob(y)
        return lp


# used to calculate scale for actor loss
class ReturnEMA(nn.Module):

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        # Using out-of-place update for torch.compile compatibility
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()



# replay buffer
class ReplayBuffer:
    def __init__(self, buf_size, seq_len, batch_size, o_dim, a_dim, max_ep_steps, frac, device):
        self.buf_size = buf_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buf_observation = np.zeros((buf_size, *o_dim))
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
    def __init__(self, o_dim, s_dim, a_dim, belief_dim, h_dim, seq_len, imagination_horizon, df, buf_size, batch_size, lr_actor, lr_critic, lr_model, _lambda, tau, rho, eta, max_ep_steps, frac, n_bins, n_channels, device):
        super().__init__()
        self.actor = Actor(s_dim, belief_dim, a_dim, h_dim).to(device)
        self.critic_V = Critic(s_dim, belief_dim, h_dim, n_bins).to(device)
        self.target_critic_V = deepcopy(self.critic_V)
        self.replay_buffer = ReplayBuffer(buf_size, seq_len, batch_size, o_dim, a_dim, max_ep_steps, frac, device)
        self.rssm = RSSM(s_dim + a_dim, o_dim, belief_dim, h_dim, s_dim, batch_size, n_channels, device).to(device)
        self.df_model = Discount_Model(s_dim + belief_dim, h_dim, 1).to(device)
        self.reward_model = Reward_Model(s_dim + belief_dim, h_dim, n_bins).to(device)
        self.observation_model = Observation_Model_Deconv_Gaussian(s_dim, belief_dim, h_dim, n_channels).to(device)
        self.return_ema = ReturnEMA(device).to(device)
        # self.state_model = StochasticNet_Gaussian(o_dim, h_dim, s_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic_V = torch.optim.Adam(params=self.critic_V.parameters(), lr=lr_critic)
        self.optimizer_model = torch.optim.Adam(params=list(self.rssm.parameters()) + list(self.df_model.parameters()) + \
        list(self.reward_model.parameters()) + list(self.observation_model.parameters()), lr=lr_model)
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
        self._lambda = _lambda
        self.tau = tau # used when updating target_critic_V
        self.rho = rho # used for weighing actor dynamics loss and actor reinforce loss
        self.eta = eta # used for weighing entropy regulaization in actor loss
        self.batch_size = batch_size

    def get_action(self, state, belief):
        policy = self.actor.policy_dist(state, belief)
        action = policy.rsample() # gumbel-softmax sample is differentiable
        # # for straight through gradient
        # action = action + policy.probs - policy.probs.clone().detach()
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
        rewards = rewards[1:] # r[t+2 : t+H+1]
        discounts = discounts[1:] # df[t+2 : t+H+1]
        state_values_target = state_values_target[1:] # v[t+2 : t+H+1]

        for t in range(len(rewards)-1, -1, -1): # t is just going from last element to first element, since all arrays are of length H
            accumulator = rewards[t] + discounts[t] * ( (1 - self._lambda)*state_values_target[t] + self._lambda*accumulator )
            # V_lambda[j] = reward[j+1] + df[j+1] * ( (1-lambda) * V_t[j+1] + lambda * V_lambda[j+1] )
            lambda_returns = [accumulator] + lambda_returns
        lambda_returns = torch.stack(lambda_returns, dim=0) # V_lambda[t+1 : t+H]
        return lambda_returns


    def train(self):

        # sample from replay buffer
        observation, action, reward, done = self.replay_buffer.sample() 
        observation = observation.permute(0,1,4,2,3) # [H, b, c, h, w]

        # # shift samples
        # observation = observation[1:] # o[t : t+H]
        # reward = reward[:-1] # r[t-1 : t+H-1]
        # action = action[:-1] # a[t-1 : t+H-1]
        # done = done[:-1] # d[t-1 : t+H-1]


        #########################
        ## dynamics learning (using experience sampled from replay buffer)
        #########################

        # unfreeze dynamics model params
        self.unfreeze_model_params(self.rssm)
        self.unfreeze_model_params(self.df_model)
        self.unfreeze_model_params(self.reward_model)
        self.unfreeze_model_params(self.observation_model)
        # self.unfreeze_model_params(self.state_model)

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
        prev_action = torch.zeros(self.batch_size, self.a_dim).to(device)

        # rssm rollout 
        for t in range(observation.shape[0] - 1): # [t : t+H-1]

            # reset if done
            if t > 0:
                prev_state = prev_state * (1. - done[t-1])
                prev_belief = prev_belief * (1. - done[t-1])
                prev_action = prev_action * (1. - done[t-1])

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
            prev_action = action[t+1] # since a_t in replay buffer is action taken to reach o_t


        # rollout ended - stack lists into tensors
        beliefs = torch.stack(belief_list, dim=0) # h[t : t+H-1]
        states = torch.stack(state_posterior_list, dim=0) # s[t : t+H-1]
        logits_posterior = torch.stack(logits_posterior_list, dim=0) 
        logits_prior = torch.stack(logits_prior_list, dim=0) 


        ## calculate loss terms 

        # reward loss
        lp_reward = self.reward_model.log_prob(states, beliefs, reward[:-1]) # logp( r[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_reward = lp_reward.mean()

        # observation loss (reconstruction)
        lp_obs = self.observation_model.log_prob(states, beliefs, observation[:-1]) # logp( o[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_obs = lp_obs.mean() 

        # df loss
        lp_df = self.df_model.log_prob(states, beliefs, (1. - done[:-1])) # logp( d[t:t+H-1] | s[t:t+H-1], h[t:t+H-1] )
        lp_df = lp_df.mean() 
            
        # KL divergence - using free nats and weighings
        kl_dyn = self.rssm.kl(logits_posterior.detach(), logits_prior).sum(-1) # sum over n_latents. So shape = [horizon, batch]
        kl_rep = self.rssm.kl(logits_posterior, logits_prior.detach()).sum(-1) # sum over n_latents. So shape = [horizon, batch]
        kl_dyn = torch.clip(kl_dyn, min=1.)
        kl_rep = torch.clip(kl_rep, min=1.)
        kl_pq = 0.5 * kl_dyn + 0.1 * kl_rep
        kl_div = kl_pq.mean()

        # vib objective
        vib_objective = lp_reward + lp_obs + lp_df - kl_div

        # loss
        loss_dynamics = -vib_objective

        # update dynamics model
        self.optimizer_model.zero_grad()
        loss_dynamics.backward()
        nn.utils.clip_grad_norm_(list(self.rssm.parameters()) + list(self.df_model.parameters()) + list(self.reward_model.parameters()) + \
                                list(self.observation_model.parameters()) , 100., norm_type=2)
        self.optimizer_model.step()

        # loss accumulators for book keeping and plotting
        loss_reward = -lp_reward
        loss_obs = -lp_obs
        loss_df = -lp_df
        loss_kl = kl_div


        ####################
        ## behaviour learning (using imagined rollouts over the learnt dynamics model)
        ####################

        # freeze dynamics model params
        self.freeze_model_params(self.rssm)
        self.freeze_model_params(self.df_model)
        self.freeze_model_params(self.reward_model)
        self.freeze_model_params(self.observation_model)

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
        rewards = self.reward_model.mode(im_states, im_beliefs) # r[t+1 : t+H+1]
        discounts = self.df * self.df_model.mean(im_states, im_beliefs) # df[t+1 : t+H+1]
        state_values = self.critic_V.mode(im_states, im_beliefs).detach()  # v[t+1 : t+H+1]

        # calculate lambda returns 
        lambda_returns = self.calculate_lambda_return(rewards, discounts, state_values) # v_lambda[t+1 : t+H]

        ## calculate actor loss

        # create discount cumprods
        # discounts_overwriteFirst = torch.cat((torch.ones_like(discounts[:1]), discounts[1:]), dim=0) # [df[t+1] = 1] + df[t+2 : t+H+1]
        discounts_cumprod = torch.cumprod(discounts[:-1], dim=0).detach() # df[t+1 : t+H]

        # loss through dynamics 
        loss_actor_dynamics = -lambda_returns * discounts_cumprod
        loss_actor_dynamics = loss_actor_dynamics.sum(dim=0).mean() # sum over horizon dim and mean over batch dim

        # reinforce loss
        ret_offset, ret_scale = self.return_ema(lambda_returns)
        advantage = (lambda_returns - state_values[:-1]) / ret_scale # advantage[t+1 : t+H]
        loss_actor_reinforce = (-log_pi[1:].unsqueeze(-1) * advantage.detach()) * discounts_cumprod 
        loss_actor_reinforce = loss_actor_reinforce.sum(dim=0).mean()

        # policy entropy for regularization (and encourage exploration)
        policy_entropy = entropy_pi[1:].unsqueeze(-1) * discounts_cumprod
        policy_entropy = policy_entropy.sum(dim=0).mean()

        # total loss for actor - weight by discount factor
        loss_actor = (1 - self.rho) * loss_actor_dynamics + self.rho * loss_actor_reinforce - self.eta * policy_entropy

        ## calculate critic loss 

        target_values = self.target_critic_V.mode( im_states[:-1], im_beliefs[:-1] ).detach()

        lp_critic = self.critic_V.log_prob( im_states[:-1].detach(), im_beliefs[:-1].detach(), lambda_returns ) + \
                      self.critic_V.log_prob( im_states[:-1].detach(), im_beliefs[:-1].detach(), target_values )

        loss_critic = -lp_critic.sum(dim=0).mean() # sum over horizon dim and mean over batch dim

        ## update actor and critic

        self.optimizer_actor.zero_grad()
        self.optimizer_critic_V.zero_grad()

        loss_actor.backward()
        loss_critic.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters() , 100., norm_type=2)
        nn.utils.clip_grad_norm_(self.critic_V.parameters() , 100., norm_type=2)

        self.optimizer_actor.step()
        self.optimizer_critic_V.step()        

        return loss_dynamics, loss_reward, loss_obs, loss_df, loss_kl, loss_actor, loss_critic, policy_entropy, ret_scale


# utility function to preprocess atari frame 
def preprocess_atari(obs):
    # # Grayscale
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # [210, 160]
    # gray = obs # [210, 160, 3]

    # Crop score area and resize
    gray = gray[34:194]                          # [160, 160]
    gray = cv2.resize(gray, (64, 64), 
                      interpolation=cv2.INTER_AREA)  # [64, 64]
    gray = gray.astype(np.float32) / 255.0
    gray = gray * 2 - 1                             # normalize to [-1, 1]

    gray = torch.from_numpy(gray).float().clip(-1,1)
    gray = gray.unsqueeze(-1)
    return gray                   



# main
if __name__ == '__main__':

    # hyperparams
    n_channels = 1
    n_bins = 255
    h_dim = 256 # 512
    n_latents = 32
    n_classes = 16 # 32
    s_dim = n_latents * n_classes 
    belief_dim = 512 # 1024
    lr_actor = 4e-5 # 4e-5
    lr_critic = 1e-4 # 1e-4
    lr_model = 2e-4 # 2e-4
    sample_seq_len = 64 # 50 # length of contiguous sequence sampled from replay buffer (when training)
    imagination_horizon = 16 # 15 # length of imagined rollouts using the learnt dynamics model (when behaviour learning)
    _lambda = .95 # lambda - used to calculate lambda return
    tau = 0.02 # used when updating target_critic_V
    target_critic_update_step = 1
    rho = 1 # used for weighing actor dynamics loss and actor reinforce loss
    eta = 3e-4 # used for weighing entropy regulaization in actor loss
    df = 0.997
    frac = 0 # 0.25
    reward_intrinsic_scale = 0.1
    delay_ep = 0 # 10
    # STD = 1
    minClip, maxClip = -2, 2
    random_seed = 1010
    batch_size = 128 # 64
    replay_buffer_size = 10**5
    num_episodes = 150 * 3
    init_random_episodes = 0
    num_train_calls = 15  
    train_episode = 1
    record_episode = num_episodes // 10
    action_repeat = 1 # 2
    explore_minLimit, explore_maxLimit = 0.05, 0.4 # 0.1, 0.9
    explore_decay = (explore_maxLimit - explore_minLimit) / (num_episodes * 0.8) # 50 # 100
    max_ep_steps = 500 # this isn't used for anything

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
    a_dim = env.action_space.n
    o_dim = env.observation_space.shape
    o_dim = (64, 64, n_channels) # since frames are cropped during preprocessing
    print('a_dim: ', a_dim)
    print('o_dim: ', o_dim)

    # hyperparam dict
    hyperparam_dict = {}
    hyperparam_dict['env'] = 'Breakout-v5'
    hyperparam_dict['algo'] = 'dreamerV3_r2_noICMdream_rssmNorms'
    hyperparam_dict['Sdim'] = str(s_dim)
    hyperparam_dict['Bdim'] = str(belief_dim)
    hyperparam_dict['Hdim'] = str(h_dim)
    # hyperparam_dict['lrActor'] = str(lr_actor)
    # hyperparam_dict['lrCritic'] = str(lr_critic)
    # hyperparam_dict['lrModel'] = str(lr_model)
    # hyperparam_dict['_lambda'] = str(_lambda)
    hyperparam_dict['L'] = str(sample_seq_len)
    hyperparam_dict['H'] = str(imagination_horizon)
    # hyperparam_dict['rho'] = str(rho)
    # hyperparam_dict['eta'] = str(eta)
    # hyperparam_dict['tau'] = str(tau)
    # hyperparam_dict['df'] = str(df)
    hyperparam_dict['B'] = str(batch_size)
    hyperparam_dict['EP'] = str(num_episodes)
    hyperparam_dict['trCalls'] = str(num_train_calls)
    hyperparam_dict['trEP'] = str(train_episode)
    # hyperparam_dict['initEP'] = str(init_random_episodes)
    # hyperparam_dict['VTupdateSt'] = str(target_critic_update_step)
    # hyperparam_dict['maxSt'] = str(max_time_steps)
    # hyperparam_dict['random_seed'] = str(random_seed)
    # hyperparam_dict['actionRep'] = str(action_repeat)
    # hyperparam_dict['buffSz'] = str(replay_buffer_size)
    # hyperparam_dict['xploreMin'] = str(explore_minLimit)
    # hyperparam_dict['xploreMax'] = str(explore_maxLimit)
    hyperparam_dict['xploreDecay'] = str(explore_decay)
    hyperparam_dict['frac'] = str(frac)
    # hyperparam_dict['minClip'] = str(minClip)
    # hyperparam_dict['maxClip'] = str(maxClip)
    # hyperparam_dict['std'] = str(STD)
    hyperparam_dict['delayEP'] = str(delay_ep)
    hyperparam_dict['intrinsicRScale'] = str(reward_intrinsic_scale)

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
    agent = DreamerV2(o_dim, s_dim, a_dim, belief_dim, h_dim, sample_seq_len, imagination_horizon, df, replay_buffer_size, batch_size, lr_actor, lr_critic, lr_model, _lambda, tau, rho, eta, max_ep_steps, frac, n_bins, n_channels, device)

    # results and stats containers
    ep_return_list, ep_return_extrinsic_list = [], []
    loss_dynamics_list = []
    loss_reward_list = []
    loss_obs_list = []
    loss_df_list = []
    loss_kl_list = []
    loss_actor_list = []
    loss_critic_list = []
    policy_entropy_list = []
    ret_scale_list = []
    unique_states_visited = set()
    n_states_list = []


    # interactive episodes
    total_gradient_steps = 0
    interaction_steps = 0 
    explore_const = explore_maxLimit

    for ep in tqdm(range(num_episodes)):
        done = False
        ep_return, ep_return_extrinsic = 0, 0
        ep_steps = 0
        frames = []

        explore_const = max(explore_const - explore_decay, explore_minLimit)

        # first observation of the episode
        observation, info = env.reset()

        # init state and action
        prev_state = torch.zeros(1, s_dim)
        prev_action = torch.zeros(1, a_dim)
        prev_belief = torch.zeros(1, belief_dim)

        # first step 
        action_numpy = np.zeros(a_dim)
        reward = 0
        observation = preprocess_atari(observation)
        oar_tuple = [observation, action_numpy, reward, done]
        agent.replay_buffer.add(oar_tuple)


        while not done:

            # infer state from observation using representation model
            rssm_input = torch.cat( (prev_state, prev_action), dim=1 ).to(device)
            obs_input = observation.unsqueeze(0).to(device)
            obs_input = obs_input.permute(0,3,1,2) # [1, 3, 210, 160]
            prev_belief = prev_belief.to(device)
            belief, state_prior, state, logits_prior, logits_posterior = agent.rssm.sample(prev_belief, rssm_input, obs_input)

            if ep_steps % action_repeat == 0: # action repeat
                # sample action from the (stochastic) policy
                action, policy = agent.get_action(state, belief)
                # eps-greedy exploration
                action = agent.action_exploration(action, explore_const)
            else:
                action = prev_action

            action_numpy = action.squeeze(0).detach().cpu().numpy()
            action_scalar = np.argmax(action_numpy).squeeze()
            next_observation, reward_extrinsic, terminated, truncated, _ = env.step(action_scalar)
            done = terminated or truncated

            # calculate intrinsic reward - ICM dream
            reward_intrinsic = 0
            if ep > delay_ep:
                kl_pq = agent.rssm.kl(logits_posterior.detach(), logits_prior.detach()).sum(-1)
                reward_intrinsic = kl_pq.mean().item() * reward_intrinsic_scale

            reward = reward_extrinsic + reward_intrinsic

            next_observation = preprocess_atari(next_observation)

            oar_tuple = [next_observation, action_numpy, reward_extrinsic, done]
            agent.replay_buffer.add(oar_tuple)

            # record frame (both real and imagination)
            if (ep+1) % record_episode == 0: 
                # get imagination frame
                imagined_frame = agent.observation_model.sample(state_prior, belief).squeeze().cpu().detach() 
                imagined_frame = imagined_frame * 0.5 + 0.5
                imagined_frame = (imagined_frame * 255).clip(0, 255).int()
                real_frame = env.render()
                real_frame = real_frame[34:194] # [160, 160]
                real_frame = cv2.cvtColor(real_frame, cv2.COLOR_RGB2GRAY) 
                real_frame = cv2.resize(real_frame, (64, 64), interpolation=cv2.INTER_AREA)  # [64, 64]
                real_frame = torch.tensor(real_frame).clip(0, 255).int()
                # concat real and imagination frame
                concat_frame = torch.cat([real_frame, imagined_frame], dim=1)
                concat_frame = concat_frame.numpy().astype(np.uint8)
                frames.append(concat_frame)


            # for next step in episode
            observation = next_observation
            prev_state = state.detach().cpu()
            prev_action = action.detach().cpu()
            prev_belief = belief.detach().cpu()

            # unique states set 
            prev_state_clone = prev_state.clone().bool()
            prev_state_clone = torch.reshape(prev_state_clone, (n_latents, n_classes))
            x, y = torch.where(prev_state_clone == True)
            state_id = y.sum().item() + n_latents 
            unique_states_visited.add(state_id)
            n_states_list.append(len(unique_states_visited))

            # ep_return += (df ** ep_steps) * reward
            ep_return += reward
            ep_return_extrinsic += reward_extrinsic
            ep_steps += 1
            interaction_steps += 1

        ## episode ended
        # train agent
        if ep % train_episode == 0:
            for _ in range(num_train_calls):

                l_dyn, l_rew, l_obs, l_df, l_kl, l_act, l_cri, p_entr, ret_scale = agent.train()
                loss_dynamics_list.append(l_dyn.item())
                loss_reward_list.append(l_rew.item())
                loss_obs_list.append(l_obs.item())
                loss_df_list.append(l_df.item())
                loss_kl_list.append(l_kl.item())
                loss_actor_list.append(l_act.item())
                loss_critic_list.append(l_cri.item())
                policy_entropy_list.append(p_entr.item())
                ret_scale_list.append(ret_scale.item())

                total_gradient_steps += 1
                if total_gradient_steps % target_critic_update_step == 0:
                    # update critic
                    with torch.no_grad():
                        for target_param, current_param in zip(agent.target_critic_V.parameters(), agent.critic_V.parameters()):
                            target_param.data.copy_(agent.tau * current_param.data + (1 - agent.tau) * target_param.data)

        # store episode stats
        ep_return_list.append(ep_return)
        ep_return_extrinsic_list.append(ep_return_extrinsic)
        if ep % (num_episodes//10) == 0:
            print('ep:{} \t ep_return:{:.2f} \t ep_return_extrinsic:{:.2f}'.format(ep, ep_return, ep_return_extrinsic))

        # save episode recording 
        if (ep+1) % record_episode == 0:
            imageio.mimsave('plots/' + hyperstr + '_' + str(ep) + '.gif', frames, fps=16)

    print('total interaction steps: ', interaction_steps)



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
ep_returns_extrinsic_moving_mean = ep_return_extrinsic_list
loss_dynamics_moving_mean = get_moving_mean_list(loss_dynamics_list)
loss_reward_moving_mean = get_moving_mean_list(loss_reward_list)
loss_obs_moving_mean = get_moving_mean_list(loss_obs_list)
loss_df_moving_mean = get_moving_mean_list(loss_df_list)
loss_kl_moving_mean = get_moving_mean_list(loss_kl_list)
loss_actor_moving_mean = get_moving_mean_list(loss_actor_list)
loss_critic_moving_mean = get_moving_mean_list(loss_critic_list)
policy_entropy_moving_mean = get_moving_mean_list(policy_entropy_list)


# plot results
fig, ax = plt.subplots(2,4, figsize=(20,10))

ax[0,0].plot(ep_returns_moving_mean, color='blue', label='ep_return')
ax[0,0].plot(ep_returns_extrinsic_moving_mean, color='green', label='ep_return_extrinsic')
ax[0,0].legend()
ax[0,0].set_title('return:{:.2f} return_extrinsic:{:.2f}'.format(ep_returns_moving_mean[-1], ep_returns_extrinsic_moving_mean[-1]))
ax[0,0].set(xlabel='episode')
ax[0,0].grid()

ax[0,1].plot(loss_dynamics_moving_mean, color='red', label='dynamics_loss')
ax[0,1].plot(loss_reward_moving_mean, color='lime', label='reward_loss')
ax[0,1].plot(loss_obs_moving_mean, color='blue', label='obs_loss')
ax[0,1].plot(loss_df_moving_mean, color='magenta', label='df_loss')
ax[0,1].plot(loss_kl_moving_mean, color='black', label='kl_loss')
ax[0,1].legend()
ax[0,1].set_title('dyn:{:.2f} obs:{:.2f} rew:{:.2f}'.format(loss_dynamics_moving_mean[-1], loss_obs_moving_mean[-1], loss_reward_moving_mean[-1]))
ax[0,1].set(xlabel='steps')
ax[0,1].grid()
# ax[0,1].set_ylim([-500,500])

ax[1,0].plot(loss_actor_moving_mean, color='blue', label='actor_loss')
ax[1,0].legend()
ax[1,0].set_title('actor_loss:{:.3f}'.format(loss_actor_moving_mean[-1]))
ax[1,0].set(xlabel='steps')
ax[1,0].grid()

ax[1,1].plot(loss_critic_moving_mean, color='gray', label='critic_loss')
ax[1,1].legend()
ax[1,1].set_title('critic_loss:{:.3f}'.format(loss_critic_moving_mean[-1]))
ax[1,1].set(xlabel='steps')
ax[1,1].grid()

ax[0,2].plot(loss_df_moving_mean, color='magenta', label='df_loss')
ax[0,2].plot(loss_kl_moving_mean, color='black', label='kl_loss')
ax[0,2].legend()
ax[0,2].set_title('df_loss:{:.3f} kl_loss:{:.3f}'.format(loss_df_moving_mean[-1], loss_kl_moving_mean[-1]))
ax[0,2].set(xlabel='steps')
ax[0,2].grid()

ax[0,3].plot(loss_reward_moving_mean, color='green', label='rew_loss')
ax[0,3].legend()
ax[0,3].set_title('rew_loss:{:.3f}'.format(loss_reward_moving_mean[-1]))
ax[0,3].set(xlabel='steps')
ax[0,3].grid()

ax[1,2].plot(n_states_list, color='red', label='unique_states')
ax[1,2].legend()
ax[1,2].set_title(f'unique_states:{n_states_list[-1]}') 
ax[1,2].set(xlabel='steps')
ax[1,2].grid()

ax[1,3].plot(policy_entropy_moving_mean, color='blue', label='policy_entropy')
ax[1,3].plot(ret_scale_list, color='green', label='ret_scale')
ax[1,3].legend()
ax[1,3].set_title('policy_entropy:{:.3f} ret_scale:{:2f}'.format(policy_entropy_moving_mean[-1], ret_scale_list[-1]))
ax[1,3].set(xlabel='steps')
ax[1,3].grid()

plt.savefig('plots/' + hyperstr + '.png')