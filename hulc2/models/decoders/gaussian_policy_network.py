import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from hulc2.models.decoders.action_decoder import ActionDecoder
from hulc2.models.decoders.utils.gripper_control import tcp_to_world_frame, world_to_tcp_frame
from hulc2.models.decoders.utils.rnn import gru_decoder, lstm_decoder, mlp_decoder, rnn_decoder

logger = logging.getLogger(__name__)

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class GaussianPolicyNetwork(ActionDecoder):
    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        hidden_size: int,
        out_features: int,
        n_mixtures: int,
        policy_rnn_dropout_p: float,
        log_scale_min: float,
        log_scale_max: float,
        num_layers: int,
        rnn_model: str,
        perceptual_emb_slice: tuple,
        gripper_control: bool,
    ):
        super(GaussianPolicyNetwork, self).__init__()
        self.plan_features = plan_features
        self.gripper_control = gripper_control
        in_features = (perceptual_emb_slice[1] - perceptual_emb_slice[0]) + latent_goal_features + plan_features
        self.rnn = eval(rnn_model)
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.gaussian_mixture_model = MDN(
            in_features=hidden_size,
            out_features=out_features,
            n_gaussians=n_mixtures,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
        )
        self.perceptual_emb_slice = perceptual_emb_slice
        self.hidden_state = None

    def clear_hidden_state(self) -> None:
        self.hidden_state = None

    def loss_and_act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi, sigma, mu, _ = self(latent_plan, perceptual_emb, latent_goal)
        # loss
        if self.gripper_control:
            actions_tcp = world_to_tcp_frame(actions, robot_obs)
            loss = self.gaussian_mixture_model.loss(pi, sigma, mu, actions_tcp)
        else:
            loss = self.gaussian_mixture_model.loss(pi, sigma, mu, actions)
        # act
        pred_actions = self._sample(pi, sigma, mu)
        if self.gripper_control:
            pred_actions_world = tcp_to_world_frame(pred_actions, robot_obs)
            return loss, pred_actions_world
        return loss, pred_actions

    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pi, sigma, mu, self.hidden_state = self(latent_plan, perceptual_emb, latent_goal, self.hidden_state)
        pred_actions = self._sample(pi, sigma, mu)
        if self.gripper_control:
            pred_actions_world = tcp_to_world_frame(pred_actions, robot_obs)
            return pred_actions_world
        else:
            return pred_actions

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pi, sigma, mu, _ = self(latent_plan, perceptual_emb, latent_goal)
        if self.gripper_control:
            actions_tcp = world_to_tcp_frame(actions, robot_obs)
            return self.gaussian_mixture_model.loss(pi, sigma, mu, actions_tcp)
        return self.gaussian_mixture_model.loss(pi, sigma, mu, actions)

    def _sample(self, *args, **kwargs):
        return self.gaussian_mixture_model.sample(*args, **kwargs)

    def forward(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        perceptual_emb = perceptual_emb[..., slice(*self.perceptual_emb_slice)]
        batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        latent_plan = latent_plan.unsqueeze(1).expand(-1, seq_len, -1)
        latent_goal = latent_goal.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([latent_plan, perceptual_emb, latent_goal], dim=-1)  # b, s, (plan + visuo-propio + goal)
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            x, h_n = self.rnn(x, h_0)
        else:
            x = self.rnn(x)
            h_n = None
        pi, std, mu = self.gaussian_mixture_model(x)
        return pi, std, mu, h_n


# Reference: https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn.py
class MDN(nn.Module):
    """Mixture Density Network - Gaussian Mixture Model, see Bishop, 1994
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        n_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxSxD): B is the batch size, S sequence length and D is the number of input dimensions.
    Output:
        (pi, sigma, mu) (BxSxK, BxSxKxO, BxSxKxO): B is the batch size,  S sequence length, K is the
            number of Gaussians, and O is the number of dimensions for each Gaussian. Pi is a multinomial
            distribution of the Gaussians. Sigma is the standard deviation of each Gaussian.
            Mu is the mean of each  Gaussian.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_gaussians: int,
        log_scale_min: float = -7.0,
        log_scale_max: float = 7.0,
    ):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_gaussians = n_gaussians
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.pi = nn.Sequential(  # priors - Softmax guarantees sum = 1
            nn.Linear(in_features, n_gaussians), nn.Softmax(dim=-1)
        )
        self.log_var = nn.Linear(in_features, out_features * n_gaussians)
        self.mu = nn.Linear(in_features, out_features * n_gaussians)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[0], x.shape[1]
        pi = self.pi(x)  # b, s, k number of gaussians
        log_var = self.log_var(x)
        log_scales = torch.clamp(log_var, min=self.log_scale_min, max=self.log_scale_max)  # avoid going to -inf / +inf
        std = torch.exp(log_scales)  # Guarantees that sigma is positive
        std = std.view(batch_size, seq_len, self.n_gaussians, self.out_features)  # b, s, k, o
        mu = self.mu(x)
        mu = mu.view(batch_size, seq_len, self.n_gaussians, self.out_features)  # b, s, k, o
        return pi, std, mu

    def loss(self, pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.
        """
        gmm = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=pi),
            component_distribution=D.Independent(D.Normal(mu, sigma), 1),
        )
        log_probs = gmm.log_prob(target)
        return -torch.mean(log_probs)

    def sample(self, pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        gmm = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=pi),
            component_distribution=D.Independent(D.Normal(mu, sigma), 1),
        )
        return gmm.sample()
