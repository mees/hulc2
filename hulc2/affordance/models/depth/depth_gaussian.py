import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Independent
import torchvision.models as models

import numpy as np
from hulc2.affordance.models.language_encoders.clip_lang_encoder import CLIPLang
from hulc2.affordance.datasets.transforms import NormalizeVectorInverse

class DepthEstimationGaussian(nn.Module):
    def __init__(self, input_shape, output_dim, cfg):
        super(DepthEstimationGaussian, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.lang_fusion_type = self.cfg['lang_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1

        # self.undo_norm = NormalizeInverse(mean=cfg.mean, std=cfg.std)
        # Use clip preprocessing
        # self.text_enc = CLIPLang(self.device)
        self.loss_fcn = nn.GaussianNLLLoss()

        self.normalized = cfg.normalized
        self.init_depth_transforms(cfg.depth_norm_values)
        self._build_decoder()

    def init_depth_transforms(self, depth_norm_values):
        self.depth_norm_inverse = NormalizeVectorInverse(depth_norm_values["mean"], depth_norm_values["std"])

    def sample(self, depth_dist, reparametrize=True):
        '''
            Sample from distribution and undo normalization.Used for inference
            output:
                depth(torch.tensor): estimated depth in world/camera frame
            input:
                depth_dist: output of forward pass
        '''
        dist, loc, scale = depth_dist
        if reparametrize:
            sample = dist.rsample()
        else:
            sample = dist.sample()
        
        if self.normalized:
            sample = self.depth_norm_inverse(sample)
            
        return sample

    def _build_decoder(self):
        # B, C, H, W
        self.proj_input_dim = 1024
        linear_in = np.prod(self.input_shape)
        hidden_dim = 256
        self.fc1 = nn.Linear(linear_in + self.proj_input_dim, hidden_dim * 3)
        self.fc2 = nn.Linear(hidden_dim * 3 + self.proj_input_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.depth_mu = nn.Linear(hidden_dim, 1)
        self.depth_sigma = nn.Linear(hidden_dim, 1)

    def loss(self, pred, gt_depth):
        dist, mu, sigma = pred
        depth_loss = self.loss_fcn(mu, gt_depth, sigma)
        # depth_loss = -pred["depth_dist"].log_prob(gt_depth).mean()
        # neg_samples = pred["depth_dist"].sample()
        # depth_loss += pred["depth_dist"].log_prob(neg_samples).mean()
        # pred_depth = out["depth_dist"].rsample()
        # depth_loss = F.mse_loss(pred_depth, gt_depth)
        return depth_loss

    def forward(self, x, text_enc):
        in_type = x.dtype
        x = x.to(in_type)

        # encode text
        l_enc, l_emb, l_mask  = text_enc
        l_input = l_enc.to(dtype=x.dtype)
    
        _info = {"hidden_layers": [x],
                 "l_input": l_input,
                 "fusion_type": self.lang_fusion_type}
        # Decoder
        B, C, H, W = x.shape
        x = x.reshape((B, -1))

        # Predict distribution
        x = torch.cat([x, l_input], -1)
        x = F.relu(self.fc1(x))
        
        x = torch.cat([x, l_input], -1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.depth_mu(x)
        log_sigma = self.depth_sigma(x)
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()

        # Sample
        dist = Independent(Normal(mu, sigma), 1)
        return (dist, mu, sigma), _info