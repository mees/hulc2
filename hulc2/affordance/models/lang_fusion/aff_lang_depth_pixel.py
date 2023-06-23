import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hulc2.models as models
from hulc2.utils.tensor_utils import unravel_idx


class AffDepthLangFusionPixel(nn.Module):
    def __init__(self, modules_cfg, in_shape, cfg, device, output_dim=1):
        super().__init__()
        self.fusion_type = cfg.attn_stream_fusion_type
        self.modules_cfg = modules_cfg
        self.cfg = cfg
        self.batchnorm = self.cfg.batchnorm

        self.padding = np.zeros((3, 2), dtype=int)  # H, W, C
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)  # H, W, C

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = list(in_shape)

        # for torch: left, right,(W) top, bottom,(H) front, back(C)
        self.padding = self.padding[[1, 0, 2]]  # C, H, W
        self.padding = tuple(self.padding.flatten())
        self.in_shape = in_shape
        self.output_dim = output_dim
        self._build_nets(device)

    @property
    def decoder_layers(self):
        return self.attn_stream.decoder_layers

    def _build_nets(self, device):
        aff_net_fcn, lang_enc_fcn, depth_est_fcn = self.modules_cfg
        lang_enc_model = models.lang_encoders[lang_enc_fcn]
        aff_net_model = models.vision_encoders[aff_net_fcn]

        # Clip loads both the language and visual encoder
        self.lang_encoder = lang_enc_model(freeze_backbone=self.cfg.freeze_encoder.lang)
        if aff_net_fcn == "clip":
            kwargs = {"clip_rn50": self.lang_encoder.model}
        else:
            kwargs = {}

        # Encoder and decoder
        self.aff_stream = aff_net_model(
            device=device, input_shape=self.in_shape, output_dim=self.output_dim, cfg=self.cfg, **kwargs
        )
        # Optional
        if depth_est_fcn:
            depth_est_model = models.deth_est_nets[depth_est_fcn]
            _in_shape = self.aff_stream.calc_img_enc_size()
            self.depth_stream = depth_est_model(_in_shape, 1, self.cfg)
        else:
            self.depth_stream = None

        print(f"Aff FCN: {aff_net_fcn}, Depth Est: {depth_est_fcn}")

    def predict(self, inp_img, lang_goal):
        """
        inputs:
            inp_img(tensor): [B, C, W, H] inp batch of images with transforms already applied
            lang_goal: str or list of strings
        outputs:
            p0_px(np.array): B, 2
            depth_pred(np.array): B
            logits(np.array): B, H, W, 1
        """
        B = inp_img.shape[0]

        if isinstance(lang_goal, str):
            lang_goal = [lang_goal]

        output, _info = self.forward(inp_img, lang_goal)

        # Get aff predicted pixels
        pick_conf = output["aff"]
        logits = pick_conf.detach().cpu().numpy()  # B, H, W, 1
        indices = np.argmax(logits.reshape((B, -1)), -1)  # B
        p0_pix = unravel_idx(indices, shape=logits.shape[1:-1])

        if "depth_dist" in output:
            depth_pred = self.depth_stream.sample(output["depth_dist"])
            depth_pred = depth_pred.detach().cpu().numpy().squeeze()

            uncertainty = self.depth_stream.depth_norm_inverse(output["depth_dist"][-1])
            uncertainty = uncertainty.detach().cpu().numpy().squeeze()

        return p0_pix, depth_pred, uncertainty, logits.squeeze()

    def forward(self, inp_img, lang_goal, softmax=True):
        """
        Forward pass.
        input img has transforms already applied
        """
        in_data = F.pad(inp_img, self.padding, mode="constant")
        in_tens = in_data.to(dtype=torch.float)  # [B 3 H W]

        # FORWARD PASS TROUGH NETWORKS
        text_enc = self.lang_encoder.encode_text(lang_goal)
        # Shared language encoder
        logits, _info = self.aff_stream(in_tens, text_enc)
        if self.depth_stream:
            depth_in = _info["hidden_layers"][-1]  # encouder output
            depth_out = self.depth_stream(depth_in, text_enc)
        else:
            depth_out = None

        # Apply softmax
        c0 = np.array([self.padding[2], self.padding[0]])  # top(H), left(W)
        c1 = c0 + inp_img.shape[2:]
        logits = logits[:, :, c0[0] : c1[0], c0[1] : c1[1]]

        logits = logits.permute(0, 2, 3, 1)  # [B H W 1]
        aff_out = logits.reshape(logits.shape[0], np.prod(logits.shape[1:]))
        if softmax:
            aff_out = F.softmax(aff_out, dim=-1)
            aff_out = aff_out.reshape(logits.shape)

        output = {"aff": aff_out}
        if depth_out:
            output["depth_dist"], depth_info = depth_out
            _info["depth"] = depth_info
        return output, _info
