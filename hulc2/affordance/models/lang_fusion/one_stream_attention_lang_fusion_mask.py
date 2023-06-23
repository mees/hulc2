import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hulc2.affordance.models.lang_fusion.one_stream_attention_lang_fusion_pixel import AttentionLangFusionPixel


class AttentionLangFusionMask(AttentionLangFusionPixel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.output_dim = out_channels = n_classes
        if self.output_dim > 1:
            # Softmax over channels
            self.act_fnc = torch.nn.Softmax(1)
        else:
            self.act_fnc = torch.nn.Sigmoid()

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        in_data = F.pad(inp_img, self.padding, mode="constant")
        in_tens = in_data.to(dtype=torch.float)  # [B 3 H W]

        # Forward pass.
        aff_out, info = self.attend(in_tens, lang_goal)
        if softmax:
            aff_out = self.act_fnc(aff_out)

        c0 = np.array([self.padding[2], self.padding[0]])  # top(H), left(W)
        c1 = c0 + inp_img.shape[2:]
        aff_out = aff_out[:, :, c0[0] : c1[0], c0[1] : c1[1]]

        info["affordance"] = aff_out
        return info
