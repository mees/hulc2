import numpy as np
from omegaconf import open_dict
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

from hulc2.affordance.datasets.transforms import NormalizeVectorInverse
from hulc2.affordance.models.language_encoders.clip_lang_encoder import CLIPLang
import hulc2.models as models


class DepthModule(LightningModule):
    def __init__(
        self, cfg, in_shape=(200, 200, 3), transforms=None, depth_dist=None, depth_norm_values=None, *args, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.in_shape = in_shape
        self.depth_est_fcn = depth_dist
        # append depth transforms to cfg
        with open_dict(cfg):
            cfg.depth_norm_values = depth_norm_values

        # Padding images
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
        self.output_dim = 1

        # Build nets
        self.init_depth_transforms(depth_norm_values)
        self._build_model()
        self.save_hyperparameters()

    def init_depth_transforms(self, depth_norm_values):
        self.depth_norm_inverse = NormalizeVectorInverse(depth_norm_values["mean"], depth_norm_values["std"])

    def _build_model(self):
        # Initialize language encoder
        lang_enc_model = models.lang_encoders[self.cfg.streams.lang_enc]
        self.lang_encoder = lang_enc_model(self.device)

        # Init depth net
        depth_est_model = models.deth_est_nets[self.depth_est_fcn]
        self.depth_stream = depth_est_model(self.in_shape, self.output_dim, self.cfg, self.device)

    def forward(self, inp):
        """
        inp(dict):
            - 'img'(torch.Tensor):
        """
        inp_img = inp["img"]
        lang_goal = inp["lang_goal"]
        in_data = F.pad(inp_img, self.padding, mode="constant")
        in_tens = in_data.to(dtype=torch.float, device=self.device)

        # in_tens = inp["img"]
        text_enc = self.lang_encoder.encode_text(lang_goal)
        dist, _info = self.depth_stream(in_tens, text_enc)
        out = {"depth_dist": dist}
        return out, _info

    def predict(self, inp, transforms):
        inp_img = torch.tensor(inp["img"]).permute((2, 0, 1)).unsqueeze(0).to(self.device_type)
        inp_img = transforms(inp_img)
        # dist, _info = self.depth_est(inp['img'], inp['lang_goal'])
        net_inp = {"img": inp_img, "lang_goal": inp["lang_goal"]}
        output, _info = self.forward(net_inp)
        depth_pred = self.depth_stream.sample(output["depth_dist"])
        depth_pred = depth_pred.squeeze().detach().cpu().numpy()
        return depth_pred

    def criterion(self, inp, pred, label, compute_err):
        if self.depth_stream.normalized:
            depth_label = "normalized_depth"
        else:
            depth_label = "depth"

        gt_depth = label[depth_label].unsqueeze(-1).float()
        depth_loss = self.depth_stream.loss(pred["depth_dist"], gt_depth)

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # Always compute error w.r.t true depth (not normalized)
            # output, _info = self.forward(inp)
            depth_pred = self.depth_stream.sample(pred["depth_dist"])
            depth_sample = depth_pred.detach().cpu().numpy().squeeze()
            unormalized_depth = label["depth"].detach().cpu().numpy()
            depth_error = np.sum(np.abs(depth_sample - unormalized_depth))
            err = {"depth": depth_error, "px_dist": depth_error}  # Hack for training depth w/ aff cfg

        loss = depth_loss
        return loss, err

    def training_step(self, batch, batch_idx):
        frame, label = batch

        # Get training losses.
        pred, _info = self.forward(frame)
        total_loss, err = self.criterion(frame, pred, label, compute_err=True)
        bs = frame["img"].shape[0]

        self.log("Training/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=bs)
        for err_type, value in err.items():
            self.log("Training/%s_err" % err_type, value, on_step=False, on_epoch=True, batch_size=bs)
        return total_loss

    def validation_step(self, batch, batch_idx):
        frame, label = batch

        pred, _info = self.forward(frame)
        total_loss, err = self.criterion(frame, pred, label, compute_err=True)
        bs = frame["img"].shape[0]

        self.log("Validation/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=bs)
        for err_type, value in err.items():
            self.log("Validation/%s_err" % err_type, value, on_step=False, on_epoch=True, batch_size=bs)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optim
