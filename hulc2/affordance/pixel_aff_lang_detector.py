import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import open_dict
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn

from hulc2.affordance.models.lang_fusion.aff_lang_depth_pixel import AffDepthLangFusionPixel
from hulc2.affordance.utils.losses import cross_entropy_with_logits
from hulc2.utils.img_utils import add_img_text, blend_imgs, get_transforms, resize_pixel
from hulc2.utils.tensor_utils import tt, unravel_idx


class PixelAffLangDetector(LightningModule):
    def __init__(
        self,
        model_cfg,
        optimizer,
        loss_weights,
        normalize_depth=False,
        in_shape=(200, 200, 3),
        transforms=None,
        depth_dist=None,
        depth_norm_values=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        # append depth transforms to cfg
        with open_dict(model_cfg):
            model_cfg.normalized = normalize_depth
            model_cfg.depth_norm_values = depth_norm_values
        self.optimizer_cfg = optimizer
        self.model_cfg = model_cfg
        self.in_shape = in_shape
        self._batch_loss = []
        self.cmd_log = logging.getLogger(__name__)
        self.pred_depth = depth_dist is not None
        self.depth_est_dist = depth_dist
        if transforms is not None:
            self.pred_transforms = get_transforms(transforms, self.in_shape[0])["transforms"]
        else:
            self.pred_transforms = nn.Identity()
        self._build_model()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # self.model.train()

        frame, label = batch

        # Get training losses.
        pred = self.forward(frame, softmax=False)
        total_loss, err, info = self.criterion(frame, pred, label, compute_err=False)

        bs = frame["img"].shape[0]

        self.log("Training/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=bs)
        for loss_fnc, value in info.items():
            self.log("Training/%s" % loss_fnc, value, on_step=False, on_epoch=True, batch_size=bs)

        for err_type, value in err.items():
            self.log("Training/%s_err" % err_type, value, on_step=False, on_epoch=True, batch_size=bs)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # self.model.eval()

        frame, label = batch

        pred = self.forward(frame, softmax=False)
        val_total_loss, err, info = self.criterion(frame, pred, label, compute_err=True)

        bs = frame["img"].shape[0]
        self.log("Validation/total_loss", val_total_loss, on_step=False, on_epoch=True, batch_size=bs)
        for loss_fnc, value in info.items():
            self.log("Validation/%s" % loss_fnc, value, on_step=False, on_epoch=True, batch_size=bs)

        for err_type, value in err.items():
            self.log("Validation/%s_err" % err_type, value, on_step=False, on_epoch=True, batch_size=bs)
        return dict(
            val_loss=val_total_loss,
            val_attn_dist_err=err["px_dist"],
            val_depth_err=err["depth"],
            n_imgs=batch[1]["p0"].shape[0],
        )

    def validation_epoch_end(self, all_outputs):
        total_dist_err = np.sum([v["val_attn_dist_err"] for v in all_outputs])
        total_depth_err = np.sum([v["val_depth_err"] for v in all_outputs])
        total_imgs = np.sum([v["n_imgs"] for v in all_outputs])
        mean_img_error = total_dist_err / total_imgs

        mean_depth_error = total_depth_err / total_imgs
        self.log("Validation/mean_dist_error", mean_img_error)
        self.log("Validation/mean_depth_error", mean_depth_error)

        print("\n Err - Dist: {:.2f}".format(total_dist_err))

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        return optim

    def _build_model(self):
        self.model = AffDepthLangFusionPixel(
            modules_cfg=[self.model_cfg.streams.vision_net, self.model_cfg.streams.lang_enc, self.depth_est_dist],
            in_shape=self.in_shape,
            cfg=self.model_cfg,
            device=self.device,
        )

    def forward(self, inp, softmax=True):
        inp_img = inp["img"]
        lang_goal = inp["lang_goal"]
        output, _info = self.model(inp_img, lang_goal, softmax=softmax)
        return output

    def criterion(self, inp, pred, label, compute_err=False):
        if self.model.depth_stream.normalized:
            depth_label = "normalized_depth"
        else:
            depth_label = "depth"

        # AFFORDANCE CRITERION #
        inp_img = inp["img"]
        B = inp_img.shape[0]

        # B, H, W, 1
        label_size = (inp_img.shape[0],) + inp_img.shape[2:]
        aff_label = torch.zeros(label_size)
        p0 = label["p0"].detach().cpu().numpy()  # B, 2
        aff_label[np.arange(B), p0[:, 0], p0[:, 1]] = 1  # B, H, W

        # B, 1, H, W
        # aff_label = aff_label.permute((2, 0, 1))
        aff_label = aff_label.reshape(B, np.prod(aff_label.shape[1:]))  # B, H*W
        aff_label = aff_label.to(dtype=torch.float, device=pred["aff"].device)

        # Get loss.
        aff_loss = cross_entropy_with_logits(pred["aff"], aff_label)
        if self.pred_depth:
            gt_depth = label[depth_label].to(self.device).unsqueeze(-1).float()
            depth_loss = self.model.depth_stream.loss(pred["depth_dist"], gt_depth)
        else:
            depth_loss = 0

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # Pixel distance error
            p0_pix, depth_sample, _, _ = self.model.predict(inp["img"], inp["lang_goal"])  # B, H, W
            # Depth error
            depth_error = 0
            if self.pred_depth:
                # Depth sample is unormalized in predict
                unormalized_depth = label["depth"].detach().cpu().numpy()
                depth_error = np.sum(np.abs(depth_sample - unormalized_depth))
            err = {"px_dist": np.sum(np.linalg.norm(p0 - p0_pix, axis=1)), "depth": depth_error}

        loss = self.loss_weights.aff * aff_loss
        loss += self.loss_weights.depth * depth_loss
        # loss = depth_loss

        info = {"aff_loss": aff_loss, "depth_loss": depth_loss}
        return loss, err, info

    def predict(self, obs, goal=None, info=None):
        """Run inference and return best pixel given visual observations.
        Args:
            obs(dict):
                img: np.ndarray (H, W, C)  values between 0-255 uint8
                lang_goal: str
            goal(str)
        Returns:
            (Tuple) nd.array: pixel position

        """
        # Get inputs
        # obs["img"] is  150, 200, 3
        img = np.expand_dims(obs["img"], 0)  # B, H, W, C
        img = tt(img, self.device)
        img = img.permute((0, 3, 1, 2))  # 1, 3, 150, 200

        img = self.pred_transforms(img)  # 1, 3, 224, 224

        lang_goal = goal if goal is not None else obs["lang_goal"]
        # Attention model forward pass.
        net_inp = {"img": img, "lang_goal": [lang_goal]}
        p0_pix, depth, uncertainty, logits = self.model.predict(img, lang_goal)
        p0_pix = p0_pix.squeeze()
        depth = depth.squeeze()

        err = None
        if info is not None:
            net_inp["img"] = img
            pred = self.forward(net_inp, softmax=False)
            _, err, _ = self.criterion(net_inp, pred, info, compute_err=True)

        # Get Aff mask
        affordance_heatmap_scale = 30
        pick_logits_disp = (logits * 255 * affordance_heatmap_scale).astype("uint8")
        pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)

        return {
            "softmax": pick_logits_disp,
            "pixel": (p0_pix[1], p0_pix[0]),
            "depth": depth,  # Net produces positive values
            "error": err,
        }

    def get_preds_viz(self, inp, pred, gt_depth=0, out_shape=(300, 300), waitkey=0):
        """
        Arguments:
            inp(dict):
                img(np.ndarray): between 0-1, shape= H, W, C
                lang_goal(list): language instruction
            pred(dict): output of self.predict(inp)
        """
        # frame = inp["img"][0].detach().cpu().numpy()
        # frame = (frame * 255).astype("uint8")
        # frame = np.transpose(frame, (1, 2, 0))
        # if frame.shape[-1] == 1:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        frame = inp["img"].copy()
        pred_img = frame.copy()

        cm = plt.get_cmap("viridis")
        heatmap = cm(pred["softmax"])[:, :, [0, 1, 2]] * 255
        heatmap = heatmap.astype("uint8")

        # frame = cv2.resize(frame, heatmap.shape[:2])
        heatmap = cv2.resize(heatmap, frame.shape[:2][::-1])
        heatmap = blend_imgs(frame.copy(), heatmap, alpha=0.8)

        pixel = pred["pixel"]
        # print(pred["error"], pred["pixel"], (x, y))

        pred_img = cv2.resize(pred_img, out_shape, interpolation=cv2.INTER_CUBIC)
        pixel = resize_pixel(pixel, self.in_shape[:2], pred_img.shape[:2])
        pred_img = cv2.drawMarker(
            pred_img,
            (pixel[0], pixel[1]),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        heatmap = cv2.resize(heatmap, out_shape, interpolation=cv2.INTER_CUBIC)
        pred_img = pred_img.astype(float) / 255

        out_img = np.concatenate([pred_img, heatmap], axis=1)

        # Prints the text.
        depth_est = float(pred["depth"])
        if torch.is_tensor(gt_depth):
            gt_depth = float(gt_depth.detach().cpu().numpy())
            text = "DepthErr: %.3f, Goal: %s" % (np.abs(depth_est - gt_depth), inp["lang_goal"])
        else:
            text = "DepthPred: %.3f, Goal: %s" % (-1 * depth_est, inp["lang_goal"])
        out_img = add_img_text(out_img, text)

        return out_img, {"pred_pixel": pred_img, "heatmap": heatmap}
