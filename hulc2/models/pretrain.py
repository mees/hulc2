import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

logger = logging.getLogger(__name__)


class PretrainPlayLMP(pl.LightningModule):
    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_recognition: DictConfig,
        language_encoder: DictConfig,
        language_goal: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        distribution: DictConfig,
        val_instructions: DictConfig,
        clip_proj: Optional[DictConfig] = None,
    ):
        super(PretrainPlayLMP, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder, device=self.device)
        self.setup_input_sizes(self.perceptual_encoder, plan_recognition, distribution)
        self.dist = hydra.utils.instantiate(distribution)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.dist)
        self.lang_encoder = hydra.utils.instantiate(language_encoder)
        self.language_goal = hydra.utils.instantiate(language_goal)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.proj_vis_lang = hydra.utils.instantiate(clip_proj)
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.state_recons = False
        self.lang_recons = False
        self.lang_contrastive = False
        self.img_lang_matching_clip = True
        self.lang_clip_beta = 1
        self.save_hyperparameters()

        self.encoded_lang_train: Optional[torch.Tensor] = None
        self.encoded_lang_val: Optional[torch.Tensor] = None
        self.train_lang_emb: Optional[torch.Tensor] = None
        self.lang_data_val = None
        self.task_to_id: Optional[Dict] = None
        self.val_dataset = None
        self.train_lang_task_ids: Optional[np.ndarray] = None
        self.val_lang_emb: Optional[torch.Tensor] = None
        self.val_lang_task_ids: Optional[np.ndarray] = None
        self.val_instructions = val_instructions

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        plan_recognition,
        distribution,
    ):
        plan_recognition.in_features = perceptual_encoder.latent_size

        if distribution.dist == "discrete":
            plan_recognition.plan_features = distribution.class_size * distribution.category_size
        elif distribution.dist == "continuous":
            plan_recognition.plan_features = distribution.plan_features

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if "num_warmup_steps" in self.lr_scheduler:
            self.lr_scheduler.num_training_steps, self.lr_scheduler.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.lr_scheduler.num_training_steps,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.lr_scheduler.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.lr_scheduler.num_warmup_steps}")
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_fit_start(self) -> None:
        train_dataset = self.trainer.datamodule.train_datasets["lang"]  # type: ignore
        val_dataset = self.trainer.datamodule.val_datasets["lang"]  # type: ignore
        self.val_dataset = val_dataset
        lang_data_train = np.load(
            train_dataset.abs_datasets_dir / train_dataset.lang_folder / "auto_lang_ann.npy", allow_pickle=True
        ).item()
        self.lang_data_val = np.load(
            val_dataset.abs_datasets_dir / val_dataset.lang_folder / "auto_lang_ann.npy", allow_pickle=True
        ).item()
        lang_embeddings_val = np.load(
            val_dataset.abs_datasets_dir / val_dataset.lang_folder / "embeddings.npy", allow_pickle=True
        ).item()
        train_lang_instructions = list(set(lang_data_train["language"]["ann"]))
        train_lang_ids = [
            lang_data_train["language"]["ann"].index(instruction) for instruction in train_lang_instructions
        ]
        self.train_lang_emb = torch.from_numpy(lang_data_train["language"]["emb"][train_lang_ids]).cuda().squeeze()
        train_lang_tasks = list(np.array(lang_data_train["language"]["task"])[train_lang_ids])
        train_lang_task_ids = [list(set(train_lang_tasks)).index(task) for task in train_lang_tasks]

        self.task_to_id = {k: v for k, v in zip(set(train_lang_tasks), set(train_lang_task_ids))}
        self.train_lang_task_ids = np.array(train_lang_task_ids)
        val_lang_tasks = []
        val_lang_emb = []
        val_lang_instructions = []
        for val_task, val_instructions in self.val_instructions.items():
            val_lang_tasks.append(val_task)
            val_lang_emb.append(torch.from_numpy(lang_embeddings_val[val_task]["emb"][0]).cuda())
            val_lang_instructions.append(list(lang_embeddings_val[val_task]["ann"])[0])
        self.val_lang_emb = torch.cat(val_lang_emb)
        self.val_lang_task_ids = np.array([self.task_to_id[task] for task in val_lang_tasks])

    def training_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Dict,
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        batch: tuple(
           train_obs: Tensor,
           train_rgbs: tuple(Tensor, ),
           train_depths: tuple(Tensor, ),
           train_acts: Tensor)
        """
        proprio_loss, lang_pred_loss, lang_contrastive_loss, lang_clip_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        batch_size: Dict[str, int] = {}
        total_bs = 0
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            # if self.state_recons:
            #     proprio_loss += self.perceptual_encoder.state_reconstruction_loss()
            if "lang" in self.modality_scope:
                batch_size["aux_lang"] = torch.sum(dataset_batch["use_for_aux_lang_loss"]).detach()  # type:ignore
                latent_goal = self.language_goal(dataset_batch["lang"][dataset_batch["use_for_aux_lang_loss"]])
                pr_state, seq_feat = self.plan_recognition(perceptual_emb[dataset_batch["use_for_aux_lang_loss"]])
                # if self.lang_recons:
                #     lang_pred_loss += self.lang_regression_loss(seq_feat, dataset_batch["lang"], None)
                if self.img_lang_matching_clip:
                    lang_clip_loss += self.clip_loss(seq_feat, latent_goal, None)
                # if self.lang_contrastive:
                #     lang_contrastive_loss += self.contrastive_lang_loss(seq_feat, latent_goal, None)
            total_bs += dataset_batch["actions"].shape[0]

        if self.state_recons:
            proprio_loss = proprio_loss / len(batch)
            total_loss = total_loss + self.st_recon_beta * proprio_loss
            self.log(
                "train/pred_proprio",
                self.st_recon_beta * proprio_loss,
                on_step=False,
                on_epoch=True,
                batch_size=total_bs,
            )
        if self.lang_recons:
            total_loss = total_loss + self.lang_recon_beta * lang_pred_loss
            self.log(
                "train/pred_lang",
                self.lang_recon_beta * lang_pred_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
            )
        if self.lang_contrastive:
            total_loss = total_loss + self.lang_contrastive_beta * lang_contrastive_loss
            self.log(
                "train/lang_contrastive",
                self.lang_contrastive_beta * lang_contrastive_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
            )
        if self.img_lang_matching_clip:
            total_loss = total_loss + self.lang_clip_beta * lang_clip_loss
            self.log(
                "train/lang_clip_loss",
                self.lang_clip_beta * lang_clip_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
            )
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.encoded_lang_train = self.language_goal(self.train_lang_emb)
        self.encoded_lang_val = self.language_goal(self.val_lang_emb)

    def validation_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Dict,
        ],
        batch_idx: int,
    ) -> None:
        """
        batch: tuple(
           val_obs: Tensor,
           val_rgbs: tuple(Tensor, ),
           val_depths: tuple(Tensor, ),
           val_acts: Tensor)
        """
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            # if self.state_recons:
            #     state_recon_loss = self.perceptual_encoder.state_reconstruction_loss()
            #     self.log(f"val/proprio_loss_{self.modality_scope}", state_recon_loss, sync_dist=True)
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"][dataset_batch["use_for_aux_lang_loss"]])
                pr_state, seq_feat = self.plan_recognition(perceptual_emb[dataset_batch["use_for_aux_lang_loss"]])
                # if self.lang_recons:
                #     val_pred_lang_loss = self.lang_regression_loss(seq_feat, dataset_batch["lang"], None)
                #     self.log("val/lang_pred_loss", val_pred_lang_loss, sync_dist=True)
                if self.img_lang_matching_clip:
                    val_pred_clip_loss = self.clip_loss(seq_feat, latent_goal, None)
                    self.log("val/val_pred_clip_loss", val_pred_clip_loss, sync_dist=True)

                    train_gt_loss, val_gt_loss, train_sr, val_sr = self.clip_groundtruth(seq_feat, dataset_batch["idx"])
                    self.log("lang_gt/train_gt", train_gt_loss, sync_dist=True)
                    self.log("lang_gt/val_gt", val_gt_loss, sync_dist=True)
                    self.log("lang_gt/train_sr", train_sr, sync_dist=True)
                    self.log("lang_gt/val_sr", val_sr, sync_dist=True)
                # if self.lang_contrastive:
                #     val_pred_contrastive_loss = self.contrastive_lang_loss(seq_feat, latent_goal, None)
                #     self.log("val/lang_contrastive_loss", val_pred_contrastive_loss, sync_dist=True)

    def clip_groundtruth(self, seq_feat_vis, idx):
        gt_tasks = [
            self.task_to_id[self.lang_data_val["language"]["task"][self.val_dataset.lang_lookup[i]]] for i in idx
        ]

        train_score, train_sr = self._clip_groundtruth_loss(
            seq_feat_vis, self.encoded_lang_train, self.train_lang_task_ids, gt_tasks
        )
        val_score, val_sr = self._clip_groundtruth_loss(
            seq_feat_vis, self.encoded_lang_val, self.val_lang_task_ids, gt_tasks
        )

        return train_score, val_score, train_sr, val_sr

    def _clip_groundtruth_loss(self, seq_feat_vis, encoded_lang, task_ids, gt_tasks):

        image_features, lang_features = self.proj_vis_lang(seq_feat_vis, encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        scores = logits_per_image
        scores -= torch.min(scores, dim=1)[0].unsqueeze(1)
        scores /= torch.max(scores, dim=1)[0].unsqueeze(1) - torch.min(scores, dim=1)[0].unsqueeze(1)

        loss = []

        for score, gt_task in zip(scores, gt_tasks):
            positive_ids = np.where(task_ids == gt_task)[0]
            negative_ids = np.where(task_ids != gt_task)[0]
            loss.append(torch.sum(score[positive_ids]) - torch.sum(score[negative_ids]))

        loss = torch.mean(torch.stack(loss))

        sr = np.mean(task_ids[torch.argmax(scores, dim=1).cpu()] == np.array(gt_tasks))
        return loss, sr

    def clip_loss(self, seq_vis_feat, encoded_lang, use_for_aux_loss):
        assert self.img_lang_matching_clip is not None
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                return torch.tensor(0.0).to(self.device)
            seq_vis_feat = seq_vis_feat[use_for_aux_loss]
            encoded_lang = encoded_lang[use_for_aux_loss]
        image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
        loss_i = cross_entropy(logits_per_image, labels)
        loss_t = cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss

    def clip_inference(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            encoded_lang = self.language_goal(goal["lang"])
            _, seq_vis_feat = self.plan_recognition(perceptual_emb)
            image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            return logits_per_image, logits_per_text
