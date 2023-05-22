"""Main training script."""

import os
from torch.utils.data import DataLoader

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import logging
from omegaconf import OmegaConf
from hulc2.utils.utils import get_abspath
from hulc2.affordance.models.depth.depth_module import DepthModule

def print_cfg(cfg):
    print_cfg = OmegaConf.to_container(cfg)
    print_cfg.pop("paths")
    print_cfg.pop("trainer")
    return OmegaConf.create(print_cfg)


@hydra.main(config_path="./config", config_name='train_affordance')
def main(cfg):
    # Log main config for debug
    print("Running configuration: ", cfg)
    logger = logging.getLogger(__name__)
    logger.info("Running configuration: %s", OmegaConf.to_yaml(print_cfg(cfg)))

    # Logger
    _name = "%s_%s" % (cfg.run_name, cfg.aff_detection.model.depth_dist)
    if cfg.aff_detection.model.cfg.normalize_depth:
        _name += "_normed"
    cfg.wandb.logger.name = _name
    wandb_logger = WandbLogger(**cfg.wandb.logger)

    # Checkpoint saver
    checkpoint_dir = get_abspath(cfg.checkpoint.path)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_path, cfg.checkpoint.model_name)
    last_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) and cfg.load_from_last_ckpt else None

    # Initialize checkpoints
    callbacks = []
    for name, saver_cfg in cfg.wandb.saver.items():
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=name,
            **saver_cfg
        )
        callbacks.append(checkpoint_callback)
    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        **cfg.trainer
    )

    # Dataloaders
    train = hydra.utils.instantiate(cfg.aff_detection.dataset, split="training", log=logger)
    val = hydra.utils.instantiate(cfg.aff_detection.dataset, split="validation", log=logger)
    logger.info("train_data {}".format(train.__len__()))
    logger.info("val_data {}".format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **cfg.dataloader)
    val_loader = DataLoader(val, **cfg.dataloader)
    logger.info("train minibatches {}".format(len(train_loader)))
    logger.info("val minibatches {}".format(len(val_loader)))

    # Initialize agent
    in_shape = train.out_shape[::-1]  # H, W, C
    model = DepthModule(cfg.aff_detection.model.cfg,
                        in_shape=in_shape,
                        depth_dist=cfg.aff_detection.model.depth_dist,
                        depth_norm_values=val.depth_norm_values)

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        model = model.load_from_checkpoint(last_checkpoint).cuda()
        logger.info("Model successfully loaded: %s" % last_checkpoint)

    # Main training loop
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    main()
