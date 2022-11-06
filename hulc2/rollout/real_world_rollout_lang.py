from pathlib import Path
import time

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from robot_io.utils.utils import FpsController

from hulc2.evaluation.utils import imshow_tensor
from hulc2.models.encoders.language_network import SBert
from hulc2.models.play_lmp import PlayLMP
from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from hulc2.wrappers.panda_lfp_wrapper import PandaLfpWrapper


def load_model(cfg):
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)
    lang_folder = train_cfg.datamodule.datasets.lang_dataset.lang_folder
    lang_model_string = lang_folder.split("_")[1]
    lang_encoder = SBert(lang_model_string)
    train_cfg["datamodule"]["datasets"].pop("lang_dataset", None)

    # we don't want to use shm dataset for evaluation
    # since we don't use the trainer during inference, manually set up data_module
    # vision_lang_folder = train_cfg.datamodule.datasets.vision_dataset.lang_folder
    # lang_lang_folder = train_cfg.datamodule.datasets.lang_dataset.lang_folder
    # train_cfg.datamodule.datasets = cfg.datamodule.datasets
    # train_cfg.datamodule.datasets.vision_dataset.lang_folder = vision_lang_folder
    # train_cfg.datamodule.datasets.lang_dataset.lang_folder = lang_lang_folder

    train_cfg.datamodule.root_data_dir = cfg.datamodule.root_data_dir
    data_module = hydra.utils.instantiate(train_cfg.datamodule, num_workers=0)
    print(train_cfg.datamodule)
    print("data module prepare_data()")
    data_module.prepare_data()
    data_module.setup()
    print("data module setup complete")
    dataloader = data_module.val_dataloader()
    # dataset = dataloader.dataset.datasets["lang"]
    dataset = dataloader.dataset.datasets["vis"]

    # reusing this function which is meant for getting multiple checkpoints
    checkpoint = get_checkpoints_for_epochs(Path(cfg.train_folder), [cfg.checkpoint])[0]
    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    model = PlayLMP.load_from_checkpoint(checkpoint)
    model.freeze()
    if train_cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(train_cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda()
    print("Model loaded!")
    return model, lang_encoder, dataset


def lang_rollout(model, lang_encoder, env, dataset):
    print("Type your instruction which the robot will try to follow")

    while 1:
        lang_input = [input("What should I do? \n")]
        lang_embedding = lang_encoder(lang_input)
        goal = {"lang": lang_embedding.squeeze(0)}
        print("sleeping 5 seconds...)")
        time.sleep(6)
        rollout(env, model, goal)


def rollout(env, model, goal, ep_len=500):
    # env.reset()
    model.reset()
    obs = env.get_obs()
    model.replan_freq = 15
    for step in range(ep_len):
        action = model.step(obs, goal)
        obs, _, _, _ = env.step(action)
        imshow_tensor("rgb_static", obs["rgb_obs"]["rgb_static"], wait=1, resize=True, unnormalize=False)
        k = imshow_tensor("rgb_gripper", obs["rgb_obs"]["rgb_gripper"], wait=1, resize=True, unnormalize=True)
        # press ESC to stop rollout and return
        if k == 27:
            return


@hydra.main(config_path="../../conf", config_name="inference_real")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    model, lang_encoder, dataset = load_model(cfg)
    env = PandaLfpWrapper(env, dataset)

    lang_rollout(model, lang_encoder, env, dataset)


if __name__ == "__main__":
    main()
