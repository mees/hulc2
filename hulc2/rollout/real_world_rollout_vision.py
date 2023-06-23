from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from robot_io.utils.utils import FpsController

from hulc2.evaluation.utils import imshow_tensor
from hulc2.models.encoders.language_network import SBert
from hulc2.models.hulc2 import Hulc2
from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from hulc2.wrappers.panda_lfp_wrapper import PandaLfpWrapper


def load_model(cfg):
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)
    # lang_folder = train_cfg.datamodule.datasets.lang_dataset.lang_folder
    # lang_model_string = lang_folder.split('_')[1]
    # lang_encoder = SBert(lang_model_string)
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
    model = Hulc2.load_from_checkpoint(checkpoint)
    model.freeze()
    if train_cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(train_cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda()
    print("Model loaded!")
    return model, dataset


def evaluate_policy_dataset(model, env, dataset):
    # i = 100000
    i = 0
    print("Press A / D to move through episodes, E / Q to skip 50 episodes.")
    print("Press P to replay recorded actions of the current episode.")
    print("Press O to run inference with the model, but use goal from episode.")
    print("Press L to run inference with the model and use your own language instruction.")

    while 1:
        episode = dataset[i]
        imshow_tensor("start", episode["rgb_obs"]["rgb_static"][0], wait=1, resize=True, unnormalize=False)
        imshow_tensor("start_gripper", episode["rgb_obs"]["rgb_gripper"][0], wait=1, resize=True, unnormalize=True)
        imshow_tensor("goal_gripper", episode["rgb_obs"]["rgb_gripper"][-1], wait=1, resize=True, unnormalize=True)
        k = imshow_tensor("goal", episode["rgb_obs"]["rgb_static"][-1], wait=0, resize=True, unnormalize=False)

        if k == ord("a"):
            i -= 1
            i = int(np.clip(i, 0, len(dataset)))
        if k == ord("d"):
            i += 1
            i = int(np.clip(i, 0, len(dataset)))
        if k == ord("q"):
            i -= 50
            i = int(np.clip(i, 0, len(dataset)))
        if k == ord("e"):
            i += 50
            i = int(np.clip(i, 0, len(dataset)))

        # replay episode with recorded actions
        if k == ord("p"):
            env.reset(episode=episode)
            for action in episode["actions"]:
                env.step(action)
                env.render("human")
        # inference with model, but goal from episode
        if k == ord("o"):
            # env.reset(episode=episode)
            # goal = {"lang": episode["lang"].unsqueeze(0).cuda()}
            goal = {
                "rgb_obs": {
                    "rgb_static": episode["rgb_obs"]["rgb_static"][-1].permute(1, 2, 0).cpu().numpy(),
                    "rgb_gripper": episode["rgb_obs"]["rgb_gripper"][-1].permute(1, 2, 0).cpu().numpy(),
                },
                "robot_obs": episode["state_info"]["robot_obs"][-1].cpu().numpy(),
            }
            # goal = {"vis": episode["lang"].unsqueeze(0).cuda()}
            rollout(env, model, goal)
        # inference with model language prompt
        if k == ord("l"):
            raise NotImplementedError


def rollout(env, model, goal, ep_len=500):
    # env.reset()
    model.reset()
    obs = env.get_obs()
    goal = env.transform_vis(goal)
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

    model, dataset = load_model(cfg)
    env = PandaLfpWrapper(env, dataset)

    evaluate_policy_dataset(model, env, dataset)


if __name__ == "__main__":
    main()
