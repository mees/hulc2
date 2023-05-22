from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from hulc2.datasets.utils.episode_utils import process_rgb
from hulc2.evaluation.utils import imshow_tensor
from hulc2.models.hulc2 import Hulc2
from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from hulc2.env_wrappers.aff_lfp_real_world_wrapper import PandaLfpWrapper
import torchvision
from hulc2.models.encoders.language_network import SBert
from hulc2.affordance.models.language_encoders.sbert_lang_encoder import SBertLang
import logging

logger = logging.getLogger(__name__)


def instantiate_transforms(transforms):
    _transforms = {
        cam: [hydra.utils.instantiate(transform) for transform in transforms[cam]] for cam in transforms
    }
    _transforms = {key: torchvision.transforms.Compose(val) for key, val in _transforms.items()}
    return _transforms


def load_dataset(cfg):
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)
    cfg.datamodule.transforms.train.rgb_static = train_cfg.aff_detection.streams.transforms.training
    cfg.datamodule.transforms.val.rgb_static = train_cfg.aff_detection.streams.transforms.validation
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    print("data module prepare_data()")
    data_module.prepare_data()
    data_module.setup()
    print("data module setup complete")
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["vis"]

    return dataset, cfg.datamodule.root_data_dir


def load_model_free(cfg):
    train_cfg_path = Path(cfg.model_free.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    print("loading mdodel free cfg: ", train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)
    lang_folder = train_cfg.datamodule.datasets.lang_dataset.lang_folder
    lang_model_string = lang_folder.split("_")[1]
    print("loading sbert: ", lang_model_string)
    lang_encoder = SBert(lang_model_string)
    checkpoint = get_checkpoints_for_epochs(Path(cfg.model_free.train_folder), [cfg.model_free.checkpoint])[0]
    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    model = Hulc2.load_from_checkpoint(checkpoint)
    model.freeze()
    if train_cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(train_cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda()
    train_cfg["datamodule"]["transforms"]["val"].pop("robot_obs", None)
    _transforms = train_cfg.datamodule.transforms
    transforms = instantiate_transforms(_transforms["val"])

    return model, transforms, lang_encoder


def evaluate_combined(model_aff, model_free, lang_encoder, mf_transforms, env, use_affordances, use_model_free, max_ts):
    print(max_ts)
    while 1:
        goal = input("Type an instruction \n")
        rollout(env, model_aff, model_free, lang_encoder, mf_transforms, goal, use_affordances, use_model_free, max_ts)


def rollout(env, model_aff, model_free, lang_encoder, mf_transforms, goal, use_affordances, use_model_free, ep_len):
    move_robot = True

    if use_affordances:
        rollout_aff(env, model_aff, goal, move_robot)
    if use_model_free:
        rollout_model_free(env, model_free, lang_encoder, mf_transforms, goal, move_robot, ep_len)


def rollout_aff(env, model, goal, move_robot):
    target_orn = np.array([-3.11, 0.047, 0.027])
    rotate_orn = np.array([3.12, -0.022, 1.38])
    obs = env.get_obs()
    target_pos, _move_flag = model.get_aff_pred(goal, obs, (500, 500))
    if move_robot and _move_flag:
        print("moving to: ", target_pos)
        print("moving to rot: ", target_orn)
        target_pos = np.clip(target_pos, [0.1, -0.45, 0.1], [0.45, 0.45, 0.7])
        print("after clipping: ", target_pos)
        env.reset()
        print("going to final pos")
        if target_pos[1] < -0.35 and target_pos[2] < 0.35:
            print("increasing height for avoiding collision with box")
            target_pos[2] = 0.35
        if target_pos[1] > 0.4:
            env.reset(target_pos=target_pos, target_orn=rotate_orn)
        else:
            env.reset(target_pos=target_pos, target_orn=target_orn)
    else:
        print("move false!")


def transform_observation(obs, mf_transforms, observation_space_keys):
    rgb_obs = process_rgb(obs["rgb_obs"], observation_space_keys, mf_transforms)
    rgb_obs.update({"rgb_obs": {k: v.to("cuda").unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
    obs_dict = {**rgb_obs}
    obs_dict["depth_obs"] = {}
    obs_dict["robot_obs"] = {}
    obs_dict["robot_obs_raw"] = torch.from_numpy(obs["robot_obs"]).to("cuda")
    return obs_dict


def rollout_model_free(env, model, lang_encoder, mf_transforms, goal, move_robot, ep_len=300):
    obs = env.get_obs()
    # the observation keys are not loaded correctly in the wrong wrapper, fix this in future
    obs = transform_observation(obs, mf_transforms, env.observation_space_keys)
    model.replan_freq = 15
    lang_embedding = lang_encoder([goal])
    print("orig shape: ",lang_embedding.shape)
    goal = {"lang": lang_embedding.squeeze(0)}
    print(goal["lang"].shape)
    print(goal["lang"].device)
    print("ep len: ", ep_len)
    for step in range(ep_len):
        action = model.step(obs, goal)
        if move_robot:
            obs, _, _, _ = env.step(action)
        obs = transform_observation(obs, mf_transforms, env.observation_space_keys)
        imshow_tensor("rgb_static", obs["rgb_obs"]["rgb_static"], wait=1, resize=True, unnormalize=False)
        k = imshow_tensor("rgb_gripper", obs["rgb_obs"]["rgb_gripper"], wait=1, resize=True, unnormalize=True)
        # press ESC to stop rollout and return
        if k == 27:
            return


@hydra.main(config_path="../../config", config_name="cfg_real_world")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    dataset, dataset_path = load_dataset(cfg)
    env = PandaLfpWrapper(env, dataset)
    use_affordances = cfg.train_folder is not None
    cfg.agent.aff_cfg.train_folder = cfg.train_folder
    model_aff = hydra.utils.instantiate(cfg.agent,
                                        dataset_path=dataset_path,
                                        env=env,
                                        model_free=None,
                                        use_aff=use_affordances)
    print(f"Successfully loaded affordance model: {cfg.agent.aff_cfg.train_folder}/{cfg.agent.aff_cfg.model_name}")
    logger.info(
        f"Successfully loaded affordance model: {cfg.agent.aff_cfg.train_folder}/{cfg.agent.aff_cfg.model_name}")
    model_free, mf_transforms, lang_encoder = load_model_free(cfg)
    evaluate_combined(model_aff, model_free, lang_encoder, mf_transforms, env, use_affordances=use_affordances, use_model_free=True,
                      max_ts=cfg.max_timesteps)


if __name__ == "__main__":
    main()
