# from msilib import sequence
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from hulc2.models.hulc2 import Hulc2
from hulc2.utils.utils import format_sftp_path
from hulc2.env_wrappers.aff_lfp_real_world_wrapper import PandaLfpWrapper

from hulc2.affordance.models.language_encoders.sbert_lang_encoder import SBertLang
import logging
logger = logging.getLogger(__name__)


def load_dataset(cfg):
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)

    # we don't want to use shm dataset for evaluation
    # since we don't use the trainer during inference, manually set up data_module
    # vision_lang_folder = train_cfg.datamodule.datasets.vision_dataset.lang_folder
    # lang_lang_folder = train_cfg.datamodule.datasets.lang_dataset.lang_folder
    # train_cfg.datamodule.datasets = cfg.datamodule.datasets
    # train_cfg.datamodule.datasets.vision_dataset.lang_folder = vision_lang_folder
    # train_cfg.datamodule.datasets.lang_dataset.lang_folder = lang_lang_folder
    # print(train_cfg.aff_detection.streams.transforms.validation)
    # print(cfg.datamodule.transforms.val.keys())
    #
    # exit()
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


def evaluate_aff(model, env, max_ts, use_affordances):
    while 1:
        goal = input("Type an instruction \n")
        rollout(env, model, goal, use_affordances, ep_len=max_ts)


def rollout(env, model, goal, use_affordances=False, ep_len=340):
    move_robot = True
    target_orn = np.array([-3.11,  0.047,  0.027])

    rotate_orn = np.array([3.12, -0.022, 1.38])
    obs = env.get_obs()
    if use_affordances:
        target_pos, _move_flag = model.get_aff_pred(goal, obs, (500, 500))
        if move_robot and _move_flag:
            print("moving to: ", target_pos)
            print("moving to rot: ", target_orn)
            target_pos = np.clip(target_pos, [0.1,-0.45,0.1],[0.45,0.45,0.7])
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

@hydra.main(config_path="../../config", config_name="cfg_real_world")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    dataset, dataset_path = load_dataset(cfg)
    env = PandaLfpWrapper(env, dataset)
    use_affordances = cfg.train_folder is not None
    cfg.agent.aff_cfg.train_folder = cfg.train_folder
    model = hydra.utils.instantiate(cfg.agent,
                                    dataset_path=dataset_path,
                                    env=env,
                                    model_free=None,
                                    use_aff=use_affordances)
    print(f"Successfully loaded affordance model: {cfg.agent.aff_cfg.train_folder}/{cfg.agent.aff_cfg.model_name}")
    logger.info(f"Successfully loaded affordance model: {cfg.agent.aff_cfg.train_folder}/{cfg.agent.aff_cfg.model_name}")

    evaluate_aff(model, env,
                 use_affordances=use_affordances,
                 max_ts=cfg.max_timesteps)


if __name__ == "__main__":
    main()
