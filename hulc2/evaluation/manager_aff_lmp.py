import logging
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf
from termcolor import colored
import torch

from hulc2.evaluation.utils import format_sftp_path, get_env, join_vis_lang, LangEmbeddings

logger = logging.getLogger(__name__)
import cv2


class PolicyManager:
    def __init__(self, train_folder, checkpoint, debug=False, save_viz=False, use_affordances=True) -> None:
        self.debug = debug
        self.train_folder = train_folder
        self.checkpoint = checkpoint + ".ckpt"
        self.use_affordances = use_affordances
        self.save_viz = save_viz
        self.rollout_counter = 0

    def rollout(self, env, model, task_oracle, args, subtask, lang_embeddings, val_annotations, plans):
        if args.debug:
            print(f"{subtask} ", end="")
            time.sleep(0.5)
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]

        # get language goal embedding
        # goal = lang_embeddings.get_lang_goal(lang_annotation)

        # Now HULC model loads the weights of language network
        if model.model_free.lang_encoder is not None:
            goal = {"lang": [lang_annotation]}
        else:
            goal = lang_embeddings.get_lang_goal(lang_annotation)
        start_info = env.get_info()

        # Do not reset model if holding something
        if self.use_affordances:
            # width = env.robot.get_observation()[-1]["gripper_opening_width"]
            # if width > 0.055 or width< 0.01:
            model.reset(lang_annotation)
        else:
            # If no caption provided, wont use affordance to move to something
            model.model_free.reset()

        obs = env.get_obs()
        # Reset environment
        t_obs = model.rename_observation(obs)
        plan, latent_goal = model.model_free.get_pp_plan_lang(t_obs, goal)
        plans[subtask].append((plan.cpu(), latent_goal.cpu()))

        for step in range(args.ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action.copy())
            if args.debug:
                img = env.render(mode="rgb_array")
                cv2.imshow("Gripper cam", img["rgb_gripper"][:, :, ::-1])
                # cv2.imshow("Orig", img["rgb_static"][:, :, ::-1])
                # join_vis_lang(img["rgb_static"], lang_annotation)
                # time.sleep(0.1)
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if args.debug:
                    print(colored("success", "green"), end=" ")
                # if "grasped" in lang_annotation:
                #     model.save_sequence()
                return True
        if args.debug:
            print(colored("fail", "red"), end=" ")
        # if "grasped" in lang_annotation:
        #     model.save_sequence()
        return False

    def get_default_model_and_env(
        self,
        train_folder,
        dataset_path,
        checkpoint,
        env=None,
        lang_embeddings=None,
        device_id=0,
        scene=None,
        camera_conf=None,
    ):
        # Load Dataset
        train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
        train_cfg_path = format_sftp_path(train_cfg_path)
        cfg = OmegaConf.load(train_cfg_path)
        cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace("calvin_models.", ""))
        lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize(config_path="../../config")

        # we don't want to use shm dataset for evaluation
        datasets_cfg = cfg.datamodule.datasets
        for k in datasets_cfg.keys():
            datasets_cfg[k]["_target_"] = "hulc2.datasets.npz_dataset.NpzDataset"

        # since we don't use the trainer during inference, manually set up data_module
        cfg.datamodule.datasets = datasets_cfg
        cfg.datamodule.root_data_dir = dataset_path

        cfg.datamodule._target_ = cfg.datamodule._target_.replace("lfp", "hulc2")
        data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        dataset = dataloader.dataset.datasets["lang"]
        device = torch.device(f"cuda:{device_id}")

        if lang_embeddings is None:
            lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

        if env is None:
            env = get_env(
                dataset.abs_datasets_dir,
                show_gui=False,
                obs_space=dataset.observation_space,
                scene=scene,
                camera_conf=camera_conf,
            )
            rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "config/hulc2/rollout/aff_hulc2.yaml")
            env = hydra.utils.instantiate(rollout_cfg.env_cfg, env=env, device=device)

        checkpoint = format_sftp_path(checkpoint)
        print(f"Loading policy model from {train_folder}/{checkpoint}")
        logger.info(f"Loading policy model from {train_folder}/{checkpoint}")

        # Load model model-free + model-based + aff_model from cfg_high_level
        # overwrite model-free from checkpoint
        # Policy
        cfg = hydra.compose(config_name="cfg_high_level")
        cfg.agent.checkpoint.train_folder = train_folder
        cfg.agent.checkpoint.model_name = str(checkpoint)
        cfg.agent.dataset_path = dataset_path

        # Affordance
        if self.use_affordances:
            cfg.aff_detection.checkpoint.train_folder = self.train_folder
            cfg.aff_detection.checkpoint.model_name = self.checkpoint
        model = hydra.utils.instantiate(
            cfg.agent,
            viz_obs=self.debug,
            env=env,
            save_viz=self.save_viz,
            aff_cfg=cfg.aff_detection,
            use_aff=self.use_affordances,
        )
        print(f"Successfully loaded affordance model: {self.train_folder}/{self.checkpoint}")
        logger.info(f"Successfully loaded affordance model: {self.train_folder}/{self.checkpoint}")
        return model, env, data_module, lang_embeddings
