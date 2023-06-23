import logging
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import pyhash
from termcolor import colored
import torch

from hulc2.evaluation.utils import format_sftp_path, join_vis_lang, LangEmbeddings
from hulc2.models.hulc2 import Hulc2

logger = logging.getLogger(__name__)


class PolicyManager:
    def __init__(self) -> None:
        pass

    def rollout(self, env, model, task_oracle, args, subtask, lang_embeddings, val_annotations, plans):
        if args.debug:
            print(f"{subtask} ", end="")
            time.sleep(0.5)
        obs = env.get_obs()
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]

        # get language goal embedding
        goal = lang_embeddings.get_lang_goal(lang_annotation)
        model.reset()
        start_info = env.get_info()

        plan, latent_goal = model.get_pp_plan_lang(obs, goal)
        plans[subtask].append((plan.cpu(), latent_goal.cpu()))

        for step in range(args.ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action)
            if args.debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                # time.sleep(0.1)
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if args.debug:
                    print(colored("success", "green"), end=" ")
                return True
        if args.debug:
            print(colored("fail", "red"), end=" ")
        return False

    def get_default_model_and_env(
        self, train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0, scene=None
    ):
        train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
        train_cfg_path = format_sftp_path(train_cfg_path)
        cfg = OmegaConf.load(train_cfg_path)
        cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace("calvin_models.", ""))
        lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder

        # we don't want to use shm dataset for evaluation
        datasets_cfg = cfg.datamodule.datasets
        for k in datasets_cfg.keys():
            datasets_cfg[k]["_target_"] = "hulc2.datasets.npz_dataset.NpzDataset"

        # since we don't use the trainer during inference, manually set up data_module
        cfg.datamodule.datasets = datasets_cfg
        cfg.datamodule.root_data_dir = dataset_path
        data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        dataset = dataloader.dataset.datasets["lang"]
        device = torch.device(f"cuda:{device_id}")

        if lang_embeddings is None:
            lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

        if env is None:
            rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "config/lfp/rollout/default.yaml")
            env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False, scene=scene)

        checkpoint = Path(train_folder + "/saved_models") / checkpoint
        checkpoint = format_sftp_path(checkpoint)
        print(f"Loading model from {checkpoint}")

        # Load model
        # Parameter added after model was trained
        if "spatial_softmax_temp" not in cfg.model.perceptual_encoder.rgb_static:
            perceptual_encoder = OmegaConf.to_container(cfg.model.perceptual_encoder)
            for k in perceptual_encoder.keys():
                v = perceptual_encoder[k]
                if (
                    isinstance(v, dict)
                    and "spatial_softmax_temp" not in v
                    and "_target_" in v
                    and v["_target_"] == "hulc2.models.perceptual_encoders.vision_network.VisionNetwork"
                ):
                    perceptual_encoder[k]["spatial_softmax_temp"] = 1.0
            perceptual_encoder = DictConfig(perceptual_encoder)
            model = Hulc2.load_from_checkpoint(checkpoint, perceptual_encoder=perceptual_encoder)
        else:
            model = Hulc2.load_from_checkpoint(checkpoint)
        model.freeze()

        if cfg.model.action_decoder.get("load_action_bounds", False):
            model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
        model = model.cuda(device)
        print(f"Successfully loaded policy model: {checkpoint}")
        logger.info(f"Loading policy model from {checkpoint}")
        return model, env, data_module, lang_embeddings
