import logging
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from sklearn.manifold import TSNE
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from vr_env.envs.play_table_env import get_env

from hulc2.evaluation.utils import imshow_tensor
from hulc2.utils.utils import format_sftp_path, get_last_checkpoint, timeit

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


def get_sequence(dataset, idx, seq_length):
    start_file_indx = dataset.episode_lookup[idx]
    end_file_indx = start_file_indx + seq_length

    episode = dataset.zip_sequence(start_file_indx, end_file_indx, idx)

    return episode


@timeit
def skip_frames(episode, pos_threshold, orn_threshold):
    a = episode["rel_actions"][:-1]
    b = episode["rel_actions"][1:]

    pos_cos_dist = cosine_similarity(torch.from_numpy(a)[:, :3], torch.from_numpy(b)[:, :3], dim=1).numpy()
    orn_diff = np.mean(np.abs(a[:, 3:6] - b[:, 3:6]), axis=1)

    skip_ids_pos = np.where(pos_cos_dist > pos_threshold)[0] + 1
    skip_ids_orn = np.where(orn_diff < orn_threshold)[0] + 1
    gripper_diff_ids = np.where(a[:, -1] != b[:, -1])[0] + 1
    gripper_diff_ids_extended = np.unique(np.tile(np.arange(4), len(gripper_diff_ids)) + np.repeat(gripper_diff_ids, 4))

    ids_gripper = np.setdiff1d(np.arange(len(episode["rel_actions"])), gripper_diff_ids_extended)

    print(f"pos: {len(skip_ids_pos)} orn: {len(skip_ids_orn)} gripper: {len(ids_gripper)}")
    ids = np.array(list(set(skip_ids_pos) & set(skip_ids_orn) & set(ids_gripper)))
    print(f"potential skip ids: {ids}")
    adjacent_ids = ids[np.where(ids[1:] == ids[:-1] + 1)]
    ids = np.setdiff1d(ids, np.union1d(adjacent_ids, adjacent_ids + 1)[1::2])
    # print(f"skip ids: {ids}")
    print(f"skip {len(ids)}")
    return ids


def show_img(episode, ids, j):
    img = cv2.resize(episode["rgb_static"][j][:, :, ::-1], (500, 500))
    if j in ids:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("win", img)


def visualize_skip_frames(dataset, env):
    i = 0
    shuffled_ids = np.random.permutation(np.arange(len(dataset)))
    seq_len = 64
    j = 0
    pos_threshold = 0.99
    orn_threshold = 0.08
    episode = get_sequence(dataset, i, seq_len)
    skip_ids = skip_frames(episode, pos_threshold, orn_threshold)
    show_img(episode, skip_ids, j)
    while 1:
        k = cv2.waitKey(0) % 256
        if k == ord("a"):
            j -= 1
            j = np.clip(j, 0, seq_len - 1)
            show_img(episode, skip_ids, j)
        if k == ord("d"):
            j += 1
            j = np.clip(j, 0, seq_len - 1)
            show_img(episode, skip_ids, j)
        if k == ord("q"):
            i -= 1
            j = 0
            i = np.clip(i, 0, len(dataset) - 1)
            episode = get_sequence(dataset, shuffled_ids[i], seq_len)
            skip_ids = skip_frames(episode, pos_threshold, orn_threshold)
            show_img(episode, skip_ids, j)
        if k == ord("e"):
            i += 1
            j = 0
            i = np.clip(i, 0, len(dataset) - 1)
            episode = get_sequence(dataset, shuffled_ids[i], seq_len)
            skip_ids = skip_frames(episode, pos_threshold, orn_threshold)
            show_img(episode, skip_ids, j)
        if k == ord("r"):
            env.reset(robot_obs=episode["robot_obs"][0], scene_obs=episode["scene_obs"][0])
            for action in episode["rel_actions"]:
                env.step(action.copy())
        if k == ord("t"):
            env.reset(robot_obs=episode["robot_obs"][0], scene_obs=episode["scene_obs"][0])
            prev_action = None
            for t, action in enumerate(episode["rel_actions"]):
                if t not in skip_ids:
                    env.step(action.copy())
                    prev_action = action.copy()
                else:
                    env.step(prev_action)


def load_data(cfg):
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.train_dataloader()["vis"].dataset

    env = get_env(dataset.abs_datasets_dir, show_gui=True)

    return dataset, env


@hydra.main(config_path="../../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset, env = load_data(cfg)

    visualize_skip_frames(dataset, env)


if __name__ == "__main__":
    main()
