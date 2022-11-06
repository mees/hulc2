import argparse
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import tqdm

TRAINING_DIR: str = "training"
VAL_DIR: str = "validation"


def slice_real_split(eps_list: np.ndarray, ep_start_end_ids: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    val_indx = eps_list[-idx:]
    train_indx = [ep for ep in list(range(0, len(eps_list))) if ep not in val_indx]
    val_ep_start_end_ids = ep_start_end_ids[val_indx, :]
    train_ep_start_end_ids = ep_start_end_ids[train_indx, :]
    return val_ep_start_end_ids, train_ep_start_end_ids


def main(input_params: Dict) -> None:
    dataset_root_str, last_k = (
        input_params["dataset_root"],
        input_params["last_K"],
    )
    abs_datasets_dir = Path(dataset_root_str)
    ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
    n_episodes = ep_start_end_ids.shape[0]
    ep_lens = np.arange(n_episodes)
    # n_episodes = len(list(abs_datasets_dir.glob("*")))
    # ep_lens = np.arange(n_episodes)
    # print(f"Found {n_episodes} episodes")
    # glob_generator = abs_datasets_dir.glob("**/*.npz")
    glob_generator = abs_datasets_dir.glob("*.npz")
    # # Remove camera calibration npz from iterable files
    file_names = [x for x in glob_generator if x.is_file() and "camera_info.npz" not in x.name]
    file_names.sort()
    # indices = [i for i, x in enumerate(file_names) if x.name == "frame_000000.npz"]
    # assert len(indices) == n_episodes
    # end_ids = np.array(indices[1:]) - 1
    # end_ids = np.append(end_ids, len(file_names) - 1)
    # start_ids = [0] + list(end_ids + 1)[:-1]
    # ep_start_end_ids = np.array(list(zip(start_ids, end_ids)))
    aux_naming_pattern = re.split(r"\d+", file_names[0].stem)
    n_digits = len(re.findall(r"\d+", file_names[0].stem)[0])
    split_data_path = abs_datasets_dir

    if last_k > 0:
        assert last_k < n_episodes
        splits = slice_real_split(ep_lens, ep_start_end_ids, last_k)
        (
            val_ep_start_end_ids,
            train_ep_start_end_ids,
        ) = splits
    elif last_k == 0:
        rand_perm = np.random.permutation(n_episodes)
        val_size = math.ceil(n_episodes * 0.1)
        splits = slice_real_split(rand_perm, ep_start_end_ids[rand_perm], val_size)
        (
            val_ep_start_end_ids,
            train_ep_start_end_ids,
        ) = splits
    else:
        raise NotImplementedError
    (split_data_path / TRAINING_DIR).mkdir(parents=True, exist_ok=True)
    (split_data_path / VAL_DIR).mkdir(parents=True, exist_ok=True)

    np.save(split_data_path / VAL_DIR / "ep_start_end_ids.npy", val_ep_start_end_ids)
    np.save(split_data_path / TRAINING_DIR / "ep_start_end_ids.npy", train_ep_start_end_ids)
    np.save(split_data_path / "all_ep_start_end_ids.npy", ep_start_end_ids)
    print("moving files to play_data/validation")
    print("-------")
    for x in val_ep_start_end_ids:
        range_ids = np.arange(x[0], x[1] + 1)  # to include end frame
        for frame_id in range_ids:
            filename = f"{aux_naming_pattern[0]}{frame_id:0{n_digits}d}.npz"
            file_names[frame_id].rename(split_data_path / VAL_DIR / filename)
    print("moving files to play_data/training")
    print("-------")
    for x in train_ep_start_end_ids:
        range_ids = np.arange(x[0], x[1] + 1)  # to include end frame
        for frame_id in range_ids:
            filename = f"{aux_naming_pattern[0]}{frame_id:0{n_digits}d}.npz"
            file_names[frame_id].rename(split_data_path / TRAINING_DIR / filename)
    print("finished creating splits!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data", help="directory where raw dataset is allocated")
    parser.add_argument(
        "--last_K",
        type=int,
        default="0",
        help="number of last episodes used for validation split, set to 0 for random splits",
    )
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
