from pathlib import Path
import shutil
from typing import List

import hydra
import numpy as np
from tqdm import tqdm

from hulc2.utils.split_dataset import SplitData


def get_file_list(data_dir, extension=".npz", sort_list=False):
    """retrieve a list of files inside a folder"""
    dir_path = Path(data_dir)
    dir_path = dir_path.expanduser()
    assert dir_path.is_dir(), f"{data_dir} is not a valid dir path"
    file_list = []
    for x in dir_path.iterdir():
        if x.is_file() and extension in x.suffix:
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(get_file_list(x, extension))
    if sort_list:
        file_list = sorted(file_list, key=lambda file: file.name)
    return file_list


def get_ep_start_end_ids(data_dir):
    """retrieve a list of files inside a folder"""
    dir_path = Path(data_dir)
    dir_path = dir_path.expanduser()
    assert dir_path.is_dir(), f"{dir_path} is not a valid dir path"
    ep_start_end_ids_path = dir_path / "ep_start_end_ids.npy"
    ep_start_end_ids = np.load(ep_start_end_ids_path, allow_pickle=True)
    ep_start_end_ids = ep_start_end_ids[ep_start_end_ids[:, 0].argsort()]
    return ep_start_end_ids


def get_step_to_file(data_dir):
    """Create mapping from step to file index"""
    step_to_file = {}
    file_list = get_file_list(data_dir, extension=".npz")
    for file in file_list:
        step = int(file.stem.split("_")[-1])
        step_to_file[step] = file
    return step_to_file


def combine_dataset(input_dirs: List[str], out_dir):
    # Output dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate input data dirs
    cur_idx = 0
    new_ep_start_end_ids = []
    for data_dir in tqdm(input_dirs):
        ep_start_end_ids = get_ep_start_end_ids(data_dir)
        step_to_file = get_step_to_file(data_dir)
        for start_idx, end_idx in tqdm(ep_start_end_ids):
            new_start_idx = cur_idx
            for idx in tqdm(range(start_idx, end_idx + 1)):
                out_path = out_dir / f"episode_{cur_idx:07d}.npz"
                shutil.copyfile(step_to_file[idx], out_path)
                cur_idx += 1
            new_ep_start_end_ids.append([new_start_idx, cur_idx - 1])

    new_ep_start_end_ids = np.array(new_ep_start_end_ids)
    new_ep_start_end_ids_path = out_dir / "ep_start_end_ids.npy"
    np.save(new_ep_start_end_ids_path, new_ep_start_end_ids)

    new_ep_lens = new_ep_start_end_ids[:, 1] - new_ep_start_end_ids[:, 0] + 1
    new_ep_lens_path = out_dir / "ep_lens.npy"
    np.save(new_ep_lens_path, new_ep_lens)

    split_data = SplitData(data_dir=out_dir)
    split_data()


@hydra.main(config_path="../../conf", config_name="utils/combine_dataset")
def main(cfg):
    """Script to combine several teleop datasets into one"""
    combine_dataset(cfg.src_dirs, cfg.dest)


if __name__ == "__main__":
    main()
