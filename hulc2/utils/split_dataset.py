import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import tqdm
import yaml


def get_start_end_ids(abs_datasets_dir):
    json_file = abs_datasets_dir / "split.json"
    with open(json_file.as_posix()) as f:
        ep_start_end_ids = json.load(f)
    # convert list to numpy array
    for k, v in ep_start_end_ids.items():
        ep_start_end_ids[k] = np.asarray(v)
    return ep_start_end_ids


def get_split_sequences(start_end_ids, lang_data, asarray=False):
    split_lang_data = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }
    # Language annotated episodes(64 frames)
    # keys = [(start_i, end_i), ...]
    keys = np.array([idx for idx in lang_data["info"]["indx"]])
    for start, end in start_end_ids:
        # Check if language annotated episode frames(64) are part of frames selected for non-language annotated frames(play data episodes).
        # i.e. Check that both language annotated and non-language come frome the same data
        cond = np.logical_and(keys[:, 0] >= start, keys[:, 1] <= end)
        inside_ep = np.where(cond)[0]

        # If lang-annotated ep is inside selected play-data ep copy selected ep
        for i in inside_ep:
            split_lang_data["language"]["ann"].append(lang_data["language"]["ann"][i])
            split_lang_data["language"]["task"].append(lang_data["language"]["task"][i])
            split_lang_data["language"]["emb"].append(lang_data["language"]["emb"][i])
            split_lang_data["info"]["indx"].append(lang_data["info"]["indx"][i])

    if asarray:
        split_lang_data["language"]["ann"] = np.array(split_lang_data["language"]["ann"])
        split_lang_data["language"]["task"] = np.array(split_lang_data["language"]["task"])
        split_lang_data["language"]["emb"] = np.array(split_lang_data["language"]["emb"])
        split_lang_data["info"]["indx"] = np.array(split_lang_data["info"]["indx"])

    return split_lang_data


class SplitData:
    def __init__(
        self,
        data_dir: str,
        val_percentage: float = 0.1,
        max_episodes_in_validation: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"{str(self.data_dir)} is not a dir"
        assert val_percentage <= 1.0, "val percentage must be less than 1.0"
        ep_start_end_ids = np.load(self.data_dir / "ep_start_end_ids.npy")
        self.ep_start_end_ids = ep_start_end_ids[ep_start_end_ids[:, 0].argsort()]
        self.val_percentage = val_percentage
        self.max_episodes_in_validation = max_episodes_in_validation

    def split_every_episode(self):
        """Splits every episode using beginning of episode as training
        and end of episode as validation; such that the amount of validation
        frames corresponds to a desired percentage of the total data"""

        ep_lens = self.ep_start_end_ids[:, 1] - self.ep_start_end_ids[:, 0] + 1
        val_lens = (ep_lens * self.val_percentage).astype(np.uint8)

        split = {"training": [], "validation": []}
        for ep_idx in range(len(self.ep_start_end_ids)):
            start, end = self.ep_start_end_ids[ep_idx]
            split_frame = end - val_lens[ep_idx]
            split["training"].append([start.item(), (split_frame - 1).item()])
            split["validation"].append([(split_frame).item(), end.item()])

        out_filename = self.data_dir / "split.json"
        with open(out_filename, "w") as fp:
            json.dump(split, fp, indent=4)

    def find_best_split(self):
        """Splits some episodes to training and some episodes to validation;
        such that the len of frames in the validation episodes corresponds
        to a desired percentage of the total data"""

        ep_lens = self.ep_start_end_ids[:, 1] - self.ep_start_end_ids[:, 0] + 1
        if len(ep_lens) == 1:
            self.split_every_episode()
            return None

        total_len = sum(ep_lens)
        val_ideal_len = int(total_len * self.val_percentage)
        smallest_dif_to_ideal = float("inf")

        for i in range(1, self.max_episodes_in_validation + 1):
            # Iterate all possible groups
            for ind_comb in itertools.combinations(np.arange(len(ep_lens)), i):
                comb_dif = np.abs(val_ideal_len - np.sum(ep_lens[list(ind_comb)]))
                if comb_dif < smallest_dif_to_ideal:
                    smallest_dif_to_ideal = comb_dif
                    val_episodes = ind_comb
                    if comb_dif == 0:
                        break
        train_episodes = set(range(len(ep_lens))) - set(val_episodes)

        split = {"training": [], "validation": []}
        for main_key in split.keys():
            episodes = train_episodes if "train" in main_key else val_episodes
            for i, ep_idx in enumerate(episodes, 1):
                start, end = self.ep_start_end_ids[ep_idx]
                split[main_key].append([start.item(), end.item()])

        out_filename = self.data_dir / "split.json"
        with open(out_filename, "w") as fp:
            json.dump(split, fp, indent=4)

    def __call__(self):
        self.find_best_split()
        self.compute_statistics()

    def compute_statistics(self) -> None:
        dataset_root, save_format = self.data_dir, "npz"
        split = self.data_dir / "split.json"
        with open(split, "r") as fp:
            split_data = json.load(fp)

        split_data = split_data["training"]
        acc_robot_state = np.zeros((), "float64")
        acc_actions = np.zeros((), "float64")
        if dataset_root.is_dir():
            glob_generator = dataset_root.glob(f"*.{save_format}")
            file_names = []
            for x in tqdm.tqdm(glob_generator):
                if not x.is_file() or "camera_info" in x.stem:
                    continue

                ep_idx = int(x.stem.split("_")[-1])
                inside_split = np.logical_and(
                    ep_idx > np.array(split_data)[:, 0], ep_idx < np.array(split_data)[:, 1]
                ).any()
                if inside_split:
                    file_names.append(x)
                    episode = np.load(x.as_posix(), allow_pickle=True)
                    if "observations" in episode:
                        if acc_robot_state.shape == ():
                            acc_robot_state = episode["observations"]
                        else:
                            acc_robot_state = np.concatenate((acc_robot_state, episode["observations"]), axis=0)
                    if "actions" in episode:
                        if acc_actions.shape == ():
                            acc_actions = np.expand_dims(episode["actions"], axis=0)
                        else:
                            acc_actions = np.concatenate(
                                (acc_actions, np.expand_dims(episode["actions"], axis=0)), axis=0
                            )
                    else:
                        print("no actions found!!")
                        exit(0)
                    #  our play table environment
                    if "robot_obs" in episode:
                        if acc_robot_state.shape == ():
                            acc_robot_state = np.expand_dims(episode["robot_obs"], axis=0)
                        else:
                            acc_robot_state = np.concatenate(
                                (acc_robot_state, np.expand_dims(episode["robot_obs"], axis=0)), axis=0
                            )
            print(f"Computed training statistics from {dataset_root} with {len(file_names)} files")
            np.set_printoptions(precision=6, suppress=True)
            print(f"final robot obs shape {acc_robot_state.shape}")
            mean_robot_obs = np.mean(acc_robot_state, 0)
            std_robot_obs = np.std(acc_robot_state, 0)
            print(f"mean: {repr(mean_robot_obs)} and std: {repr(std_robot_obs)}")

            print(f"final robot actions shape {acc_actions.shape}")
            act_max_bounds = np.max(acc_actions, 0)
            act_min_bounds = np.min(acc_actions, 0)
            print(f"min action bounds: {repr(act_min_bounds)}")
            print(f"max action bounds: {repr(act_max_bounds)}")

            # Write file
            statistics = {
                "robot_obs": [
                    {
                        "_target_": "calvin_agent.utils.transforms.NormalizeVector",
                        "mean": list([x.item() for x in mean_robot_obs]),
                        "std": list([x.item() for x in std_robot_obs]),
                    }
                ],
                "action_min_bound": list([x.item() for x in act_min_bounds]),
                "action_max_bound": list([x.item() for x in act_max_bounds]),
            }
            statistics_confg = OmegaConf.create(statistics)
            statistics_file = os.path.join(self.data_dir, "statistics.yaml")
            OmegaConf.save(statistics_confg, statistics_file)
            # with open(statistics_file, "w") as fp:
            #     split_data = yaml.dump(statistics, fp, default_flow_style=True)


def main():
    parser = argparse.ArgumentParser(description="Parse slurm parameters and hydra config overrides")
    parser.add_argument("--dataset_root", type=str)
    args, unknownargs = parser.parse_known_args()

    split_data = SplitData(data_dir=args.dataset_root)
    split_data()


if __name__ == "__main__":
    main()
