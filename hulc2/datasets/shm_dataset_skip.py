import logging
from typing import Any, Dict

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from hulc2.datasets.shm_dataset import ShmDataset
from hulc2.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)

logger = logging.getLogger(__name__)


class ShmDatasetSkip(ShmDataset):
    """
    Dataset Loader that uses a shared memory cache and skips episode frames.
    Currently two skipping strategies are implemented, 'random' and 'diff'.
    See the method documentation for more information.

    Args:
        effective_min_ws:   The min window_size after skipping frames < self.min_window_size
        effective_max_ws:   The max window_size after skipping frames < self.max_window_size
        skip_strategy:      Which heuristic to use for frame skipping
        pos_threshold:      Threshold for position (strategy 'diff')
        orn_threshold:      Threshold for orientation (strategy 'diff')
        max_skip_ratio:     Max ratio of skipped frames (strategy 'random')
    """

    def __init__(
        self,
        effective_min_ws: int,
        effective_max_ws: int,
        skip_strategy: str,
        *args: Any,
        pos_threshold: float = 0.99,
        orn_threshold: float = 0.08,
        min_skip_ratio: float = 0,
        max_skip_ratio: float = 0.3,
        **kwargs: Any,
    ):  # type: ignore
        super().__init__(*args, **kwargs)
        self.effective_min_ws = effective_min_ws
        self.effective_max_ws = effective_max_ws
        self.pos_threshold = pos_threshold
        self.orn_threshold = orn_threshold
        assert skip_strategy == "diff" or min_skip_ratio <= max_skip_ratio
        self.min_skip_ratio = min_skip_ratio
        self.max_skip_ratio = max_skip_ratio
        assert skip_strategy in ["random", "diff"]
        self.skip_strategy = skip_strategy

    def get_pad_size(self, sequence):
        """
        Args:
            sequence (dict): Sequence before padding is applied.
        Returns:
            int: Number of frames to pad.
        """
        return self.effective_max_ws - len(sequence["actions"])

    def random_frame_skip(self, episode):
        """
        Skip frames randomly.
        Percentage of skipped frames per episode is sampled uniformly between `self.min_skip_ratio` and
        `self.max_skip_ratio`.

        Args:
            episode (dict): Episode with length between `self.min_window_size` and `self.max_window_size`
        Return:
            np.ndarray: Indices of episode frames that are NOT skipped, length between `self.effective_min_ws` and
                `self.effective_max_ws`
        """
        ep_len = len(episode["rel_actions"])
        effective_ws = int(
            ((ep_len - self.min_window_size) / (self.max_window_size - self.min_window_size))
            * (self.effective_max_ws - self.effective_min_ws)
            + self.effective_min_ws
        )
        num_frame_skip = np.random.randint(int(ep_len * self.min_skip_ratio), int(ep_len * self.max_skip_ratio) + 1)
        window_pre_skip = np.random.randint(0, ep_len - effective_ws - num_frame_skip)
        episode_ids = np.sort(
            np.random.choice(
                np.arange(window_pre_skip, window_pre_skip + effective_ws + num_frame_skip), effective_ws, replace=False
            )
        )
        return episode_ids

    def diff_frame_skip(self, episode):
        """
        Skip frames with difference based heuristic.
        The possible skip indices are determined as follows:
        - Cosine similarity between the x,y,z part of the relative actions of two consecutive frames is smaller
          than `self.pos_threshold`
        AND
        - Euclidean distance between x,y,z euler angles of the relative actions of two consecutive frames is smaller
          than `self.orn_threshold`
        AND
        - Gripper action unchanged in the current and in the previous 4 frames
        From these indices we pick the actual skip indices with a ratio that is sampled between
        `self.min_skip_ratio` and 1.

        Args:
            episode (dict): Episode with length between `self.min_window_size` and `self.max_window_size`
        Return:
            np.ndarray: indices of episode frames that are NOT skipped, length between `self.effective_min_ws` and
                `self.effective_max_ws`
        """
        ep_len = len(episode["rel_actions"])
        effective_ws = int(
            ((ep_len - self.min_window_size) / (self.max_window_size - self.min_window_size))
            * (self.effective_max_ws - self.effective_min_ws)
            + self.effective_min_ws
        )
        a = episode["rel_actions"][:-1]
        b = episode["rel_actions"][1:]

        pos_cos_dist = cosine_similarity(torch.from_numpy(a)[:, :3], torch.from_numpy(b)[:, :3], dim=1).numpy()
        orn_diff = np.mean(np.abs(a[:, 3:6] - b[:, 3:6]), axis=1)
        # frame ids that can be skipped according to position threshold
        skip_ids_pos = np.where(pos_cos_dist > self.pos_threshold)[0] + 1
        # frame ids that can be skipped according to orientation threshold
        skip_ids_orn = np.where(orn_diff < self.orn_threshold)[0] + 1

        gripper_diff_ids = np.where(a[:, -1] != b[:, -1])[0] + 1
        # do not skip frames after gripper action changes
        gripper_diff_ids_extended = np.unique(
            np.tile(np.arange(4), len(gripper_diff_ids)) + np.repeat(gripper_diff_ids, 4)
        )
        # frame ids that can be skipped according to orientation threshold
        ids_gripper = np.setdiff1d(np.arange(len(episode["rel_actions"])), gripper_diff_ids_extended)

        # all indices that could potentially be skipped
        possible_skip_ids = np.intersect1d(np.intersect1d(skip_ids_pos, skip_ids_orn), ids_gripper)
        adjacent_ids = possible_skip_ids[np.where(possible_skip_ids[1:] == possible_skip_ids[:-1] + 1)]
        # do not skip 2 consecutive frames
        possible_skip_ids = np.setdiff1d(possible_skip_ids, np.union1d(adjacent_ids, adjacent_ids + 1)[1::2])
        # sample the amount of frames to skip
        max_num_skip = min(len(possible_skip_ids), ep_len - effective_ws)
        num_skip = np.random.randint(int(max_num_skip * self.min_skip_ratio), max_num_skip + 1)
        skip_ids = np.random.choice(possible_skip_ids, num_skip, replace=False)
        # skip frames
        episode_ids = np.delete(np.arange(ep_len), skip_ids)
        start_idx = np.random.randint(0, len(episode_ids) - effective_ws + 1)
        # shorten episode to desired window size (half of original window size)
        ids = episode_ids[start_idx : start_idx + effective_ws]
        assert len(ids) == effective_ws
        return ids

    def skip_episode_frames(self, episode):
        """
        Args:
            episode (dict): Episode pre skip, numpy format.
        Returns:
            dict: Episode after applying one of the frame skip strategies.
        """
        if self.skip_strategy == "random":
            ids = self.random_frame_skip(episode)
        else:
            ids = self.diff_frame_skip(episode)
        for key in list(episode.keys()):
            if key != "language":
                episode[key] = episode[key][ids]
        return episode

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.
        Args:
            idx: index of starting frame
            window_size: length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        episode = self.load_sequence_shm(idx, window_size)

        episode = self.skip_episode_frames(episode)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self.add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict
