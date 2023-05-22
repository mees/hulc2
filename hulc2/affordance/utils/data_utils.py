import os
from copy import deepcopy
import hydra
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R


def split_by_percentage(root_dir, episodes_split, data_percent):
    new_episodes_split = deepcopy(episodes_split)

    # Change training split
    split ="training"
    # Get original data start end ids
    start_end_ids = os.path.join(root_dir, "%s/ep_start_end_ids.npy" % split)
    orig_start_end_ids = np.load(start_end_ids)

    # Split the dataset the same as it is split in learning_fom_play_repo
    new_start_end_ids = get_split_data(orig_start_end_ids, data_percent)
    for episode_dir, cam_frames in episodes_split[split].items():
        for cam, frames in cam_frames.items():
            valid_frames = []
            cam_frame_ids = np.array([int(f.split("_")[-1]) for f in frames])

            # Check valid frames
            if len(cam_frame_ids) > 0:
                for start, end in new_start_end_ids:
                    cond = np.logical_and(cam_frame_ids >= start, cam_frame_ids <= end)
                    inside_ep = np.where(cond)[0]
                    valid_frames.extend([i for i in inside_ep])

            # Replace
            new_episodes_split[split][episode_dir][cam] = list(np.array(frames)[valid_frames])
    return new_episodes_split

def get_split_data(play_start_end_ids, data_percent):
        start_end_ids = np.array(play_start_end_ids)
        cumsum = np.cumsum([e - s for s, e in play_start_end_ids])

        n_samples = int(cumsum[-1] * data_percent)
        max_idx = min(n_samples, cumsum[-1]) if n_samples > 0 else cumsum[-1]
        indices = [0]
        for i in range(len(cumsum) - 1):
            if cumsum[i] <= max_idx:
                indices.append(i + 1)

        # Valid play-data start_end_ids episodes
        start_end_ids = [start_end_ids[i] for i in indices]
        diff = cumsum[indices[-1]] - n_samples
        start_end_ids[-1][-1] = start_end_ids[-1][-1] - diff
        return np.array(start_end_ids)

def depth_img_from_uint16(depth_img, max_depth=4):
    depth_img[np.isnan(depth_img)] = 0
    return (depth_img.astype("float") / (2 ** 16 - 1)) * max_depth


def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler("xyz", euler_angles).as_quat()


def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")