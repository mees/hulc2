import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from robot_io.utils.utils import depth_img_from_uint16, quat_to_euler, to_relative_all_frames
from tqdm import tqdm

N_DIGITS = 6
MAX_REL_POS = 0.02
MAX_REL_ORN = 0.05


def load_data(data):
    robot_state = data["robot_state"][()]
    action = data["action"][()]["motion"]

    tcp_pos = robot_state["tcp_pos"]
    tcp_orn = quat_to_euler(robot_state["tcp_orn"])
    gripper_width = robot_state["gripper_opening_width"]
    joint_positions = robot_state["joint_positions"]

    gripper_action = action[-1]  # int(gripper_width >= 0.078)
    # todo: include force torque
    robot_obs = np.concatenate([tcp_pos, tcp_orn, [gripper_width], joint_positions, [gripper_action]])

    return tcp_pos, tcp_orn, action, robot_obs


def compute_rel_action(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn, gripper_action):
    rel_pos_orn_dct = to_relative_all_frames(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn)

    rel_pos_orn_dct_new = {}
    for frame, (rel_pos, rel_orn) in rel_pos_orn_dct.items():
        # clipped_rel_pos = np.clip(rel_pos, -MAX_REL_POS, MAX_REL_POS) / MAX_REL_POS
        # clipped_rel_orn = np.clip(rel_orn, -MAX_REL_ORN, MAX_REL_ORN) / MAX_REL_ORN
        clipped_rel_pos, clipped_rel_orn = rel_pos / MAX_REL_POS, rel_orn / MAX_REL_ORN
        rel_action = np.concatenate([clipped_rel_pos, clipped_rel_orn, [gripper_action]])
        rel_pos_orn_dct_new[frame] = rel_action

    return rel_pos_orn_dct_new


def process_data(recording_dir, i):
    data_prev = get_frame(recording_dir, i - 1)
    data_cur = get_frame(recording_dir, i)

    tcp_pos, tcp_orn, curr_action, robot_obs = load_data(data_cur)
    past_tcp_pos, past_tcp_orn, past_action, _ = load_data(data_prev)

    curr_gripper_action = curr_action[-1]

    tcp_pos_action = curr_action[0]
    tcp_orn_action = curr_action[1]

    past_tcp_pos_action = past_action[0]
    past_tcp_orn_action = past_action[1]

    rel_action = compute_rel_action(
        past_tcp_pos_action, past_tcp_orn_action, tcp_pos_action, tcp_orn_action, curr_gripper_action
    )
    # rel_action = compute_rel_action(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn, curr_gripper_action)
    if np.linalg.norm(rel_action["world_frame"][3:6]) > 1.5 or np.linalg.norm(rel_action["world_frame"][:3]) > 1.25:

        if np.linalg.norm(rel_action["world_frame"][:3]) > 1.25:
            print("Displacement - out of bounds", i, rel_action["world_frame"])
        else:
            print("Rotation - out of bounds", i, rel_action["world_frame"])

        data_next = get_frame(recording_dir, i + 1)
        next_tcp_pos, next_tcp_orn, _, _ = load_data(data_next)
        rel_action = compute_rel_action(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn, curr_gripper_action)
        if np.linalg.norm(rel_action["world_frame"][3:6]) > 1.5 or np.linalg.norm(rel_action["world_frame"][:3]) > 1.25:
            print("Still out of bounds", i, rel_action["world_frame"])
        else:
            print("Solved, in bounds", i, rel_action["world_frame"])
    curr_action = np.concatenate([curr_action[0], quat_to_euler(curr_action[1]), [curr_action[2]]])
    save_data = {
        "actions": curr_action,
        "rel_actions_world": rel_action["world_frame"],
        "rel_actions_gripper": rel_action["gripper_frame"],
        "robot_obs": robot_obs,
    }
    return save_data


def get_frame(path, i):
    filename = Path(path) / f"frame_{i:0{N_DIGITS}d}.npz"
    return np.load(filename, allow_pickle=True)


def listdirs(rootdir):
    list_dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            if (path / "ep_start_end_ids.npy").is_file():
                list_dirs.append(path)
            else:
                result = listdirs(path)
                if result:
                    list_dirs.append(result)
    if list_dirs:
        return list_dirs


def analyze_dataset(recording_dirs):
    ep_start_end_ids_all = []
    min_max_action_values = {
        "rel_actions_world": [np.ones(7) * np.inf, -np.ones(7) * np.inf],
        "rel_actions_gripper": [np.ones(7) * np.inf, -np.ones(7) * np.inf],
    }
    indcs_to_check = [18190, 18266, 76695]

    new_end_idx = 0
    for recording_dir in tqdm(recording_dirs):
        ep_start_end_ids = np.sort(np.load(recording_dir / "ep_start_end_ids.npy"))

        for start_idx, end_idx in tqdm(ep_start_end_ids, leave=False):
            new_start_idx = new_end_idx
            for i in range(start_idx + 1, end_idx + 1):
                save_data = process_data(recording_dir, i)
                for key in min_max_action_values.keys():
                    key_min, key_max = min_max_action_values[key]
                    min_max_action_values[key][0] = np.minimum(save_data[key], key_min)
                    min_max_action_values[key][1] = np.maximum(save_data[key], key_max)
                new_end_idx += 1

            ep_start_end_ids_all.append((new_start_idx, new_end_idx - 1))

        for k, v in min_max_action_values.items():
            print(k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="Root directory of recordings")
    args = parser.parse_args()

    list_recording_dirs = listdirs(args.dataset_root)
    print(args.dataset_root)
    print(list_recording_dirs)
    # now flatten list of lists
    recording_dirs = [item for sublist in list_recording_dirs for item in sublist]
    print("Found following subfolders containing recordings: ", recording_dirs)

    analyze_dataset(recording_dirs)


if __name__ == "__main__":
    main()
