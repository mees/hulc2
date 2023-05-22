import logging
import os

import numpy as np
import tqdm

from hulc2.affordance.dataset_creation.core.utils import check_file, get_data
from hulc2.utils.utils import get_abspath
from hulc2.affordance.utils.data_utils import depth_img_from_uint16, quat_to_euler

log = logging.getLogger(__name__)


class DataReader:
    def __init__(self, cfg, *args, **kwargs):
        self.mode = cfg.labeling.mode
        self.play_data_dir = cfg.play_data_dir
        self.language_cfg = cfg.language
        self.files, self.end_ids, self.lang_ann = get_data(self.play_data_dir, self.mode, self.language_cfg)
        self.img_hist = {"static": [], "gripper": []}

    def after_loop(self, episode):
        pass

    def after_iteration(self, episode, ep_id, curr_folder):
        pass

    def on_episode_end(self, episode):
        pass

    def closed_gripper(self, dct):
        # Object is moving so not keep a history
        self.img_hist["gripper"] = []

    def closed_to_open(self, dct):
        pass

    def open_to_closed(self, dct):
        self.img_hist = {"static": [], "gripper": []}

    def get_contact_info(self, data):
        pass

    def get_robot_state(self, data):
        if self.mode != "simulation" and "processed" not in self.mode:
            proprio = data["robot_state"].item()
            # orn = p.getEulerFromQuaternion(proprio["tcp_orn"])
            orn = proprio["tcp_orn"]
            if len(orn) > 3:
                orn = quat_to_euler(orn)
            tcp_pos = proprio["tcp_pos"]
            robot_obs = np.array([*tcp_pos, *orn, proprio["gripper_opening_width"]])
        else:
            robot_obs = data["robot_obs"][:7]  # 3 pos, 3 euler angle
        return robot_obs

    def unprocess_obs(self, data):
        obs = {}
        for key in data.files:
            obs[key] = data[key]
            if "real_world" in self.mode and "depth" in key:
                obs[key] = depth_img_from_uint16(data[key])
            else:
                obs[key] = data[key]
        return obs

    def iterate(self):
        log.info("End ids")
        for id in self.end_ids:
            log.info(id)

        past_action = 1
        frame_idx = 0
        episode = 0
        last_obs = None

        # Iterate rendered_data
        files = self.files
        end_ids = self.end_ids
        if len(files) <= 0:
            log.info("No Files found in %s" % get_abspath(self.play_data_dir))
            return
        head, tail = os.path.split(files[0])
        ep_id = int(tail[:-4].split("_")[-1])

        # Multiple folders in real-robot data
        _, curr_folder = os.path.split(head)

        for idx, filename in enumerate(tqdm.tqdm(files)):
            data = check_file(filename)
            if data is None:
                continue  # Skip file
            elif "action" in data and data["action"].item() is None:
                continue

            if "robot_state" not in data and self.mode != "simulation" and "processed" not in self.mode:
                continue

            # Preprocessed data from lfp already transforms depth imgs to float
            # data = self.unprocess_obs(data)
            if idx < len(files) - 1:
                next_folder = os.path.split(os.path.split(files[idx + 1])[0])[-1]
            else:
                next_folder = curr_folder

            # Initialize img, mask, id
            head, tail = os.path.split(filename)
            img_id = tail[:-4]
            robot_obs = self.get_robot_state(data)
            for c in ["static", "gripper"]:
                self.img_hist[c].append(
                    (frame_idx, ep_id, "%s_%s" % (c, img_id), robot_obs, data["rgb_%s" % c], data["depth_%s" % c])
                )
            frame_idx += 1
            # Start of interaction
            if self.mode == "simulation" or "processed" in self.mode:
                ep_id = int(tail[:-4].split("_")[-1])
                end_of_ep = ep_id >= end_ids[0] + 1 and len(end_ids) > 1
                gripper_action = data["actions"][-1]  # -1 -> closed, 1 -> open
            else:
                ep_id = int(tail[:-4].split("_")[-1])
                end_of_ep = (len(end_ids) > 1 and ep_id >= end_ids[0]) or curr_folder != next_folder
                gripper_action = robot_obs[-1] > 0.077  # Open
                gripper_action = (data["action"].item()["motion"][-1] + 1) / 2

            if gripper_action <= 0 or end_of_ep:  # closed gripper
                # Get mask for static images
                # open -> closed
                if past_action == 1:
                    dct = {"robot_obs": robot_obs, "last_obs": last_obs, "frame_idx": frame_idx - 1, "data": data}
                    self.open_to_closed(dct)
                else:
                    # On gripper closed
                    # Was closed and remained closed
                    dct = {"robot_obs": robot_obs, "last_obs": last_obs, "data": data}
                    self.closed_gripper(dct)
            else:
                # closed -> open
                if past_action <= 0:
                    dct = {"robot_obs": robot_obs, "last_obs": last_obs, "frame_idx": frame_idx - 1}
                    self.closed_to_open(dct)
                    last_obs = robot_obs

            # Reset everything
            if end_of_ep:
                end_ids = end_ids[1:]
                past_action = 1  # Open
                curr_folder = next_folder
                self.on_episode_end(episode)
                episode += 1
            stop = self.after_iteration(episode, ep_id, curr_folder)
            if stop:
                break
            past_action = gripper_action
        return self.after_loop(episode)
