import json
import logging
import os

import cv2
import hydra
import numpy as np
import pybullet as p
import shutil

from hulc2.affordance.base_detector import BaseDetector
from hulc2.affordance.dataset_creation.core.data_reader import DataReader
from hulc2.affordance.dataset_creation.core.utils import (
    create_data_ep_split,
    create_json_file,
    instantiate_env,
    save_dict_data,
)
import hulc2.affordance.utils.flowlib as flowlib
from hulc2.utils.img_utils import (
    get_px_after_crop_resize,
    resize_center,
)
from hulc2.utils.utils import get_abspath

log = logging.getLogger(__name__)


class DataLabeler(DataReader):
    def __init__(self, cfg, classifier=None, discovery_episodes=[], new_cfg=True, *args, **kwargs):
        super(DataLabeler, self).__init__(cfg, *args, **kwargs)
        self.output_dir = get_abspath(cfg.output_dir)
        self.create_output_dir(cfg)
        self.remove_blank_mask_instances = cfg.labeling.remove_blank_mask_instances
        self.save_viz = cfg.save_viz
        self.viz = cfg.viz
        self.mask_on_close = cfg.mask_on_close
        self.fixed_pt_del_radius = cfg.labeling.fixed_pt_del_radius
        self.mode = cfg.labeling.mode
        self.back_frames = cfg.labeling.back_frames  # [min, max]
        self.label_size = cfg.labeling.label_size
        self.output_size = {}
        for k in ["gripper", "static"]:
            _out_size = cfg.output_size[k]
            if isinstance(cfg.output_size[k], int):
                self.output_size[k] = (_out_size, _out_size)
            else:
                self.output_size[k] = tuple(_out_size)

        self.pixel_indices = {
            "gripper": np.indices(self.output_size["gripper"], dtype=np.float32).transpose(1, 2, 0),
            "static": np.indices(self.output_size["static"], dtype=np.float32).transpose(1, 2, 0),
        }
        _static, _gripper, _teleop_cfg = instantiate_env(cfg, self.mode, new_cfg=new_cfg)
        self.teleop_cfg = _teleop_cfg
        self.static_cam = _static
        self.gripper_cam = _gripper
        self.save_dict = {"gripper": {}, "static": {}, "grasps": []}
        self._fixed_points = []
        self.labeling = cfg.labeling
        self.single_split = cfg.output_cfg.single_split
        self.frames_before_saving = cfg.frames_before_saving
        self.classifier = classifier if classifier is not None else BaseDetector(cfg.task_detector)

        self.gripper_width_tresh = 0.02
        self.task_discovery_folders = discovery_episodes
        log.info("Writing to %s" % self.output_dir)

    def create_output_dir(self, cfg):
        play_data_dir = get_abspath(cfg.play_data_dir)
        output_dir = get_abspath(cfg.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # copy stard_end_ids.npy
        start_end_ids_file = "ep_start_end_ids.npy"
        # copy statistics.yaml
        statistics_file = "statistics.yaml"

        for filename in [start_end_ids_file, statistics_file]:
            src = os.path.join(play_data_dir, filename)
            dest = os.path.join(output_dir, filename)
            if os.path.isfile(src):
                shutil.copy(src, dest)

    @property
    def fixed_points(self):
        return self._fixed_points

    @fixed_points.setter
    def fixed_points(self, value):
        self._fixed_points = value

    def after_loop(self, episode=-1):
        self.save_data(episode)
        if self.single_split is not None:
            create_json_file(
                self.output_dir,
                self.single_split,
                self.labeling.min_labels,
                remove_blank_mask_instances=self.remove_blank_mask_instances,
                n_classes=self.classifier.n_classes,
            )
        else:
            create_data_ep_split(
                self.output_dir,
                self.labeling.split_by_episodes,
                self.labeling.min_labels,
                task_discovery_ep=self.task_discovery_folders,
                remove_blank_mask_instances=self.remove_blank_mask_instances,
                n_classes=self.classifier.n_classes,
            )
        # Add n_classes and orientation per class
        data = {"target_orn": None}
        orientations = {}
        for label, v in self.classifier.clusters.items():
            orientations[int(label)] = list(v["orn"])
        new_data = {"target_orn": orientations}
        output_path = os.path.join(self.output_dir, "episodes_split.json")
        with open(output_path, "r+") as outfile:
            data = json.load(outfile)
            data.update(new_data)
            outfile.seek(0)
            json.dump(data, outfile, indent=2)

    def after_iteration(self, episode, ep_id, curr_folder):
        if np.sum([len(v) for v in self.save_dict.values()]) > self.frames_before_saving:
            self.save_data(episode)
        # dont stop
        return False

    def on_episode_end(self, episode):
        self.save_data(episode)

    def closed_gripper(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_obs": last_obs,
                "data": data}
        """
        curr_robot_obs = dct["robot_obs"]
        last_obs = dct["last_obs"]
        if self.mask_on_close:
            self.label_gripper(self.img_hist["gripper"], curr_robot_obs, last_obs)
        super().closed_gripper(dct)

    def closed_to_open(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_obs": last_obs,
                "frame_idx": frame_idx}
        """
        new_pt = (dct["frame_idx"], dct["robot_obs"])
        self.fixed_points.append(new_pt)

    def open_to_closed(self, dct):
        """
        dct =  {"robot_obs": robot_obs,
                "last_obs": last_obs,
                "frame_idx": frame_idx,
                "data": data}
        """
        curr_robot_obs = dct["robot_obs"]
        last_obs = dct["last_obs"]
        frame_idx = dct["frame_idx"]

        self.label_gripper(self.img_hist["gripper"], curr_robot_obs, last_obs)
        self.label_static(self.img_hist["static"], curr_robot_obs)
        self.fixed_points = self.update_fixed_points(curr_robot_obs, frame_idx)
        super().open_to_closed(dct)

    def save_data(self, episode):
        for cam_str in ["static", "gripper"]:
            save_dir = os.path.join(self.output_dir, "episode_%02d" % episode)
            save_dict_data(self.save_dict[cam_str], save_dir, sub_dir="%s_cam" % cam_str, save_viz=self.save_viz)
        self.save_dict = {"gripper": {}, "static": {}, "grasps": []}

    def label_gripper(self, img_hist, curr_obs, last_obs):
        out_img_size = self.output_size["gripper"]
        save_dict = {}
        curr_pt, last_pt = curr_obs[:3], last_obs[:3]
        for idx, (fr_idx, ep_id, im_id, robot_obs, img, depth) in enumerate(img_hist):
            # Shape: [H x W x 2]
            H, W = out_img_size  # img.shape[:2]
            centers = []
            # Gripper width
            if robot_obs[-1] > self.gripper_width_tresh:
                for point in [curr_pt, last_pt]:
                    if point is not None:
                        # Center and directions in matrix convention (row, column)
                        center_px = self.get_gripper_label(robot_obs[:-1], point)
                        center_px = resize_center(center_px, img.shape[:2],out_img_size)
                        if np.all(center_px >= 0) and np.all(center_px < H):
                            centers.append([0, *center_px])
            else:
                centers = []

            # Visualize results
            img = cv2.resize(img, out_img_size[::-1])
            out_img = self.viz_imgs(img, centers, cam_str="gripper")
            centers = np.zeros((0, 3)) if len(centers) < 1 else np.stack(centers)
            save_dict[im_id] = {
                "frame": img,
                "centers": centers,
                "viz_out": out_img,
                "gripper_width": robot_obs[-1],
            }
        self.save_dict["gripper"].update(save_dict)

    def viz_imgs(self, rgb_img, centers,cam_str=""):
        """
        :param rgb_img(np.ndarray): H, W, C
        :param centers(np.ndarray): shape=(n_centers, 3). label, u, v = centers[i]
        """
        # Visualize results
        out_img = rgb_img
        for c in centers:
            label, v, u = c
            color = self.classifier.colors[label][:3]
            color = [int(c_item * 255) for c_item in color]
            out_img = cv2.drawMarker(
                np.array(out_img),
                (u, v),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        if self.viz:
            viz_size = (200, 200)
            viz_img = cv2.resize(out_img, viz_size[::-1])
            cv2.imshow("%s" % cam_str, viz_img[:, :, ::-1])
            cv2.waitKey(1)
        return out_img

    def label_static(self, static_hist, curr_robot_obs):
        cam = "static"
        back_min, back_max = self.back_frames
        out_img_size = self.output_size[cam]
        save_dict = {}
        for idx, (fr_idx, ep_id, im_id, robot_obs, img, depth) in enumerate(static_hist):
            # For static mask assume oclusion
            # until back_frames_min before
            centers = []
            centers_px = self.update_labels(fr_idx, img)

            # first create fp masks and place current(newest)
            # mask and optical flow on top
            label = self.classifier.predict(curr_robot_obs)
            if idx <= len(static_hist) - back_min and idx > len(static_hist) - back_max:
                # Get new grip
                pt = curr_robot_obs[:3]
                center_px = self.get_static_label(pt)
                center_px = resize_center(center_px, img.shape[:2], out_img_size)
                centers.append([label, *center_px])

            img = cv2.resize(img, out_img_size[::-1])
            centers += centers_px  # Concat to list
            if len(centers) > 0:
                centers = np.stack(centers)
            else:
                centers = np.zeros((0, 2))
            out_img = self.viz_imgs(img, centers, cam_str="static")

            save_dict[im_id] = {
                "frame": img,
                "centers": centers,
                "viz_out": out_img,
            }
        self.save_dict["static"].update(save_dict)

    def update_labels(self, frame_timestep, img):
        """
        :param mask(np.ndarray): N_classes, H, W
        """
        # Update masks with fixed_points
        centers = []
        for point_timestep, pt in self.fixed_points:
            # Only add point if it was fixed before seing img
            if frame_timestep >= point_timestep:
                label = self.classifier.predict(pt)

                center_px = self.get_static_label(pt[:3])
                center_px = resize_center(center_px, img.shape[:2], self.output_size["static"])
                centers.append([label, *center_px])
        return centers

    def update_fixed_points(self, new_point, current_frame_idx):
        x = []
        radius = self.fixed_pt_del_radius
        for frame_idx, pt in self.fixed_points:
            if np.linalg.norm(new_point[:3] - pt[:3]) > radius:
                x.append((frame_idx, pt))
        # x = [ p for (frame_idx, p) in fixed_points if
        # ( np.linalg.norm(new_point - p) > radius)]
        # # and current_frame_idx - frame_idx < 100 )
        return x

    def get_static_label(self, point):
        # Img history containes previus frames where gripper action was open
        # Point is the point in which the gripper closed for the 1st time
        # TCP in homogeneus coord.
        point = np.append(point, 1)

        # Project point to camera
        # x,y <- pixel coords
        tcp_x, tcp_y = self.static_cam.project(point)
        if "real_world" in self.mode:
            tcp_x, tcp_y = get_px_after_crop_resize(
                (tcp_x, tcp_y),
                self.static_cam.crop_coords,
                self.static_cam.resize_resolution,
            )
        return (tcp_y, tcp_x)  # matrix coord

    def preprocess_world_pt_gripper(self, robot_obs, point):
        pt, orn = robot_obs[:3], robot_obs[3:6]
        if "real_world" in self.mode:
            orn = p.getQuaternionFromEuler(orn)
            transform_matrix = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
            transform_matrix = np.vstack([transform_matrix, np.zeros(3)])
            tcp2global = np.hstack([transform_matrix, np.expand_dims(np.array([*pt, 1]), 0).T])
            global2tcp = np.linalg.inv(tcp2global)
            point = global2tcp @ np.array([*point, 1])
            point = point[:3]
        else:
            orn = p.getQuaternionFromEuler(orn)
            tcp2cam_pos, tcp2cam_orn = self.gripper_cam.tcp2cam_T
            # cam2tcp_pos = [0.1, 0, -0.1]
            # cam2tcp_orn = [0.430235, 0.4256151, 0.559869, 0.5659467]
            cam_pos, cam_orn = p.multiplyTransforms(pt, orn, tcp2cam_pos, tcp2cam_orn)

            # Create projection and view matrix
            cam_rot = p.getMatrixFromQuaternion(cam_orn)
            cam_rot = np.array(cam_rot).reshape(3, 3)
            cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]

            # Extrinsics change as robot moves
            self.gripper_cam.viewMatrix = p.computeViewMatrix(cam_pos, cam_pos + cam_rot_y, -cam_rot_z)
        return point
       
    def get_gripper_label(self, robot_obs, point):
        project_pt = self.preprocess_world_pt_gripper(robot_obs, point)
        if "real_world" in self.mode:
            project_pt = self.preprocess_world_pt_gripper(robot_obs, point)
            # Transform pt to homogeneus cords and project
            tcp_x, tcp_y = self.gripper_cam.project(project_pt)

            # Get img coords after resize
            tcp_x, tcp_y = get_px_after_crop_resize(
                (tcp_x, tcp_y),
                self.gripper_cam.crop_coords,
                self.gripper_cam.resize_resolution,
            )
        else:
            # Transform pt to homogeneus cords and project
            project_pt = np.append(project_pt, 1)
            tcp_x, tcp_y = self.gripper_cam.project(project_pt)
        return (tcp_y, tcp_x)


@hydra.main(config_path="../../../config", config_name="cfg_datacollection")
def main(cfg):
    labeler = DataLabeler(cfg, new_cfg=True)
    labeler.iterate()
    # labeler.after_loop()


if __name__ == "__main__":
    main()
