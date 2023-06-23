import logging
import os

import cv2
import hydra
import numpy as np
import pybullet as p

from hulc2.affordance.dataset_creation.core.utils import create_data_ep_split, create_json_file
from hulc2.affordance.dataset_creation.data_labeler import DataLabeler
from hulc2.affordance.dataset_creation.find_norm_values import add_norm_values
from hulc2.utils.img_utils import add_img_text, resize_center

log = logging.getLogger(__name__)


class DataLabelerLang(DataLabeler):
    def __init__(self, cfg, classifier=None, discovery_episodes=[], *args, **kwargs):
        super(DataLabelerLang, self).__init__(cfg, *args, **kwargs)
        self.saved_static_frames = set()
        self._last_frame_task = None
        self._last_frame_lang_ann = None
        self.curr_task = {"static": None, "gripper": None}
        self._project_pt = None
        if self.mode == "simulation":
            self.env = hydra.utils.instantiate(self.teleop_cfg.env)

    def get_contact_info(self, data):
        if "real_world" in self.mode:
            return True
        else:
            obs = self.env.reset(robot_obs=data["robot_obs"], scene_obs=data["scene_obs"])
            static_reset = obs["rgb_obs"]["rgb_static"]
            static_file = data["rgb_static"]
            img = np.hstack([static_reset, static_file])
            contact_pts = np.array(p.getContactPoints())[:, 1]
            contact = (contact_pts == self.env.robot.robot_uid).any()
            # # Print text
            if self.viz:
                text_label = "contact" if contact else "no contact"
                img = add_img_text(img, text_label)
                cv2.imshow("reset/file img", img[:, :, ::-1])
                cv2.waitKey(1)
            return contact

    def open_to_closed(self, dct):
        """
        dct =  {"robot_obs": robot_obs,
                "last_obs": 3d pose where interaction ocurred,
                "frame_idx": frame of orig dataset,
                "data": data}
        """
        curr_robot_obs = dct["robot_obs"]
        last_obs = dct["last_obs"]
        frame_idx = dct["frame_idx"]
        contact = self.get_contact_info(dct["data"])

        if contact:
            self.save_dict["grasps"].append(frame_idx)
            self.label_gripper(self.img_hist["gripper"], curr_robot_obs, last_obs, contact)
            self.label_static(self.img_hist["static"], curr_robot_obs)
            self.img_hist = {"static": [], "gripper": []}

    def closed_gripper(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_obs": last_obs,
                "data": data} # np file of current robot obs
        """
        curr_robot_obs = dct["robot_obs"]
        last_obs = dct["last_obs"]

        contact = self.get_contact_info(dct["data"])
        if contact:
            min_len = 1 - self.mask_on_close
            if len(self.img_hist["static"]) > min_len:
                self.label_static(self.img_hist["static"], curr_robot_obs)
                self.img_hist["static"] = []
            if len(self.img_hist["static"]) > 1:
                frame_idx = self.img_hist["static"][-1][0]
                self.save_dict["grasps"].append(frame_idx)
            if len(self.img_hist["gripper"]) > min_len:
                self.label_gripper(self.img_hist["gripper"], curr_robot_obs, last_obs, contact)
            self.img_hist = {"static": [], "gripper": []}

    def closed_to_open(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_obs": last_obs,
                "frame_idx": frame_idx}
        """
        # self._project_pt = dct["last_obs"]
        # self.label_static(self.img_hist["static"], dct["robot_obs"])
        return

    def after_loop(self, episode=0):
        self.save_data(episode)
        min_labels = 1
        if self.single_split is not None:
            create_json_file(self.output_dir, self.single_split, min_labels, only_language=True)
        else:
            create_data_ep_split(
                self.output_dir,
                self.labeling.split_by_episodes,
                min_labels,
                task_discovery_ep=self.task_discovery_folders,
                only_language=True,
            )
        add_norm_values(os.getcwd(), self.output_dir, "episodes_split.json")

    def after_iteration(self, episode, ep_id, curr_folder):
        # log.info("Saving information... current episode %d " % episode)
        if np.sum([len(v) for v in self.save_dict.values()]) > self.frames_before_saving:
            self.save_data(episode)

        # log.info("Saved frames")
        return False

    def label_gripper(self, img_hist, curr_obs, last_obs, contact):
        cam = "gripper"
        out_img_size = self.output_size[cam]
        save_dict = {}
        self._last_task_centers = None
        curr_pt = curr_obs[:3]

        _ep_id = self.img_hist["gripper"][-1][1]
        # Get lang ann
        if _ep_id in self.lang_ann:
            # lang_ann has sets
            task = list(self.lang_ann[_ep_id].keys())[-1]
            lang_ann = list(self.lang_ann[_ep_id][task])
        else:
            lang_ann = []
            task = []
            self.curr_task[cam] = None

        for idx, (fr_idx, ep_id, im_id, robot_obs, img, depth) in enumerate(img_hist):
            # Shape: [H x W x 2]
            H, W = out_img_size  # img.shape[:2]
            centers = []

            # Gripper width
            pt_cam = self.get_pt_on_cam_frame(robot_obs, curr_pt, "gripper")
            if robot_obs[-1] > self.gripper_width_tresh or contact:
                # Center in matrix convention (row, column)
                center_px = self.get_gripper_label(robot_obs[:-1], curr_pt)
                center_px = resize_center(center_px, img.shape[:2], out_img_size)
                if np.all(center_px > 0) and np.all(center_px < H):
                    # Only one class, label as one
                    centers.append([0, *center_px])
            else:
                centers = []

            # Visualize results
            img = cv2.resize(img, out_img_size[::-1])
            caption = "" if len(lang_ann) == 0 else lang_ann[-1]
            out_img = self.viz_imgs(img, centers, caption, cam_str=cam)
            centers = np.zeros((0, 3)) if len(centers) < 1 else np.stack(centers)

            save_dict[im_id] = {
                "frame": img,
                "centers": centers,
                "lang_ann": lang_ann,
                "task": task,
                "viz_out": out_img,
                "gripper_width": robot_obs[-1],
                "tcp_pos_world_frame": curr_pt,
                "tcp_pos_cam_frame": pt_cam,
                "robot_obs": curr_obs,
            }
        self.save_dict[cam].update(save_dict)

    def label_static(self, static_hist, curr_robot_obs):
        cam = "static"
        back_min, back_max = self.back_frames
        out_img_size = self.output_size[cam]
        save_dict = {}
        H, W = self.output_size[cam]
        pt = curr_robot_obs[:3]
        self._project_pt = pt
        self._curr_robot_obs = curr_robot_obs

        _ep_id = self.img_hist["static"][-1][1]
        # Get lang ann
        if _ep_id in self.lang_ann:
            # lang_ann has sets
            task = list(self.lang_ann[_ep_id].keys())[-1]
            lang_ann = list(self.lang_ann[_ep_id][task])
        else:
            lang_ann = []
            task = ""
            self.curr_task[cam] = None

        pt_cam = self.get_pt_on_cam_frame(curr_robot_obs, self._project_pt, "static")
        for idx, (fr_idx, ep_id, im_id, robot_obs, img, depth) in enumerate(static_hist):
            if im_id in self.saved_static_frames:
                continue
            # For static mask assume oclusion
            # until back_frames_min before
            centers = []

            # first create fp masks and place current(newest)
            # mask and optical flow on top
            label = self.classifier.predict(curr_robot_obs)

            # Get centers
            if idx <= len(static_hist) - back_min and idx > len(static_hist) - back_max:
                # Get new grip
                center_px = self.get_static_label(self._project_pt)
                center_px = resize_center(center_px, img.shape[:2], out_img_size)
                if np.all(center_px > 0) and np.all(center_px < H):
                    # Only one class, label as one
                    centers.append([label, *center_px])

            img = cv2.resize(img, out_img_size[::-1])

            if len(centers) > 0:
                centers = np.stack(centers)
            else:
                centers = np.zeros((0, 2))

            caption = "" if len(lang_ann) == 0 else lang_ann[-1]
            out_img = self.viz_imgs(img, centers, caption, cam_str=cam)
            save_dict[im_id] = {
                "frame": img,
                "centers": centers,
                "lang_ann": lang_ann,
                "task": task,
                "viz_out": out_img,
                "tcp_pos_world_frame": self._project_pt,
                "tcp_pos_cam_frame": pt_cam,
                "robot_obs": self._curr_robot_obs,
            }
            self.saved_static_frames.add(im_id)
        self.save_dict[cam].update(save_dict)

    def get_world_pt(self, pixel, depth, cam, depth_shape=None):
        """
        return the world point corresponding to a pixel
        in a camera image given the depth value
        """
        v, u = pixel
        if "real_world" in self.mode:
            point_cam_frame = cam.deproject([u, v], depth, homogeneous=True, depth_shape=depth_shape)
            T_world_cam = cam.extrinsic_matrix
            if point_cam_frame is not None and len(point_cam_frame) > 0:
                world_pt = T_world_cam @ point_cam_frame
                world_pt = world_pt[:3]
        return world_pt

    def get_pt_on_cam_frame(self, robot_obs, world_pt, cam_name="static"):
        """
        Transforms point in world frame to camera frame
        """
        if self.mode == "simulation":
            cam = self.gripper_cam if cam_name == "gripper" else self.static_cam
            if cam_name == "gripper":
                # Re compute the view matrix of gripper camera
                world_pt = self.preprocess_world_pt_gripper(robot_obs, world_pt)

            # View matrix maps from world frame to cam frame
            view_m = np.array(cam.viewMatrix).reshape((4, 4)).T
            world_pt = np.append(world_pt, 1)
        else:
            cam = self.gripper_cam if cam_name == "gripper" else self.static_cam
            # world to cam frame
            if cam_name == "gripper":
                # From the world frame to tcp frame
                transformed_pt = self.preprocess_world_pt_gripper(robot_obs, world_pt)
            else:
                transformed_pt = world_pt
            # Inverse of extrinsic matrix. In the gripper camera, extrinsic matrix maps
            # from cam to tcp, so T_cam_world maps from tcp to cam (static transformation)
            view_m = np.array(cam.T_cam_world)
            world_pt = np.append(transformed_pt, 1)
        pt_cam = view_m @ world_pt
        pt_cam = pt_cam[:3]
        return pt_cam

    def viz_imgs(self, rgb_img, centers, caption="", cam_str=""):
        """
        :param rgb_img(np.ndarray): H, W, C
        :param centers(np.ndarray): shape=(n_centers, 3). label, u, v = centers[i]
        """
        # Visualize results
        out_img = rgb_img.copy()

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
            orig_shape = out_img.shape[:2]
            viz_size = (int(orig_shape[0] * 1.5), int(orig_shape[1] * 1.5))
            out_img = cv2.resize(out_img, viz_size[::-1])
            out_img = add_img_text(out_img, caption)
            cv2.imshow("%s" % cam_str, out_img[:, :, ::-1])
            cv2.waitKey(1)
        return out_img


@hydra.main(config_path="../../../conf/affordance", config_name="cfg_datacollection")
def main(cfg):
    labeler = DataLabelerLang(cfg)
    labeler.iterate()
    # labeler.after_loop()


if __name__ == "__main__":
    main()
