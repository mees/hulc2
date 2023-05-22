import numpy as np
import cv2
import logging
from hulc2.utils.utils import add_img_text, resize_pixel, get_aff_model
from hulc2.evaluation.utils import join_vis_lang
from pathlib import Path
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, env, offset, aff_cfg, viz_obs=False, save_viz=False, *args, **kwargs):
        # For debugging
        self.curr_caption = ""
        #
        self._env = env
        self.viz_obs = viz_obs
        self.env.reset()
        _info = self.env.robot.get_observation()[-1]
        self.origin = np.array([-0.25, -0.3, 0.6])  #  np.array(_info["tcp_pos"])
        self.target_orn = np.array([np.pi, 0, np.pi/2]) # np.array(_info["tcp_orn"])
        self.logger = logging.getLogger(__name__)
        self.device = self.env.device
        _point_detector, _ = get_aff_model(**aff_cfg.checkpoint)

        # Move to device
        self.point_detector = _point_detector.to(self.device)
        self.point_detector.model = self.point_detector.model.to(self.device)

        self.model_free = None
        self.offset = np.array([*offset, 1])

        # Not save first
        self.save_viz = False
        self.reset_position()

        # To save images
        self.save_viz=save_viz
        # save_directory = Path(__file__).parents[2].resolve()
        save_directory = Path(os.path.expanduser("~/logs")).resolve()
        
        model = "ours" if kwargs["use_aff"] else "baseline"
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + model
        save_directory = save_directory / "evaluation_rollouts" / filename
        self.save_dir = {"parent": save_directory,
                         "sequence_counter": 0,
                         "rollout_counter": 0,
                         "step_counter": 0}
        self.sequence_data = {}
        logger.info(f"Initialized PlayLMPAgent for device {self.device}")

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

    def load_model_free(self, train_folder, model_name, **kwargs):
        self.logger.info("Base Agent has no policy implemented. Step will be a waiting period...")

    def encode(self, goal):
        return goal

    def step(self, obs, goal):
        '''
            obs(dict):  Observation comming from the environment
                - rgb_obs:
                - depth_obs:
                - robot_obs:
            goal(dict): Either a language or image goal. If language contains key "lang" which is used by the policy to make the prediction, otherwise the goal is provided in the form of an image.
                - lang: caption used to contidion the policy
                Only used if "lang" not in dictionary...
                - depth_obs: 
                - rgb_obs: 
        '''
        action = np.zeros(7)
        action[-1] = 1  # gripper
        return action

    def reset_position(self):
        return self.move_to(self.origin,
                            self.target_orn,
                            1)

    def viz_img(rgb_img, center_pixel, output_size=(300, 300), lang_goal=""):
        old_shape = rgb_img.shape[:2]
        pixel = resize_pixel(center_pixel, old_shape, output_size)
        out_img = cv2.drawMarker(
                rgb_img.copy(),
                (pixel[0], pixel[1]),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        out_img = cv2.resize(out_img, output_size)
        if(lang_goal != ""):
            # Prints the text.
            out_img = add_img_text(out_img, lang_goal)

        cv2.imshow("orig img",out_img[:, :, ::-1])
        cv2.waitKey(1)

    def move_to(self, target_pos, target_orn=None, gripper_action=None):
        '''
            Move to the specified location in world coordinates
        '''
        curr_info = self.env.robot.get_observation()[-1]
        if(target_orn is None):
            # target_orn = np.array(curr_info["tcp_orn"])
            target_orn = self.target_orn.copy()
        if(gripper_action is None):
            gripper_action = curr_info["gripper_action"]

        
        tcp_pos = np.array(curr_info["tcp_pos"])
        # Move in -z gripper
        # robot_orn = curr_info["tcp_orn"]
        # tcp_mat = pos_orn_to_matrix(tcp_pos, robot_orn)
        # offset_global_frame = tcp_mat @ np.array([0.0, 0.0, -0.08, 1.0])
        # reach_target = offset_global_frame[:3]

        tcp_up = tcp_pos[-1] + 0.07
        move_z = max(tcp_up, target_pos[-1])
        move_z = min(move_z, 0.7)

        # Move in +z
        reach_target = [*tcp_pos[:2], tcp_pos[-1] + 0.03]
        a = [reach_target.copy(), target_orn, gripper_action]
        tcp_pos, _ = self.move_to_pos(tcp_pos, a)

        # Move in -y
        reach_target = [tcp_pos[0], tcp_pos[1]-0.03, move_z]
        a = [reach_target.copy(), target_orn, gripper_action]
        tcp_pos, _ = self.move_to_pos(tcp_pos, a)

        # Move in xy
        reach_target = [*target_pos[:2], tcp_pos[-1]]
        a = [reach_target.copy(), target_orn, gripper_action]
        tcp_pos, _ = self.move_to_pos(tcp_pos, a)

        # Move to target
        a = [target_pos.copy(), target_orn, gripper_action]
        _, transition = self.move_to_pos(tcp_pos, a)
        return transition

    def save_sequence_txt(self, filename, data):
        output_dir = os.path.join(self.save_dir["parent"],
                              "seq_%03d" % self.save_dir["sequence_counter"])
        filedir = os.path.join(output_dir, "%s.txt" % filename)
        if filedir in self.sequence_data:
            self.sequence_data[filedir].append(data)
        else:
            if isinstance(data, list):
                self.sequence_data[filedir] = data
            else:
                self.sequence_data[filedir] = [data]

    def save_img(self, img, folder="./", name="img"):
        outdir = os.path.join(self.save_dir["parent"],
                              "seq_%03d" % self.save_dir["sequence_counter"])
        outdir = os.path.join(outdir,
                              "task_%02d" % self.save_dir["rollout_counter"])
        outdir = os.path.join(outdir, folder)
        output_file = os.path.join(outdir, "%s_%04d" % (name, self.save_dir["step_counter"]))
        self.sequence_data["%s.png" % output_file] = img[:, :, ::-1]
        return output_file

    def save_sequence(self):
        for filename, data in self.sequence_data.items():
            dirname = os.path.dirname(filename)
            os.makedirs(dirname, exist_ok=True)
            if filename.split('.')[-1] == "txt":
                with open(filename, 'w') as f:
                    for line in data:
                        f.write(line)
                        f.write('\n')
            else:
                cv2.imwrite(filename, data)
        self.sequence_data = {}

    def move_to_pos(self, tcp_pos, action):
        last_pos = action[0].copy()
        target_pos = action[0]
        target_orn = action[1]

        # Get env info
        ns = self.env.get_obs()
        info = self.env.get_info()
        self.img_save_viz_mb(ns)

        # When robot is moving and far from target
        curr_pos = np.array(info["robot_info"]["tcp_pos"])
        curr_orn = np.array(info["robot_info"]["tcp_orn"])
        
        kp, kd = 0.08, 0.05
        derivative = 0
        error = (target_pos - curr_pos)
        angle_diff = curr_orn - target_orn
        max_ts = 200
        ts = 0
        while (np.linalg.norm(error) > 0.01  # Large error
               and (np.linalg.norm(curr_pos - last_pos) > 0.0005  # Moving
                    or (np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) > 0.01).any()  # Large angle diff
                   )
               and ts < max_ts):
            last_pos = curr_pos
            rel_pos = error * kp + derivative * kd
            derivative = error

            action = [curr_pos + rel_pos,
                      target_orn,
                      action[-1]]

            self.img_save_viz_mb(ns)
            ns, _, _, info = self.env.step(action)             
            curr_pos = np.array(info["robot_info"]["tcp_pos"])
            curr_orn = np.array(info["robot_info"]["tcp_orn"])

            angle_diff = curr_orn - target_orn
            error = (target_pos - curr_pos)
            ts += 1
            # Change motor control to stay in place
            # ns, _, _, info = self.env.step(curr_pos)             
        self.img_save_viz_mb(ns)
        return curr_pos, (ns, None, None, info)

    def img_save_viz_mb(self, ns):
        if self.viz_obs:
            _caption = "MB: %s" % self.curr_caption
            join_vis_lang(ns['rgb_obs']['rgb_static'], _caption)
            # img = cv2.resize([:, :, ::-1], (300,300))
            # cv2.imshow("static_cam", img)
            cv2.waitKey(1)

        if self.save_viz:
            self.save_img(ns["rgb_obs"]["rgb_static"], "./model_based/static_cam")
            self.save_img(ns["rgb_obs"]["rgb_gripper"], "./model_based/gripper_cam")
            self.save_dir["step_counter"] += 1