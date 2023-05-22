import glob
import json
import logging
import os
import re
from typing import DefaultDict

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import tqdm

from hulc2.affordance.dataset_creation.core.real_cameras import CamProjections
from hulc2.utils.utils import get_abspath

logger = logging.getLogger(__name__)


# datacollection
def check_file(filename, allow_pickle=True):
    try:
        data = np.load(filename, allow_pickle=allow_pickle)
        if len(data["rgb_static"].shape) != 3 or len(data["rgb_gripper"].shape) != 3:
            raise Exception("Corrupt data")
    except Exception as e:
        # print(e)
        data = None
    return data


# Select files that have a segmentation mask
def select_files(episode_files, remove_blank_masks, min_labels=3, only_language=False):
    # Skip first n files for static cams since
    # affordances are incomplete at the beginning of episodes
    data = []
    for file in tqdm.tqdm(episode_files):
        head, tail = os.path.split(file)
        # Remove extension name
        file_name = tail.split(".")[0]
        # Last folder
        file_relative_path = os.path.basename(os.path.normpath(head))
        file_name = os.path.join(file_relative_path, file_name)
        if remove_blank_masks or only_language:
            try:
                np_file = np.load(file, allow_pickle=True)
            except:
                continue
            blank_masks_constraint = False
            gripper_label = True
            centers = np_file["centers"]
            if remove_blank_masks:
                gripper_label = "gripper_cam" in file_name and len(centers) > 0
            enough_labels = "static_cam" in file_name and len(np_file["centers"]) >= min_labels
            blank_masks_constraint = gripper_label or enough_labels
            inside_img = len(centers > 0) and all([(c[1:] < np_file["frame"].shape[:2]).all() for c in centers])
            lang_ann = only_language and ("lang_ann" in np_file and len(np_file["lang_ann"]) > 0)
            lang_ann = lang_ann or not only_language
            lang_constraint = lang_ann and inside_img
            # Only add imgs where gripper is almost completely closed
            # at least one pixel is not background
            if blank_masks_constraint and lang_constraint:
                data.append(file_name)
        else:
            data.append(file_name)
    data.sort()
    return data


def split_by_ep(root_dir, min_labels, remove_blank_mask_instances, data_discovery_episodes=[], only_language=False):
    data = {"training": {}, "validation": {}}
    # Episodes are subdirectories
    n_episodes = 0
    if isinstance(root_dir, list):
        for dir_i in root_dir:
            n_episodes += len(glob.glob(dir_i + "/*/"))
    n_episodes = len(glob.glob(root_dir + "/*/"))
    n_val_ep = min(n_episodes // 4, 1)
    train_ep = [*data_discovery_episodes]

    val_ep = [i for i in range(n_episodes - n_val_ep, n_episodes) if i not in train_ep]
    train_ep = [ep for ep in range(n_episodes) if ep not in val_ep]
    data["validation"] = {"episode_%02d" % e: [] for e in val_ep}
    data["training"] = {"episode_%02d" % e: [] for e in train_ep}

    folders = [os.path.basename(os.path.normpath(path)) for path in glob.glob(root_dir + "/*/")]
    n_episodes = len(folders)
    for ep in tqdm.tqdm(range(n_episodes)):
        ep_dir = os.path.join(root_dir, "episode_%02d" % ep)
        ep_str = "episode_%02d" % ep
        split = "validation" if ep in val_ep else "training"

        gripper_cam_files = glob.glob("%s/data/*/*gripper*" % ep_dir)
        selected_gripper_files = select_files(
            gripper_cam_files, remove_blank_mask_instances, only_language=only_language
        )

        static_cam_files = glob.glob("%s/data/*/*static*" % ep_dir)
        selected_static_files = select_files(
            static_cam_files, remove_blank_mask_instances, min_labels=min_labels, only_language=only_language
        )
        ep_data = {"gripper_cam": [f.split(os.sep)[-1] for f in selected_gripper_files],
                   "static_cam": [f.split(os.sep)[-1] for f in selected_static_files]}
        data[split].update({ep_str: ep_data})

        # data[split].update({ep_str: selected_gripper_files})
        # data[split][ep_str].extend(selected_static_files)
    return data


def split_by_ts(root_dir, min_labels, remove_blank_mask_instances=True, only_language=False):
    data = {"training": {}, "validation": {}}

    # Count episodes
    n_episodes = 0
    if isinstance(root_dir, list):
        for dir_i in root_dir:
            n_episodes += len(glob.glob(dir_i + "/*/"))
    n_episodes = len(glob.glob(root_dir + "/*/"))
    data["validation"] = {"episode_%02d" % e: [] for e in range(n_episodes)}
    data["training"] = {"episode_%02d" % e: [] for e in range(n_episodes)}
    # Iterate episodes
    for ep in tqdm.tqdm(range(n_episodes)):
        ep_dir = os.path.join(root_dir, "episode_%02d" % ep)
        full_gripper_cam_files = glob.glob("%s/data/*/*gripper*" % ep_dir)

        full_static_cam_files = glob.glob("%s/data/*/*static*" % ep_dir)
        gripper_data = select_files(full_gripper_cam_files, remove_blank_mask_instances, only_language=only_language)

        static_data = select_files(
            full_static_cam_files, remove_blank_mask_instances, min_labels=min_labels, only_language=only_language
        )

        for split in ["validation", "training"]:
            ep_str = "episode_%02d" % ep
            if split == "validation":
                gripper_cam_files = gripper_data[-len(gripper_data) // 4 :]
                static_cam_files = static_data[-len(static_data) // 4 :]
            else:
                gripper_cam_files = gripper_data[: -len(gripper_data) // 4]
                static_cam_files = static_data[: -len(static_data) // 4]
            ep_data = {"gripper_cam": [f.split(os.sep)[-1] for f in gripper_cam_files],
                    "static_cam": [f.split(os.sep)[-1] for f in static_cam_files]}
            data[split].update({ep_str: ep_data})
            # data[split].update({ep_str: gripper_cam_files})
            # data[split][ep_str].extend(static_cam_files)

    return data


def create_json_file(
    root_dir, split="training", min_labels=3, remove_blank_mask_instances=True, n_classes=2, only_language=False
):
    data = {"training": {}, "validation": {}}

    # Episodes are subdirectories
    n_episodes = 0
    if isinstance(root_dir, list):
        for dir_i in root_dir:
            n_episodes += len(glob.glob(dir_i + "/*/"))
    n_episodes = len(glob.glob(root_dir + "/*/"))
    split_ep = [ep for ep in range(n_episodes)]
    data[split] = {"episode_%02d" % e: [] for e in split_ep}

    for ep in tqdm.tqdm(range(n_episodes)):
        ep_dir = os.path.join(root_dir, "episode_%02d" % ep)
        ep_str = "episode_%02d" % ep

        gripper_cam_files = glob.glob("%s/data/*/*gripper*" % ep_dir)
        selected_gripper_files = select_files(
            gripper_cam_files, remove_blank_mask_instances, only_language=only_language
        )

        static_cam_files = glob.glob("%s/data/*/*static*" % ep_dir)
        selected_static_files = select_files(
            static_cam_files, remove_blank_mask_instances, min_labels=min_labels, only_language=only_language
        )

        ep_data = {"gripper_cam": [f.split(os.sep)[-1] for f in selected_gripper_files],
                   "static_cam": [f.split(os.sep)[-1] for f in selected_static_files]}
        data[split].update({ep_str: ep_data})
    data["n_classes"] = n_classes
    with open(root_dir + "/episodes_split.json", "w") as outfile:
        json.dump(data, outfile, indent=2)


# Split episodes into train and validation
def create_data_ep_split(
    root_dir,
    label_by_ep,
    min_labels=3,
    remove_blank_mask_instances=True,
    task_discovery_ep=[],
    n_classes=2,
    only_language=False,
):
    if label_by_ep:
        data = split_by_ep(root_dir, min_labels, remove_blank_mask_instances, task_discovery_ep, only_language)
    else:
        data = split_by_ts(root_dir, min_labels, remove_blank_mask_instances, only_language)
    data["n_classes"] = n_classes
    out_file = os.path.join(root_dir, "episodes_split.json")
    with open(out_file, "w") as outfile:
        json.dump(data, outfile, indent=2)


# Create directories if not exist
def create_dirs(root_dir, sub_dir, directory_lst):
    dir_lst = [root_dir]
    for d_name in directory_lst:
        new_dir = os.path.join(root_dir, d_name)
        new_dir = os.path.join(new_dir, sub_dir)
        dir_lst.append(new_dir)

    for directory in dir_lst:
        if not os.path.exists(directory):
            os.makedirs(directory)
    dir_lst.pop(0)  # Remove root_dir
    return dir_lst


# Save a directory wtih frames, masks, viz_out
def save_dict_data(data_dict, directory, sub_dir, save_viz=False):
    if len(data_dict.items()) <= 0:
        return

    if save_viz:
        data_dir, viz_frames, viz_aff = create_dirs(
            directory, sub_dir, ["data", "viz_frames", "viz_affordance"]
        )
    else:
        data_dir = create_dirs(directory, sub_dir, ["data"])[0]

    imgKey_to_dir = {
        "frame": viz_frames,
        "viz_out": viz_aff,
        #"viz_dir": viz_dir
    }
    for img_id, img_dict in data_dict.items():
        # Write vizualization output
        if save_viz:
            for imKey, img_dir in imgKey_to_dir.items():
                filename = os.path.join(img_dir, img_id) + ".png"
                img = img_dict[imKey]

                # Resize w/aspect
                new_height = 300
                r = new_height / img.shape[0]
                dim = (int(img.shape[1] * r), new_height)
                img = cv2.resize(img, dim)
                cv2.imwrite(filename, img[:,:,::-1])
            img_dict.pop("viz_out")

        # img_dict = {"frame":np.array, "mask":np.array, "centers": np.array}
        # frame is in BGR
        filename = os.path.join(data_dir, img_id) + ".npz"
        np.savez_compressed(filename, **img_dict)


def get_files_regex(path, search_str, recursive):
    files = glob.glob(os.path.join(path, search_str), recursive=recursive)
    if not files:
        print("No *.%s files found in %s" % (search_str, path))
    files.sort()
    return files


# Ger valid numpy files with raw data
def get_files(path, extension, recursive=False):
    if not os.path.isdir(path):
        print("path does not exist: %s" % path)
    search_str = "*.%s" % extension if not recursive else "**/*.%s" % extension
    files = get_files_regex(path, search_str, recursive)
    return files


def get_data(play_data_dir, labeling_mode, lang_cfg=""):
    # Episodes info
    # Sorted files
    files = []
    load_dir = get_abspath(play_data_dir)

    # Get language annotations if available
    language_annotations = None
    frame_annotations = DefaultDict(dict)
    lang_file = lang_cfg.file
    lang_folder = lang_cfg.folder
    if lang_file is not None:
        language_ann_path = os.path.join(load_dir, lang_folder)
        language_ann_file = os.path.join(language_ann_path, lang_file)
        if os.path.isfile(language_ann_file):
            # Dictionary
            language_annotations = np.load(language_ann_file, allow_pickle=True).item()
            for i, (start, end) in enumerate(language_annotations["info"]["indx"]):
                for fr_i in range(start, end):
                    fr_task = language_annotations["language"]["task"][i]
                    fr_ann = language_annotations["language"]["ann"][i]
                    # Add
                    if fr_task in frame_annotations[fr_i].keys():
                        frame_annotations[fr_i][fr_task].add(fr_ann)
                    else:  # Create
                        frame_annotations[fr_i][fr_task] = {fr_ann}
            logger.info("Loading language annotations from %s" % language_ann_file)
    if labeling_mode == "simulation" or "processed" in labeling_mode:
        # Simulation
        # all the episodes are in a single folder
        ep_start_end_ids = np.load(os.path.join(load_dir, "ep_start_end_ids.npy"))
        end_ids = ep_start_end_ids[:, -1]
        end_ids.sort()
        files = get_files(load_dir, "npz")
    else:
        # Real life experiments
        # Play data dir contains subdirectories
        # With different data collection runs
        eps = glob.glob(load_dir + "/*/")
        episodes = {ep_path: len(glob.glob(ep_path + "*.npz")) - 2 for ep_path in eps}
        end_ids = list(episodes.values())
        for ep_path in episodes.keys():
            f = get_files(ep_path, "npz")
            f.remove(os.path.join(ep_path, "camera_info.npz"))
            files.extend(f)

    r = re.compile(".*camera_info.*")
    if any(r.match(f) for f in files):
        files.remove(os.path.join(load_dir, "camera_info.npz"))
    return files, end_ids, frame_annotations


def viz_rendered_data(path):
    # Iterate images
    files = glob.glob(path + "/*.npz")
    for idx, filename in enumerate(files):
        try:
            data = np.load(filename, allow_pickle=True)
            cv2.imshow("static", data["rgb_static"][:, :, ::-1])  # W, H, C
            cv2.imshow("gripper", data["rgb_gripper"][:, :, ::-1])  # W, H, C
            cv2.waitKey(0)
            # tcp pos(3), euler angles (3), gripper_action(0 close - 1 open)
            print(data["actions"])
        except Exception as e:
            print("[ERROR] %s: %s" % (str(e), filename))


def instantiate_env(cfg, labeling_mode, new_cfg=True):
    play_data_dir = get_abspath(cfg.play_data_dir)
    env = None

    if "real_world" not in labeling_mode:
        teleop_cfg = OmegaConf.load(os.path.join(play_data_dir, ".hydra/merged_config.yaml"))
        if new_cfg:
            # for c_k, c_v in teleop_cfg.cameras.items():
            #     teleop_cfg.cameras[c_k]["_target_"] = teleop_cfg.cameras[c_k]["_target_"].replace(
            #         "calvin_env", "vr_env"
            #     )
            # for c_k, c_v in teleop_cfg.items():
            #     if isinstance(c_v, DictConfig) and "_target_" in c_v:
            #         teleop_cfg[c_k]["_target_"] = teleop_cfg[c_k]["_target_"].replace("calvin_env", "vr_env")
            cfg.env.robot_cfg = teleop_cfg.env.robot_cfg
            # cfg.env.robot_cfg["_target_"] = cfg.env.robot_cfg["_target_"].replace("calvin_env", "vr_env")

        # Instantiate camera to get projection and view matrices
        key_s = "static" if new_cfg else 0
        key_g = "gripper" if new_cfg else 1
        static_cam = hydra.utils.instantiate(teleop_cfg.env.cameras[key_s], cid=0, robot_id=0, objects=None)
        if "tactile" in teleop_cfg.env.cameras:
            teleop_cfg.env.cameras.pop("tactile")

        if new_cfg:
            env = hydra.utils.instantiate(teleop_cfg.env)
        else:
            # Robot cfg is important to get the transformation
            # between robot and gripper cam
            # for k, v in cfg.env.robot_cfg.items():
            #     if k in teleop_cfg.robot:
            #         cfg.env.robot_cfg[k] = v
            env = hydra.utils.instantiate(cfg.env)

            gripper_cam_cfg = cfg.env.cameras["gripper"].copy()
            for k, v in cfg.env.cameras["gripper"].items():
                if k in teleop_cfg.env.cameras[key_g]:
                    gripper_cam_cfg[k] = teleop_cfg.env.cameras[key_g][k]
            teleop_cfg.env.cameras[key_g] = gripper_cam_cfg

        gripper_cam = hydra.utils.instantiate(
            teleop_cfg.env.cameras[key_g],
            cid=env.cid,
            robot_id=env.robot.cid,
            objects=None,
        )

        if not new_cfg:  # camera was in different orn
            import pybullet as p

            gripper_T = gripper_cam.tcp2cam_T
            _displ = np.array(gripper_cam.tcp2cam_T[0]) * np.array([-1, 1, 1])
            gripper_cam.tcp2cam_T[0] = tuple(_displ)
            gripper_cam.tcp2cam_T[1] = tuple(np.abs(gripper_T[1]))
    else:
        cam_params_path = play_data_dir
        dir_content = glob.glob(play_data_dir)
        if "camera_info.npz" in dir_content:
            cam_info = np.load(os.path.join(play_data_dir, "camera_info.npz"), allow_pickle=True)
        else:
            # Has subfolders of recorded data
            if "processed" not in labeling_mode:
                cam_params_path = glob.glob(play_data_dir + "/*/")[0]
            else:
                cam_params_path = play_data_dir
            cam_info = np.load(os.path.join(cam_params_path, "camera_info.npz"), allow_pickle=True)
        teleop_cfg = OmegaConf.load(os.path.join(cam_params_path, ".hydra/config.yaml"))
        gripper_cfg = teleop_cfg.cams.gripper_cam
        gripper_cam = CamProjections(
            cam_info["gripper_intrinsics"].item(),
            cam_info["gripper_extrinsic_calibration"],
            resize_resolution=gripper_cfg.resize_resolution,
            crop_coords=gripper_cfg.crop_coords,
            resolution=gripper_cfg.resolution,
            name=gripper_cfg.name,
        )
        static_cfg = teleop_cfg.cams.static_cam
        static_cam = CamProjections(
            cam_info["static_intrinsics"].item(),
            cam_info["static_extrinsic_calibration"],
            resize_resolution=static_cfg.resize_resolution,
            crop_coords=static_cfg.crop_coords,
            resolution=static_cfg.resolution,
            name="static",
        )
    return static_cam, gripper_cam, teleop_cfg
