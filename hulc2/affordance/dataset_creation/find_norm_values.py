import numpy as np
import os
import logging
import json
import argparse


def update_json(output_path, new_data):
    with open(output_path, "r+") as outfile:
        data = json.load(outfile)
        data.update(new_data)
        outfile.seek(0)
        json.dump(data, outfile, indent=2)


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def add_norm_values(run_dir, data_dir, episodes_file="episodes_split.json"):
    '''
        For the trianing split only:
        Get mean and std of gripper img, static img, depth aff. value (static cam)
        and append them to json.
    '''
    logfile = os.path.join(run_dir, "std.log")
    logging.basicConfig(filename=logfile, 
                        format='[%(asctime)s %(message)s]', 
                        filemode='w',
                        ) 
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 

    json_filepath = os.path.join(data_dir, episodes_file)
    data = read_json(json_filepath)
    logger.info("Openning file %s" % json_filepath)

    cams = ["static_cam", "gripper_cam"]
    split = "training"
    new_data = {"depth":{cam:[] for cam in cams}}
    for cam in cams:
        split_data = []
        episodes = list(data[split].keys())

        # Get files corresponding to cam
        for ep in episodes:
            for file in data[split][ep][cam]:
                # Load file
                # cam_folder, filename = os.path.split(file.replace("\\", "/"))
                file_dir = os.path.join(data_dir, f"{ep}/data/{cam}/{file}.npz")
                step_data = np.load(file_dir)
                # new_data["%s_cam" % cam].append(data)
                tcp_cam = step_data['tcp_pos_cam_frame']
                depth = tcp_cam[-1] * -1
                new_data['depth'][cam].append(depth)
        print("%s images: %d" % (split, len(split_data)))

    for k in new_data.keys():
        for cam in cams:
            new_data[k][cam] = {
                "mean": np.mean(new_data[k][cam]),
                "std": np.std(new_data[k][cam]),
                "min": np.min(new_data[k][cam]),
                "max": np.max(new_data[k][cam])
            }
    update_json(json_filepath, {"norm_values": new_data})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hydra.run.dir", type=str, default="", help="rundir")
    parser.add_argument("--data_dir", type=str, default="", help="Specify a filepath!")
    parser.add_argument("--episodes_file", type=str, default="episodes_split")
    args = parser.parse_args()

    run_dir = args.__dict__["hydra.run.dir"].split('=')[-1]
    add_norm_values(run_dir, args.data_dir, args.episodes_file + ".json")