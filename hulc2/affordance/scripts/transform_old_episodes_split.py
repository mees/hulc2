from copy import deepcopy
import json
import os

def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def main(root_dir):
    data_old_format = read_json(os.path.join(root_dir, "episodes_split.json"))
    data_new_format = {"training":{}, "validation":{}}

    for split in ["training", "validation"]:
        for ep in data_old_format[split]:
            data_new_format[split][ep] = {"gripper_cam":[], "static_cam":[]}
            _gripper_data, _static_data = [], []
            for frame in data_old_format[split][ep]:
                cam_type, _fram_name = frame.split("/")
                data_new_format[split][ep][cam_type].append(_fram_name)

    new_file = os.path.join(root_dir, "episodes_split_new.json")
    with open(new_file, "w") as outfile:
        json.dump(data_new_format, outfile, indent=2)


if __name__=="__main__":
    root_dir = "/mnt/ssd_shared/Users/Jessica/Documents/hulc2_ssd/datasets/real_world/500k_all_tasks_dataset_15hz"
    main(root_dir)