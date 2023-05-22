import argparse
import json
import os
from pathlib import Path

import yaml


def to_abs(path):
    if os.path.isabs(path):
        return path
    else:
        repo_src_dir = Path(__file__).absolute().parents[1]
        return os.path.abspath(repo_src_dir / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--output_dir", type=str, default="", help="directory to output merged episodes_split.json")

    args = parser.parse_args()
    cfg_path = to_abs("../../config/cfg_merge_dataset.yaml")
    with open(cfg_path, "r") as stream:
        directory_list = yaml.safe_load(stream)["data_lst"]

    if args.output_dir == "":
        output_dir = to_abs(os.path.dirname(directory_list[0]))
    else:
        output_dir = to_abs(args.output_dir)

    print("Writing to %s " % output_dir)
    return output_dir, directory_list


# Merge datasets using json files
def merge_datasets():
    output_dir, directory_list = parse_args()

    new_data = {"training": {}, "validation": {}}
    for dir in directory_list:
        abs_dir = os.path.abspath(dir)
        json_path = os.path.join(abs_dir, "episodes_split.json")
        with open(json_path) as f:
            data = json.load(f)

        # Rename episode numbers if repeated
        data_keys = list(data.keys())
        split_keys = ["validation", "training"]
        other_keys = [k for k in data_keys if k not in split_keys]
        episode = 0
        for split in split_keys:
            dataset_name = os.path.basename(os.path.normpath(dir))
            for key in data[split].keys():
                new_data[split]["/%s/%s" % (dataset_name, key)] = data[split][key]
                episode += 1
        for key in other_keys:
            new_data[key] = data[key]
    # Write output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file = os.path.join(output_dir, "episodes_split.json")
    with open(out_file, "w") as outfile:
        json.dump(new_data, outfile, indent=2)


if __name__ == "__main__":
    merge_datasets()
