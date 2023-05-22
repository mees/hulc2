import argparse
import os
import json
from hulc2.utils.utils import get_abspath, split_by_percentage

def main(args):
    root_dir = get_abspath(args.root_dir)
    json_file = os.path.join(root_dir, "episodes_split.json")
    data_percent = [0.75, 0.50, 0.25]

    with open(json_file) as f:
        episodes_split = json.load(f)

    for percentage in data_percent:
        episodes_split_percentage = split_by_percentage(root_dir, episodes_split, percentage)
        jsons_filename = root_dir + "/episodes_split_%s.json" % str(percentage*100)
        with open(jsons_filename, "w") as outfile:
            json.dump(episodes_split_percentage, outfile, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create episodes_split.json for different percentage of original data")
    parser.add_argument("--root_dir", default=None, type=str, help="path to processed dataset")
    args = parser.parse_args()
    main(args)