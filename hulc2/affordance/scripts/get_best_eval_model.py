import argparse
import json
import os

import hydra

from hulc2.utils.utils import get_abspath


def main(json_file):
    with open(json_file) as f:
        data = json.load(f)
    best_model = max(data, key=lambda v: data[v]["avg_seq_len"])
    print(best_model)
    print(data[best_model]["avg_seq_len"])
    print(data[best_model]["chain_sr"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)

    args = parser.parse_args()

    json_file = get_abspath(args.file)
    main(json_file)
