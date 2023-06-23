import pathlib

import yaml


def read_tasks():
    file_dir = pathlib.Path(__file__).parent.resolve()
    file = file_dir / "tasks.yaml"
    with open(file.as_posix(), "r") as stream:
        try:
            tasks = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return tasks


def read_colors():
    file_dir = pathlib.Path(__file__).parent.resolve()
    file = file_dir / "colors.yaml"
    with open(file.as_posix(), "r") as stream:
        try:
            colors = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return colors
