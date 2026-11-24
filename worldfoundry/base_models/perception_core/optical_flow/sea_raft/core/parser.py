import argparse
import json


def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path) as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args


def parse_args(entry):
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):  # noqa: B007 - Loop target name kept as upstream.
        args_dict[key] = value
    return args
