# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os

from datasets import load_dataset, load_from_disk
from swebench_docker.constants import KEY_ID

from typing import Dict


def get_eval_refs(data_path_or_name: str) -> Dict:
    decode_keys = False
    if os.path.isfile(data_path_or_name):
        if data_path_or_name.endswith(".jsonl"):
            data = [json.loads(l) for l in open(data_path_or_name).readlines()]
        elif data_path_or_name.endswith(".json"):
            data = json.load(open(data_path_or_name, "r"))
    elif os.path.isdir(data_path_or_name):
        data = load_from_disk(data_path_or_name)
        decode_keys = True
    else:
        if "testgeneval" in data_path_or_name:
            data = load_dataset(data_path_or_name)
            decode_keys = True
        else:
            with open(data_path_or_name, "r") as f:
                data = [json.loads(line) for line in f.readlines()]
    if isinstance(data, dict):
        all_data = list()
        all_data.extend(data["test"])
        data = all_data

    return {d[KEY_ID]: d for d in data}


def get_test_tasks(data_path_or_name: str) -> Dict:
    decode_keys = False
    data = None
    if os.path.isfile(data_path_or_name):
        if data_path_or_name.endswith(".jsonl"):
            with open(data_path_or_name, "r", encoding="utf-8") as jsonl_file:
                data = [json.loads(line) for line in jsonl_file]
        elif data_path_or_name.endswith(".json"):
            with open(data_path_or_name, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
    print(f"Loading data from {data_path_or_name} file, from path {data_path_or_name}")
    return data
