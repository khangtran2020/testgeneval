import os
import json
import asyncio
import aiofiles
import datetime
from tqdm.asyncio import tqdm
from utils.utils import (
    # extract_preamble_classes_and_functions,
    # postprocess_functions,
    # postprocess_tests,
    extract_minimal_test,
    find_tests_in_script,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets

# typing
from rich.console import Console
from typing import Dict


class Data(object):

    def __init__(
        self,
        data_name: str,
        data_path: str = None,
        save_path: str = None,
        num_processes: int = 1,
        console: Console = None,
    ) -> None:

        if (
            data_name not in ["kjain14/testgeneval", "kjain14/testgenevallite"]
            and data_path is None
        ):
            raise ValueError(
                "Invalid data name without data path, please provide data path"
            )
        self.data_name = data_name
        self.data_path = data_path
        self.save_path = save_path
        self.console = console
        self.num_processes = num_processes

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def load_raw_data(self) -> None:
        if self.data_path is None:
            dataset = load_dataset(self.data_name, split="test")
        else:
            dataset = load_from_disk(self.data_path)
        self.dataset = dataset
        self.console.log(f"Data {self.data_name} loaded successfully")

    def process_data(self) -> None:
        if os.path.exists(
            os.path.join(self.save_path, f"{self.data_name.split('/')[-1]}.jsonl")
        ):
            self.console.log(f"Data {self.data_name} already processed")
            return
        asyncio.run(self.process_raw_data())

    async def process_raw_data(self) -> None:
        data_list = []
        num_test_cases = 0
        num_data_point = 0

        semaphore = asyncio.Semaphore(self.num_processes)

        tasks = [
            self.process_one_raw(self.dataset[i], semaphore)
            for i in range(len(self.dataset))
        ]
        await asyncio.gather(*tasks)

    async def process_one_raw(self, data: Dict, semaphore) -> Dict:

        async with semaphore:
            repo = data["repo"]
            commit_id = data["base_commit"]
            version = data["version"]
            instance_id = data["instance_id"]
            patch = data["patch"]
            test_patch = data["test_patch"]
            preds_context = data["preds_context"]
            code_src = data["code_src"]
            test_src = data["test_src"]
            code_file = data["code_file"]
            test_file = data["test_file"]
            local_imports = data["local_imports"]
            idx = data["id"]
            baseline_covs = data["baseline_covs"]

            self.console.log(f"[blue]Working on instance id: {instance_id}[/blue]")

            # preamble, classes, test_functions = extract_preamble_classes_and_functions(
            #     code=test_src
            # )
            test_dict = find_tests_in_script(test_src)

            # self.console.log(f"==================== {idx} ====================")
            test_cases = {}
            test_id = 0
            for tar_func in test_dict["test_functions"]:
                # trimmed_code = trim_test_cases(
                #     source_code=test_src,
                #     target=tar_func,
                # )
                trimmed_code = extract_minimal_test(script=test_src, target=tar_func)
                if trimmed_code is not None:
                    test_cases[f"test_case_{test_id}"] = {
                        "target": tar_func,
                        "code": trimmed_code,
                    }
                    test_id += 1

            for class_name in test_dict["test_classes"]:
                for method in test_dict["test_classes"][class_name]:
                    trimmed_code = extract_minimal_test(
                        script=test_src,
                        target=f"{class_name}|class_method_split|{method}",
                    )
                    if trimmed_code is not None:
                        test_cases[f"test_case_{test_id}"] = {
                            "target": f"{class_name}.{method}",
                            "code": trimmed_code,
                        }
                        test_id += 1

            branches = {}
            for key in test_cases.keys():
                branches[key] = []
            processed_data = {
                "repo": repo,
                "base_commit": commit_id,
                "version": version,
                "instance_id": instance_id,
                "patch": patch,
                "test_patch": test_patch,
                "preds_context": preds_context,
                "code_src": code_src,
                "test_src": test_src,
                "code_file": code_file,
                "test_file": test_file,
                "local_imports": local_imports,
                "id": idx,
                "baseline_covs": baseline_covs,
                "test_cases": test_cases,
                "branches": branches,
            }

            if len(processed_data["test_cases"].keys()) == 0:
                self.console.log(f"No test cases found for {idx}")

            async with aiofiles.open(
                os.path.join(self.save_path, f"{self.data_name.split('/')[-1]}.jsonl"),
                "a",
            ) as f:
                await f.write(json.dumps(processed_data) + "\n")

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.console.log(f"Done {idx} at {current_time}")
            return len(processed_data["test_cases"].keys())
