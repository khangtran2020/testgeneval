import os
import ast
import json
import argparse
import asyncio
import logging
import subprocess
import tempfile
from rich.pretty import pretty_repr
from swebench_docker.constants import KEY_BASELINES, KEY_ID, REPO_ID
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_tasks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation_baseline")


def get_importables(code):
    try:
        tree = ast.parse(code)
        importables = []

        for node in tree.body:  # Only iterates through top-level statements
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    importables.append(node.name)

            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    importables.append(node.name)

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if not target.id.startswith("_"):
                            importables.append(target.id)

        return importables
    except:
        return []


def extract_imports(code):
    try:
        tree = ast.parse(code)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                names = ", ".join(alias.name for alias in node.names)
                imports.append((module, names))
        return imports
    except:
        return []


async def main(
    data_path: str,
    res_path: str,
    namespace: str,
    log_dir: str,
    repo: str = None,
    timeout: int = 60,
    num_processes: int = -1,
    debug: bool = False,
    raw: int = 1,
    translated: int = -1,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.
    """
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")
    os.chmod(log_dir, 0o777)

    tasks = get_test_tasks(data_path)
    if not isinstance(tasks, list):
        raise ValueError(f"Data from {data_path} must contain an array of tasks")

    if debug:
        print(f"First task keys: {pretty_repr(tasks[0].keys())}")
        print(f"Number of tasks: {len(tasks)}")
        # exit()
    if repo != "all":
        tasks = [t for t in tasks if t[REPO_ID] == repo]
    print(f"Number of tasks after filtering by repo: {len(tasks)}")

    num_test_case = 0
    for task in tasks:
        num_test_case += len(task["test_cases"].keys())
    logger.info(
        f"# of task to evaluate: {len(tasks)}. # of test cases: {num_test_case}"
    )

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(tasks))
    asyncio_tasks = []
    if debug:
        tasks = tasks[:1]
        test_case_keys = ["test_case_0"]
        print(f"Task: {tasks[0][KEY_ID]}, version {tasks[0]['version']}")

    task_dict = {task[KEY_ID]: task for task in tasks}

    for task_instance in tasks:
        if debug:
            for testcase in test_case_keys:

                async def run_docker_throttled(task_instance, testcase):
                    async with sem:
                        return await run_docker_evaluation(
                            task_instance,
                            namespace,
                            log_dir,
                            testcase,
                            timeout,
                            raw=raw,
                            translated=translated,
                            verbose=True,
                            skip_mutation=True,
                        )

                task = asyncio.create_task(
                    run_docker_throttled(task_instance, testcase)
                )
                asyncio_tasks.append(task)
        else:
            # if debug:
            if len(task_instance["test_cases"].keys()) > 0:
                max_id = max(
                    [int(x.split("_")[-1]) for x in task_instance["test_cases"].keys()]
                )
                logger.info(
                    f"# of test cases: {len(task_instance['test_cases'].keys())}, and max id: {max_id}, {max_id == len(task_instance['test_cases'].keys()) - 1}"
                )
            else:
                logger.info(f"No test cases found for {task_instance[KEY_ID]}")
                max_id = 0

            for testcase in task_instance["test_cases"].keys():

                if translated != -1:
                    if task_instance[f"translate_{translated}"][testcase] == "":
                        continue

                try:
                    # Create a temporary file (not auto-deleted)
                    temp = tempfile.NamedTemporaryFile(delete=False)
                    temp_name = temp.name

                    os.chmod(temp_name, 0o666)
                    temp.write(
                        task_instance["test_cases"][testcase]["code"].encode("utf-8")
                    )
                    temp.flush()
                    try:
                        result = subprocess.run(
                            [
                                "ruff",
                                "check",
                                temp_name,
                                "--select",
                                "F401",
                                "--fix",
                            ],
                            check=True,
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"Error occurred while running ruff: {e}")
                        continue

                    temp.seek(0)
                    with open(temp_name, "r") as f:
                        cleaned_code = f.read()
                    task_instance["test_cases"][testcase]["code"] = cleaned_code
                finally:
                    # Always remove the temp file after processing
                    try:
                        temp.close()
                        os.remove(temp_name)
                        print(f"Temporary file {temp_name} removed.")
                    except Exception as e:
                        print(f"Error removing temp file: {e}")

                imports = extract_imports(
                    code=task_instance["test_cases"][testcase]["code"]
                )
                module_path = (
                    task_instance["code_file"].replace("/", ".").split(".py")[0]
                )
                module_code = task_instance["code_src"]

                is_directly_imported = False
                for imp in imports:
                    if isinstance(imp, tuple):
                        module = imp[0]
                        if module == module_path:
                            is_directly_imported = True
                            break
                    else:
                        if module_path in imp:
                            is_directly_imported = True
                            break

                if not is_directly_imported:
                    logger.info(
                        f"Skipping test case {testcase} in task {task_instance[KEY_ID]} as module {module_path} is not directly imported."
                    )
                    continue

                async def run_docker_throttled(task_instance, testcase):
                    async with sem:
                        return await run_docker_evaluation(
                            task_instance,
                            namespace,
                            log_dir,
                            testcase,
                            timeout,
                            raw=raw,
                            translated=translated,
                            # generated=generated,
                            only_baseline=True,
                            verbose=True,
                            skip_mutation=True,
                        )

                task = asyncio.create_task(
                    run_docker_throttled(task_instance, testcase)
                )
                asyncio_tasks.append(task)

    results = await asyncio.gather(*asyncio_tasks)
    # setting_res = []
    for result in results:
        # print(result)
        if result is None:
            continue
        res, setting = result
        if translated == -1:
            arcs_key = "arcs"
            branch_key = "branches"
            test_case_key = "test_cases"
        else:
            arcs_key = None
            branch_key = f"branch_translate_{translated}"
            test_case_key = f"translate_{translated}"
        logger.info(f"================== Task {res[KEY_ID]} ==================")
        # logger.info(f"Task instance branches at setting {setting}: {res[branch_key]}")
        logger.info(
            f"Task {res[KEY_ID]} orignally has {len(task_dict[res[KEY_ID]][branch_key])} branches and {len(task_dict[res[KEY_ID]][test_case_key])} test cases"
        )
        logger.info(
            f"Results {res[KEY_ID]} orignally has {len(res[branch_key])} branches and {len(res[test_case_key])} test cases"
        )
        # for key in res[branch_key].keys():
        # logger.info(f"Setting {setting} has {len(res[branch_key][setting])} branches")
        for setting_ in res[branch_key].keys():
            if res[branch_key][setting_] != []:
                logger.info(
                    f"Setting {setting_} at setting {setting} has {len(res[branch_key][setting_])} branches"
                )
                task_dict[res[KEY_ID]][branch_key][setting_] = res[branch_key][setting_]
                # break

            if arcs_key is not None:
                if res[arcs_key][setting_] != []:
                    logger.info(
                        f"Setting {setting_} at setting {setting} has {len(res[arcs_key][setting_])} arcs"
                    )
                    task_dict[res[KEY_ID]][arcs_key][setting_] = res[arcs_key][setting_]
                    # break

        # task_dict[res[KEY_ID]]["branches"][setting] = res["branches"][setting]

    with open(res_path, "w") as f:
        for item in task_dict.values():
            f.write(json.dumps(item) + "\n")
    print(f"Evaluation complete")

    try:
        # List all files in the directory
        files = os.listdir(log_dir)
        # Iterate through the files
        for file in files:
            # Construct full file path
            file_path = os.path.join(log_dir, file)

            # Check if the file has a .json extension and is a file
            if file.endswith(".json") and os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        print("All .json files have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--repo", type=str, required=False, default="all")
    parser.add_argument("--namespace", type=str, default="aorwall")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--translated", type=int, required=True)
    parser.add_argument("--raw", type=int, required=True)
    # parser.add_argument("--generated", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
