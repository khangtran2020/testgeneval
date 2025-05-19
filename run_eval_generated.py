import argparse
import asyncio
import logging
import os
import json
from rich.pretty import pretty_repr
from swebench_docker.constants import KEY_BASELINES, KEY_ID, REPO_ID
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_tasks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation_baseline")


async def main(
    data_path: str,
    res_path: str,
    gen_path: str,
    namespace: str,
    log_dir: str,
    repo: str = None,
    timeout: int = 60,
    num_processes: int = -1,
    debug: bool = False,
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

    print(f"Debug mode: {debug}")
    if debug:
        print(f"First task keys: {pretty_repr(tasks[0].keys())}")
        print(f"Number of tasks: {len(tasks)}")
        # exit()
    if repo != "all":
        tasks = [t for t in tasks if t[REPO_ID] == repo]
    print(f"Number of tasks after filtering by repo: {len(tasks)}")

    if gen_path.endswith(".jsonl"):
        with open(gen_path, "r", encoding="utf-8") as jsonl_file:
            gen_data = [json.loads(line) for line in jsonl_file]
    elif gen_path.endswith(".json"):
        with open(gen_path, "r", encoding="utf-8") as json_file:
            gen_data = json.load(json_file)

    gen_dict = {}
    for key, value in gen_data.items():

        if "_test_case_" in key:
            uuid = key.split("_test_case_")[0]
            test_id = key.split("_test_case_")[-1]
        else:
            test_id = key.split("_")[-1].strip()
            uuid = key.replace(f"_{test_id}", "")
        if uuid not in gen_dict:
            gen_dict[uuid] = {
                "test_cases": {},
                "branches": {},
            }
        gen_dict[uuid]["test_cases"][f"test_case_{test_id}"] = value
        gen_dict[uuid]["branches"][f"test_case_{test_id}"] = []

    new_tasks = []
    for task in tasks:
        if task[KEY_ID] in gen_dict:
            task["test_cases"] = gen_dict[task[KEY_ID]]["test_cases"]
            task["branches"] = gen_dict[task[KEY_ID]]["branches"]
            new_tasks.append(task)
        else:
            logger.warning(f"Task {task[KEY_ID]} not found in generated data")

    num_test_case = 0
    for task in new_tasks:
        num_test_case += len(task["test_cases"].keys())
    logger.info(
        f"# of task to evaluate: {len(new_tasks)}. # of test cases: {num_test_case}"
    )

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(new_tasks))
    asyncio_tasks = []
    if debug:
        new_tasks = new_tasks[:1]
        test_case_keys = ["test_case_0"]
        print(f"Task: {new_tasks[0][KEY_ID]}, version {new_tasks[0]['version']}")

    task_dict = {task[KEY_ID]: task for task in new_tasks}

    for task_instance in new_tasks:
        if debug:
            print(f"An example of task_instance: {pretty_repr(task_instance.keys())}")
            # exit()
            for testcase in test_case_keys:

                async def run_docker_throttled(task_instance, testcase):
                    async with sem:
                        # TODO: remove generated task
                        return await run_docker_evaluation(
                            task_instance,
                            namespace,
                            log_dir,
                            testcase,
                            timeout,
                            translated=-1,
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

                async def run_docker_throttled(task_instance, testcase):
                    async with sem:
                        return await run_docker_evaluation(
                            task_instance,
                            namespace,
                            log_dir,
                            testcase,
                            timeout,
                            translated=-1,
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
        branch_key = "branches"
        test_case_key = "test_cases"
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
                break

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
    parser.add_argument("--gen_path", type=str, required=True)
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--namespace", type=str, default="aorwall")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
