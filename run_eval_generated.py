import argparse
import asyncio
import logging
import os
import json
from rich.pretty import pretty_repr
from swebench_docker.constants import KEY_BASELINES, KEY_ID, REPO_ID, KEY_INSTANCE_ID
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
    name: str,
    namespace: str,
    log_dir: str,
    raw: int,
    repo: str = None,
    timeout: int = 60,
    num_processes: int = -1,
    debug: bool = False,
    with_imports: bool = False,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.
    """
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")
    os.chmod(log_dir, 0o777)

    # Get ground truth data
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

    # Load generated data
    if gen_path.endswith(".jsonl"):
        with open(gen_path, "r", encoding="utf-8") as jsonl_file:
            gen_data = [json.loads(line) for line in jsonl_file]
    elif gen_path.endswith(".json"):
        with open(gen_path, "r", encoding="utf-8") as json_file:
            gen_data = json.load(json_file)

    if isinstance(gen_data, list):
        gen_data = {list(item.keys())[0]: list(item.values())[0] for item in gen_data}

    evaluation_dict = {}
    gen_dict = {}
    for key, value in gen_data.items():

        if "_test_case_" in key:
            uuid = key.split("_test_case_")[0]
            test_id = key.split("_test_case_")[-1]
        else:
            test_id = key.split("_")[-1].strip()
            uuid = key.replace(f"_{test_id}", "")

        if uuid not in gen_dict.keys():
            gen_dict[uuid] = {
                "test_cases": {},
                "branches": {},
            }
        gen_dict[uuid]["test_cases"][f"test_case_{test_id}"] = {"code": value}
        gen_dict[uuid]["branches"][f"test_case_{test_id}"] = []

    new_tasks = []
    for task in tasks:
        if task[KEY_ID] in gen_dict:
            # since the train/test split is divided by the instance_id, the instance should have all test cases
            evaluation_dict[task[KEY_ID]] = {
                "original_branches": {
                    k: v
                    for k, v in task["branches"].items()
                    if k in gen_dict[task[KEY_ID]]["branches"].keys()
                },
                "generated_branches": gen_dict[task[KEY_ID]]["branches"],
            }
            task["test_cases"] = gen_dict[task[KEY_ID]]["test_cases"]
            task["branches"] = gen_dict[task[KEY_ID]]["branches"]
            new_tasks.append(task)

    num_test_case = 0
    for task in new_tasks:
        num_test_case += len(task["test_cases"].keys())
    logger.info(
        f"# of task to evaluate: {len(new_tasks)}. # of test cases: {num_test_case}"
    )

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(new_tasks))
    asyncio_tasks = []
    task_dict = {task[KEY_ID]: task for task in new_tasks}

    # Check same set of keys for evaluation_dict:
    for task_id in evaluation_dict.keys():
        for case_key in evaluation_dict[task_id]["original_branches"].keys():
            if case_key not in evaluation_dict[task_id]["generated_branches"].keys():
                logger.info(
                    f"Task {task_id} - Case {case_key} missing in generated branches. Setting to empty list."
                )
                evaluation_dict[task_id]["generated_branches"][case_key] = []

    for task_instance in new_tasks:

        if len(task_instance["test_cases"].keys()) == 0:
            continue

        async def run_docker_throttled(task_instance):
            async with sem:
                return await run_docker_evaluation(
                    task_instance,
                    namespace,
                    log_dir,
                    "branch_evaluation",
                    timeout,
                    only_baseline=True,
                    verbose=True,
                    skip_mutation=True,
                    with_imports=with_imports,
                )

        task = asyncio.create_task(run_docker_throttled(task_instance))
        asyncio_tasks.append(task)

    results = await asyncio.gather(*asyncio_tasks)
    for res in results:
        # print(result)
        if res is None:
            continue
        branch_key = "branches"
        num_branch_extracted = 0
        for setting_ in res[branch_key].keys():
            if res[branch_key][setting_] != []:
                evaluation_dict[res[KEY_ID]]["generated_branches"][setting_] = res[
                    branch_key
                ][setting_]
                task_dict[res[KEY_ID]][branch_key][setting_] = res[branch_key][setting_]
                num_branch_extracted += 1
        logger.info(
            f"Task {res[KEY_ID]} - Extracted branches for {num_branch_extracted} test cases."
        )

    branch_path = os.path.join(res_path, f"{name}_evaluation_branches.jsonl")
    with open(branch_path, "w") as f:
        for item in task_dict.values():
            f.write(json.dumps(item) + "\n")

    eval_path = os.path.join(res_path, f"{name}_evaluation_summary.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation_dict, f, indent=4)

    overall_acc = 0
    overall_overlap_original = 0
    overall_overlap_generated = 0
    num_tasks = 0

    # Compute branch accuracy and overlap
    for task_id in evaluation_dict.keys():
        total_cases = len(evaluation_dict[task_id]["original_branches"].keys())

        # Compute branch accuracy
        correct_cases = 0
        for case_key in evaluation_dict[task_id]["original_branches"].keys():
            original = evaluation_dict[task_id]["original_branches"][case_key]
            generated = evaluation_dict[task_id]["generated_branches"][case_key]

            contain_all = True
            for branch in original:
                if branch not in generated:
                    contain_all = False
                    break
            if contain_all and original != []:
                correct_cases += 1
        accuracy = correct_cases / total_cases if total_cases > 0 else 0
        overall_acc += accuracy
        num_tasks += 1
        logger.info(
            f"Task {task_id} - Branch Accuracy: {correct_cases}/{total_cases} = {accuracy:.2f}"
        )

        # Compute branch overlap: how many executed branch are in the original branches
        total_overlap_original = 0
        total_overlap_generated = 0
        for case_key in evaluation_dict[task_id]["original_branches"].keys():
            original = set(
                tuple(branch)
                for branch in evaluation_dict[task_id]["original_branches"][case_key]
            )
            generated = set(
                tuple(branch)
                for branch in evaluation_dict[task_id]["generated_branches"][case_key]
            )
            overlap = len(original.intersection(generated))
            total_overlap_original += (
                overlap / len(original) if len(original) > 0 else 0
            )
            total_overlap_generated += (
                overlap / len(generated) if len(generated) > 0 else 0
            )
        average_overlap_original = (
            total_overlap_original / total_cases if total_cases > 0 else 0
        )
        average_overlap_generated = (
            total_overlap_generated / total_cases if total_cases > 0 else 0
        )
        overall_overlap_original += average_overlap_original
        overall_overlap_generated += average_overlap_generated
        logger.info(
            f"Task {task_id} - Average Branch Overlap Original: {average_overlap_original:.2f}, Generated: {average_overlap_generated:.2f}"
        )

    # Save the computed metrics
    evaluation_results = {
        "branch_accuracy": overall_acc / num_tasks if num_tasks > 0 else 0,
        "average_branch_overlap_original": (
            overall_overlap_original / num_tasks if num_tasks > 0 else 0
        ),
        "average_branch_overlap_generated": (
            overall_overlap_generated / num_tasks if num_tasks > 0 else 0
        ),
    }
    eval_result_path = os.path.join(res_path, f"{name}_evaluation_report.json")
    with open(eval_result_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

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
                # print(f"Deleted: {file_path}")
        # print("All .json files have been deleted.")
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
    parser.add_argument("--raw", type=int, required=True)
    parser.add_argument("--name", type=str, help="name of this evaluation run")
    parser.add_argument("--namespace", type=str, default="aorwall")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--with_imports", action="store_true", help="(Optional) Include imports"
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
