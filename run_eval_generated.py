import ast
import argparse
import asyncio
import logging
import os
import json
from rich.pretty import pretty_repr
from swebench_docker.constants import KEY_BASELINES, KEY_ID, REPO_ID
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_tasks
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from rich.progress import Progress
from typing import Dict
from utils.function_analyzer import combine_translate_and_preamble
import nest_asyncio

nest_asyncio.apply()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation_baseline")

SYSTEM_MESSAGE_FULL = "You are an expert Python automated testing assistant. Your job is to generate a test function in Pytest format given a human-written test script for a Python module."

PROMPT_FULL = """Below is a human-written test script for the code file:
```python
{test_src}
```

Here are some examples of how to import the code file, (you should use these as reference)
```python
{imports}
```

And here is the function names of the methods and classes under test: 
```json
{method}
```

Your job is to output a corresponding unit test function in Pytest format that obtains the same coverage as the human-written test script.
The unit test must be a function starting with test_. Include all your test imports and setup before your first test. Do not 
run the tests in the function, just output a test function. Do not include a main method to run the tests.

Output the unit test Python function in this format:

```python
Unit test Python code (file level)
```
"""


async def run_request(data, client, args, semaphore):
    async with semaphore:
        test_case_key, message_text = data
        try:
            completion = await client.chat.completions.create(
                model=args.model,
                messages=message_text,
                temperature=args.temperature,
                max_tokens=4096,
                n=args.num_try,
                timeout=180,
            )
            response = [
                completion.choices[i].message.content for i in range(args.num_try)
            ]
            return test_case_key, response
        except Exception as e:
            logger.error(f"Error: {e}")
            return None


async def run_translate(prompt_list, client, args, semaphore):
    tasks = []
    for data in prompt_list:
        task = asyncio.create_task(
            run_request(data=data, client=client, args=args, semaphore=semaphore)
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results


def combine_one_task(task_instance: dict) -> int:

    preamble = task_instance["preds_context"]["preamble"]
    num_fail = 0
    for key in task_instance[f"gen_tests"].keys():
        test_content = combine_translate_and_preamble(
            preamble=preamble,
            translated=task_instance[f"gen_tests"][key],
            repo=task_instance["repo"],
        )
        task_instance[f"gen_tests"][key] = test_content
        if test_content == "":
            num_fail += 1
    return num_fail


def construct_prompt(
    code_src: str, test_case: str, preamble: str, method: str, tokenizer
) -> str:
    message_text = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE_FULL,
        },
        {
            "role": "user",
            "content": PROMPT_FULL.format(
                code_src=code_src,
                test_src=test_case,
                imports=preamble,
                method=method,
            ),
        },
    ]
    # prompt = tokenizer.apply_chat_template(message_text, tokenize=False)
    # prompt += "\nLet's think step by step and execute the request"
    return message_text


def main(args):

    # read data
    tasks = get_test_tasks(args.data_path)
    if not isinstance(tasks, list):
        raise ValueError(f"Data from {args.data_path} must contain an array of tasks")

    with open(args.gen_path, "r") as f:
        gen_tasks = json.load(f)

    gen_dict = {}
    key_list = []
    for key in gen_tasks.keys():
        if "_test_case_" in key:
            uuid = key.split("_test_case_")[0]
            gen_test_case = f"test_case_{key.split('_test_case_')[-1].strip()}"
        else:
            test_case_id = key.split("_test_case_")[-1].strip()
            uuid = key.replace(f"_{test_case_id}", "")
            gen_test_case = f"test_case_{test_case_id}"

        key_list.append(uuid)

        gen_code = gen_tasks[key]
        if not is_pytest_test_case(gen_code):
            gen_code = ""

        if uuid not in gen_dict:
            gen_dict[uuid] = {
                "test_cases": {},
            }
        gen_dict[uuid]["test_cases"][gen_test_case] = gen_code

    print(f"Number of generated test cases: {len(gen_dict)}")
    print(f"Number of tasks: {len(gen_dict.keys())}, gen_dict keys: {gen_dict.keys()}")

    print(f"Debug mode: {args.debug}")
    if args.debug:
        print(f"First task keys: {pretty_repr(tasks[0].keys())}")
        print(f"Number of tasks: {len(tasks)}")
        # exit()
    if args.repo != "all":
        tasks = [t for t in tasks if t[REPO_ID] == args.repo]
    print(f"Number of tasks after filtering by repo: {len(tasks)}")

    task_dict = {task[KEY_ID]: task for task in tasks}
    task_dict_new = {}

    for key in task_dict.keys():
        if key not in key_list:
            continue
        else:
            task_dict_new[key] = task_dict[key]
            task_dict_new[key]["gen_tests"] = {}
            task_dict_new[key]["gen_tests_branches"] = {}
            for test_case_key in task_dict[key]["test_cases"].keys():
                if (test_case_key not in gen_dict[key]["test_cases"].keys()) or (
                    gen_dict[key]["test_cases"][test_case_key] == ""
                ):
                    continue
                else:
                    task_dict_new[key]["gen_tests"][test_case_key] = gen_dict[key][
                        "test_cases"
                    ][test_case_key]
                    task_dict_new[key]["gen_tests_branches"][test_case_key] = []

    num_test_case = 0
    for key in task_dict_new.keys():
        num_test_case += len(task_dict_new[key]["test_cases"].keys())
    logger.info(
        f"# of task to evaluate: {len(task_dict.keys())}. # of test cases: {num_test_case}"
    )

    # openai_api_key = "EMPTY"
    # openai_api_base = f"http://{args.host}:{args.port}/v1"
    # client = AsyncOpenAI(
    #     api_key=openai_api_key,
    #     base_url=openai_api_base,
    # )

    # check existed results
    already_processed = []
    if os.path.exists(args.res_path):
        with open(args.res_path, "r") as f:
            for line in f:
                task = json.loads(line)
                already_processed.append(task[KEY_ID])
        logger.info(f"Already processed: {len(already_processed)}")

    semaphore = asyncio.Semaphore(args.num_processes)

    for i, key in enumerate(task_dict_new.keys()):

        if key in already_processed:
            continue

        logger.info(
            f"Processing task {i+1}/{len(task_dict_new.keys())} and save to {args.res_path}"
        )
        # combine_one_task(task_instance=task_dict[key])

        with open(args.res_path, "a") as f:
            f.write(json.dumps(task_dict_new[key]) + "\n")

    # with Progress() as progress:
    #     main_task = progress.add_task("# of Task", total=len(task_dict.keys()))
    #     for key in task_dict.keys():

    #         inner_task_progress = progress.add_task(
    #             f"# test case", total=len(task_dict[key]["test_cases"].keys())
    #         )
    #         for test_case_key in task_dict[key]["test_cases"].keys():

    #             # if task_dict[key]["branches"][test_case_key] == []:
    #             #     continue
    #             src_code = task_dict[key]["code_src"]
    #             test_case = task_dict[key]["test_cases"][test_case_key]
    #             preamble = task_dict[key]["preds_context"]["preamble"]
    #             method = task_dict[key]["func_info"][test_case_key]

    #             message_text = construct_prompt(
    #                 src_code, test_case, preamble, method, tokenizer
    #             )
    #             for time in range(args.num_try):
    #                 response_ok = False
    #                 try:
    #                     completion = client.chat.completions.create(
    #                         model=args.model,
    #                         messages=message_text,
    #                         temperature=args.temperature,
    #                         max_tokens=8192,
    #                     )
    #                     response = completion.choices[0].message.content
    #                     response_ok = True
    #                 except Exception as e:
    #                     logger.error(f"Error: {e}")

    #                 if response_ok:
    #                     response = response.replace("```python", "```")
    #                     if "```" not in response:
    #                         task_dict[key][f"translate_{time}"][test_case_key] = ""
    #                     else:
    #                         text_cleaned = response.split("```")[1].split("```")[0]
    #                         if is_pytest_test_case(text_cleaned):
    #                             task_dict[key][f"translate_{time}"][
    #                                 test_case_key
    #                             ] = text_cleaned
    #                         else:
    #                             task_dict[key][f"translate_{time}"][test_case_key] = ""
    #                 else:
    #                     task_dict[key][f"translate_{time}"][test_case_key] = ""
    #             progress.advance(inner_task_progress)
    #         progress.remove_task(inner_task_progress)
    #         progress.advance(main_task)

    # res_path = args.res_path.replace(".jsonl", f"num_try_{args.num_try}.jsonl")
    # with open(args.res_path, "w") as f:
    #     for item in task_dict.values():
    #         f.write(json.dumps(item) + "\n")
    logger.info(f"Process complete")
    # logger.info(f"Processing task {i+1}/{len(task_dict_new.keys())}")


def is_pytest_test_case(code_snippet):
    """
    Checks if the given code snippet is a pytest test case.

    Args:
        code_snippet (str): The Python code snippet to analyze.

    Returns:
        bool: True if it is a pytest test case, otherwise False.
    """
    try:
        tree = ast.parse(code_snippet)

        for node in ast.walk(tree):
            # Look for function definitions
            if isinstance(node, ast.FunctionDef):
                # Check if function name starts with "test_"
                if node.name.startswith("test_"):
                    return True

                # Check if the function has a pytest decorator (@pytest.mark.*)
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute) and isinstance(
                        decorator.value, ast.Name
                    ):
                        if (
                            decorator.value.id == "pytest"
                            and decorator.attr.startswith("mark")
                        ):
                            return True

        return False
    except SyntaxError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gen_path", type=str, required=True)
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
