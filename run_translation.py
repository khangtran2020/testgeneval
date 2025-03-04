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
from utils.function_analyzer import combine_translate_one_task
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # read data
    tasks = get_test_tasks(args.data_path)
    if not isinstance(tasks, list):
        raise ValueError(f"Data from {args.data_path} must contain an array of tasks")

    print(f"Debug mode: {args.debug}")
    if args.debug:
        print(f"First task keys: {pretty_repr(tasks[0].keys())}")
        print(f"Number of tasks: {len(tasks)}")
        # exit()
    if args.repo != "all":
        tasks = [t for t in tasks if t[REPO_ID] == args.repo]
    print(f"Number of tasks after filtering by repo: {len(tasks)}")

    num_test_case = 0
    for task in tasks:
        num_test_case += len(task["test_cases"].keys())
    logger.info(
        f"# of task to translate: {len(tasks)}. # of test cases: {num_test_case}"
    )
    task_dict = {task[KEY_ID]: task for task in tasks}

    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.host}:{args.port}/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # check existed results
    already_processed = []
    if os.path.exists(args.res_path):
        with open(args.res_path, "r") as f:
            for line in f:
                task = json.loads(line)
                already_processed.append(task[KEY_ID])
        logger.info(f"Already processed: {len(already_processed)}")

    for key in task_dict.keys():
        if key in already_processed:
            continue

        for time in range(args.num_try):
            task_dict[key][f"translate_{time}"] = {}
            task_dict[key][f"branch_translate_{time}"] = {}
            for test_case_key in task_dict[key]["test_cases"].keys():
                task_dict[key][f"translate_{time}"][test_case_key] = ""
                task_dict[key][f"branch_translate_{time}"][test_case_key] = []

    semaphore = asyncio.Semaphore(args.num_processes)

    for i, key in enumerate(task_dict.keys()):

        if key in already_processed:
            continue

        logger.info(f"Processing task {i+1}/{len(task_dict.keys())}")

        prompt_list = []
        prompt_dict = {}

        for test_case_key in task_dict[key]["test_cases"].keys():
            src_code = task_dict[key]["code_src"]
            test_case = task_dict[key]["test_cases"][test_case_key]
            preamble = task_dict[key]["preds_context"]["preamble"]
            method = task_dict[key]["func_info"][test_case_key]
            test_case = test_case.replace(preamble, "")

            message_text = construct_prompt(
                code_src=src_code,
                test_case=test_case,
                preamble=preamble,
                method=method,
                tokenizer=None,
            )

            prompt_list.append((test_case_key, message_text))
            prompt_dict[test_case_key] = message_text

            # if args.debug:
            #     logger.info(f"Prompt: {message_text}")

        results = asyncio.run(run_translate(prompt_list, client, args, semaphore))

        for res in results:

            if res is None:
                continue

            test_case_key, response = res
            if args.debug:
                logger.info(f"Prompt: {prompt_dict[test_case_key]}")
                logger.info(f"Response: {response}")

            for time in range(args.num_try):
                response[time] = response[time].replace("```json", "```")
                if "```" not in response[time]:
                    task_dict[key][f"translate_{time}"][test_case_key] = ""
                else:
                    text_cleaned = response[time].split("```")[1].split("```")[0]
                    if is_pytest_test_case(text_cleaned):
                        task_dict[key][f"translate_{time}"][
                            test_case_key
                        ] = text_cleaned
                    else:
                        task_dict[key][f"translate_{time}"][test_case_key] = ""

        combine_translate_one_task(task_instance=task_dict[key], num_try=args.num_try)

        with open(args.res_path, "a") as f:
            f.write(json.dumps(task_dict[key]) + "\n")

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
    logger.info(f"Translation complete")
    logger.info(f"Processing task {i+1}/{len(task_dict.keys())}")


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
    parser.add_argument("--num_try", type=int, required=True)
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--model", type=str, default="aorwall")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="2605")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
