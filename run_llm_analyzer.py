import argparse
import asyncio
import logging
import os
import json
from rich.pretty import pretty_repr
from swebench_docker.constants import KEY_BASELINES, KEY_ID, REPO_ID
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_tasks
from openai import OpenAI
from transformers import AutoTokenizer
from rich.progress import Progress

from typing import Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation_baseline")

SYSTEM_MESSAGE_FULL = "You are an expert Python automated testing assistant. Your job is to analyze a test script for a Python module."

PROMPT_FULL = """Below is a code file:
```python
{code_src}
```

The human-written test script for the code file:
```python
{test_src}
```

Here are some examples of how to import the code file, (you should use these as reference)
```python
{imports}
```
Your job is to output function names or class functions that are being test in the test script .
Output the answer in this format:

```json
{{
    "function_names": <YOUR ANSWERS>,
    "class_functions": <YOUR ANSWERS>
}}
```
"""


def construct_prompt(code_src: str, test_case: str, preamble: str, tokenizer) -> str:
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
            ),
        },
    ]
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
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    for key in task_dict.keys():
        task_dict[key]["func_info"] = {}
        for test_case_key in task_dict[key]["test_cases"].keys():
            task_dict[key]["func_info"][test_case_key] = ""

    with Progress() as progress:
        main_task = progress.add_task("# of Task", total=len(task_dict.keys()))
        for key in task_dict.keys():

            inner_task_progress = progress.add_task(
                f"# test case", total=len(task_dict[key]["test_cases"].keys())
            )
            for test_case_key in task_dict[key]["test_cases"].keys():

                # if task_dict[key]["branches"][test_case_key] == []:
                #     continue

                src_code = task_dict[key]["code_src"]
                test_case = task_dict[key]["test_cases"][test_case_key]
                preamble = task_dict[key]["preds_context"]["preamble"]

                message_text = construct_prompt(
                    src_code, test_case, preamble, tokenizer
                )
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=message_text,
                    temperature=args.temperature,
                    max_tokens=8192,
                )

                response = completion.choices[0].text
                if args.debug:
                    logger.info(f"Prompt: {message_text}")
                    logger.info(f"Response: {response}")
                response = response.replace("```json", "```")
                if "```" not in response:
                    task_dict[key]["func_info"][test_case_key] = ""
                else:
                    text_cleaned = response.split("```")[1].split("```")[0]
                    task_dict[key]["func_info"][test_case_key] = text_cleaned
                progress.advance(inner_task_progress)
            progress.remove_task(inner_task_progress)
            progress.advance(main_task)

    with open(args.res_path, "w") as f:
        for item in task_dict.values():
            f.write(json.dumps(item) + "\n")
    print(f"Translation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=str, required=True)
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
