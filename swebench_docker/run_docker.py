# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/run_docker.py

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import time

import dotenv
from swebench_docker.constants import MAP_VERSION_TO_INSTALL, KEY_ID
from typing import Dict

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# Needs to be a fully qualified path for log dir

REPO_DICT = {
    "apimd": "codamosa-kmolyuan_apimd",
    "codetiming": "codamosa-realpython_codetiming",
    "dataclasses_json": "codamosa-lidatong_dataclasses-json",
    "docstring_parser": "codamosa-rr_docstring_parser",
    "flutes": "codamosa-huzecong_flutes",
    "flutils": "codamosa-finite-loop_flutils",
    "httpie": "codamosa-httpie_httpie",
    "isort": "codamosa-pycqa_isort",
    "mimesis": "codamosa-lk-geimfari_mimesis",
    "py_backwards": "codamosa-nvbn_py-backwards",
    "pymonet": "codamosa-przemyslawjanpietrzak_pymonet",
    "pypara": "codamosa-vst_pypara",
    "semantic_release": "codamosa-relekang_python-semantic-release",
    "string_utils": "codamosa-daveoncode_python-string-utils",
    "pytutils": "codamosa-akatrevorjay_pytutils",
    "sanic": "codamosa-sanic-org_sanic",
    "sty": "codamosa-feluxe_sty",
    "thonny": "codamosa-thonny_thonny",
    "typesystem": "codamosa-encode_typesystem",
    "pysnooper": "codamosa-cool-rr_pysnooper",
    "ansible": "codamosa-ansible_ansible",
    "cookiecutter": "codamosa-cookiecutter_cookiecutter",
    "fastapi": "codamosa-tiangolo_fastapi",
    "keras": "codamosa-keras-team_keras",
    "luigi": "codamosa-spotify_luigi",
    "pandas": "codamosa-pandas-dev_pandas",
    "scrapy": "codamosa-scrapy_scrapy",
    "spacy": "codamosa-explosion_spacy",
    "thefuck": "codamosa-nvbn_thefuck",
    "tornado": "codamosa-tornadoweb_tornado",
    "tqdm": "codamosa-tqdm_tqdm",
    "youtube_dl": "codamosa-ytdl-org_youtube-dl",
}


async def run_docker_evaluation(
    task_instance: dict,
    namespace: str,
    log_dir: str,
    setting: str,
    timeout: int = 180,
    verbose: bool = False,
    only_baseline: bool = False,
    skip_mutation: bool = False,
    with_imports: bool = False,
) -> Dict:
    repo_name = task_instance["repo"].replace("/", "_")

    specifications = MAP_VERSION_TO_INSTALL.get(task_instance["repo"], {}).get(
        task_instance.get("version", ""), {}
    )
    image_prefix = "swe-bench"

    # TODO: Change this when deciding
    if "packages" in specifications and specifications["packages"] == "environment.yml":
        container_log_dir = "/home/swe-bench/logs"
    else:
        container_log_dir = "/opt/logs"

    if specifications.get("instance_image", False):
        docker_image = f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
    else:
        if repo_name not in REPO_DICT.keys():
            docker_image = f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"
        else:
            docker_image = f"{namespace}/{image_prefix}-{REPO_DICT[repo_name]}:latest"

    swebench_docker_fork_dir = os.environ.get("SWEBENCH_DOCKER_FORK_DIR")
    logger.info(f"SWEBENCH_DOCKER_FORK_DIR: {swebench_docker_fork_dir}")

    if swebench_docker_fork_dir:
        # Create a temporary file to store the task_instance JSON
        tmpfile_path = tempfile.mktemp(suffix=".json")
        with open(tmpfile_path, "w") as f:
            json.dump(task_instance, f)

        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{log_dir}:{container_log_dir}",
            # Map the swebench_docker fork dir to the container
            # for some reason, swebench_docker has different locations for the different containers :(
            # so we need to map all of them to make it work
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/opt/swebench_docker",
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench_docker",
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench",
            # =======
            # Map file instead pass the instance as env var to avoid "Argument list too long" error
            "-v",
            f"{tmpfile_path}:/home/swe-bench/task_instance.json",
            "-e",
            f"LOG_DIR={container_log_dir}",
            "-e",
            f"SETTING={setting}",
            "-e",
            f"TIMEOUT={timeout}",
            "-e",
            f"ONLY_BASELINE={only_baseline}",
            "-e",
            f"SKIP_MUTATION={skip_mutation}",
            "-e",
            f"IMPORTS={True if with_imports else False}",
            docker_image,
        ]
    else:
        # Base64 encode the instance JSON to be sure it can be passed as an environment variable
        instance_b64 = base64.b64encode(
            json.dumps(task_instance).encode("utf-8")
        ).decode("utf-8")
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "--memory_swappiness" "5",
            "-v",
            f"{log_dir}:{container_log_dir}",
            "-e",
            f"INSTANCE={instance_b64}",
            "-e",
            f"LOG_DIR={container_log_dir}",
            "-e",
            f"SETTING={setting}",
            "-e",
            f"TIMEOUT={timeout}",
            "-e",
            f"ONLY_BASELINE={only_baseline}",
            "-e",
            f"SKIP_MUTATION={skip_mutation}",
            "-e",
            f"IMPORTS={True if with_imports else False}",
            docker_image,
        ]

    cmd_string = " ".join(docker_command)

    if verbose:
        logger.info(cmd_string)

    start_time = time.time()

    try:
        process = await asyncio.create_subprocess_shell(
            cmd_string, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        stdout, stderr = await process.communicate()
        # Decode stdout and stderr from bytes to str
        str_stdout = stdout.decode() if stdout else ""
        str_stderr = stderr.decode() if stderr else ""

        elapsed_time = time.time() - start_time

        if process.returncode != 0:
            logger.warning(
                f"[{task_instance['id']}][{docker_image}]  Error running container:"
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"Stdout - {str_stdout}")
            logger.warning(f"Stderr - {str_stderr}")

        elif "Evaluation succeeded" not in str_stdout:
            logger.warning(
                f"[{task_instance['id']}][{docker_image}]  Container ran successfully in {elapsed_time} seconds, but evaluation failed."
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"stdout - {str_stdout}")
        else:
            logger.info(
                f"[{task_instance['id']}][{docker_image}]  Container ran successfully in {elapsed_time} seconds."
            )

        if ("ground_truth" in setting) or ("branch_evaluation" in setting):
            # read task instance from tmpfile_path
            if os.path.exists(os.path.join(log_dir, f"{task_instance[KEY_ID]}.json")):
                with open(
                    os.path.join(log_dir, f"{task_instance[KEY_ID]}.json"),
                    "r",
                ) as f:
                    task_instance = json.load(f)
                return task_instance
            else:
                logger.error("task_instance_results.json not found")
                return task_instance
    except Exception as e:
        logger.warning(
            f"[{task_instance['id']}][{docker_image}]  Error running container: {e}"
        )
    finally:
        if swebench_docker_fork_dir:
            # Ensure the temporary file is deleted after the Docker process completes
            os.unlink(tmpfile_path)
