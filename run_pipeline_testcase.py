import os
import sys
import json
import argparse
import subprocess
from utils.data import Data
from utils.function_analyzer import combine_translate_all
from utils.console import console


def main(args):

    console.log(
        "NOTE: Make sure you have built the docker images for the appropriate dataset"
    )

    data_suf = args.dataset.split("/")[-1]
    model_suf = args.model.split("/")[-1]

    if model_suf == "Meta-Llama-3.1-405B-Instruct":
        args.model = model_suf

    base_dir = os.path.join(os.path.abspath(args.results_dir), data_suf)
    print(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    log_dir = os.path.join(base_dir, "data_logs", model_suf)
    print("Log dirs:", log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pred_dir = os.path.join(base_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    pred_output_filename = f"{model_suf}__{data_suf}__{args.temperature}__test.jsonl"
    print(pred_output_filename)
    preds_file = os.path.join(pred_dir, pred_output_filename)

    if os.path.exists(preds_file) and args.rerun_preds:
        os.remove(preds_file)

    # Load the dataset
    dataset = Data(
        data_name=args.dataset,
        save_path=args.data_path,
        console=console,
        num_processes=args.num_processes,
        data_path=os.path.join(args.data_path, f"{data_suf}_raw.jsonl"),
    )
    dataset.load_raw_data()
    dataset.process_data()

    # log number of tasks
    data_path = os.path.join(args.data_path, f"{data_suf}.jsonl")
    with open(data_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f.readlines()]
    console.log(f"Number of tasks in dataset: {len(tasks)}")

    # for task in dataset.tasks:

    if args.debug:
        console.log("Debug mode on")

    if args.process_data_only:
        exit(0)

    if args.get_ground_truth_branch:
        eval_cmd = [
            "python",
            "run_eval_testcase.py",
            "--log_dir",
            log_dir,
            "--num_processes",
            str(args.num_processes),
            "--namespace",
            args.namespace,
            "--repo",
            args.repo,
            "--data_path",
            os.path.join(args.data_path, f"{data_suf}.jsonl"),
            "--res_path",
            os.path.join(args.data_path, f"{data_suf}_processed.jsonl"),
            "--timeout",
            str(args.timeout),
        ]
        if args.debug:
            eval_cmd.append("--debug")
        subprocess.run(eval_cmd)

    if args.eval_generated:

        eval_cmd = [
            "python",
            "run_eval_generated.py",
            "--log_dir",
            log_dir,
            "--num_processes",
            str(args.num_processes),
            "--namespace",
            args.namespace,
            "--name",
            args.name,
            "--repo",
            args.repo,
            "--data_path",
            args.ground_truth_path,
            "--gen_path",
            args.gen_path,
            "--res_path",
            args.results_path,
            "--timeout",
            str(args.timeout),
            "--raw",
            str(0),
        ]
        if args.with_imports:
            eval_cmd.append("--with_imports")
        if args.debug:
            eval_cmd.append("--debug")
        subprocess.run(eval_cmd)

    if args.eval_branch:
        extra_cmd = ["--skip_existing"] if not args.rerun_eval else []
        extra_cmd += (
            ["--skip_mutation"]
            if args.skip_mutation and args.model != "baseline"
            else []
        )

        # make output directory
        os.makedirs(os.path.join(args.results_path), exist_ok=True)

        eval_cmd = [
            "python",
            "run_evaluation_glmf.py",
            "--predictions_path",
            args.gen_path,
            "--log_dir",
            log_dir,
            "--save_dir",
            os.path.join(args.results_path, args.glmf_generated_output),
            "--repo",
            args.repo,
            "--timeout",
            str(args.timeout),
            "--swe_bench_tasks",
            args.dataset,
            "--num_processes",
            str(args.num_processes),
            "--namespace",
            args.namespace,
            "--model",
            args.model,
            # "--debug" if args.debug else "",
        ] + extra_cmd
        if args.with_imports:
            eval_cmd.append("--with_imports")

        report_cmd = [
            "python",
            "generate_report.py",
            "--predictions_path",
            os.path.join(args.results_path, args.glmf_generated_output),
            "--log_dir",
            log_dir,
            "--output_dir",
            base_dir,
            "--swe_bench_tasks",
            args.dataset,
        ]
        subprocess.run(eval_cmd)
        subprocess.run(report_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline testcase")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use",
        required=True,
        default="kjain14/testgenevallite",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="directory to save the results",
        required=True,
        default="./results",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LLM to translate the test cases",
        required=False,
        default="Qwen/CodeQwen1.5-7B-Chat",
    )
    parser.add_argument(
        "--num_try",
        type=int,
        help="number of tries to run the translate",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Docker repository namespace",
        required=False,
        default="aorwall",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository name",
        required=False,
        default="astropy/astropy",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data",
        required=False,
        default="./data",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="host to the model",
        required=False,
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=str,
        help="port of the server to the model",
        required=False,
        default="2605",
    )
    parser.add_argument(
        "--temperature", type=float, help="(Optional) Model temperature", default=0.2
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="(Optional) Timeout for evaluation testcase",
        default=300,
    )
    parser.add_argument(
        "--num_processes", type=int, help="Number of processes to run", default=1
    )
    parser.add_argument(
        "--translate", action="store_true", help="(Optional) Skip LLM translation"
    )
    parser.add_argument(
        "--rerun_eval", action="store_true", help="(Optional) Skip LLM translation"
    )
    parser.add_argument(
        "--skip_mutation", action="store_true", help="(Optional) Skip LLM translation"
    )
    parser.add_argument(
        "--process_data_only", action="store_true", help="Only process data"
    )
    parser.add_argument(
        "--get_ground_truth_branch",
        action="store_true",
        help="Extract ground truth branch from human testcases",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Extract the method under test with LLM",
    )
    parser.add_argument("--debug", action="store_true", help="(Optional) Debug mode")
    parser.add_argument(
        "--combine",
        action="store_true",
        help="combine the preamble and the translation",
    )
    parser.add_argument(
        "--name", type=str, help="name of this evaluation run", default="eval_run"
    )
    parser.add_argument(
        "--eval_translate",
        action="store_true",
        help="Extract ground truth branch from human testcases",
    )
    parser.add_argument(
        "--eval_generated",
        action="store_true",
        help="Extract branch from glmf generated testcases",
    )
    parser.add_argument(
        "--eval_branch",
        action="store_true",
        help="Compute branch coverage for the testcases",
    )
    parser.add_argument(
        "--glmf_generated_path",
        type=str,
        help="Path to the glmf generated",
        required=False,
        default="./data",
    )
    parser.add_argument(
        "--glmf_generated_output",
        type=str,
        help="name for the branch output glmf generated",
        required=False,
        default="./data",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge the ground truth and the translated data",
    )
    parser.add_argument(
        "--ground_truth_path", type=str, help="Path to the ground truth data"
    )
    parser.add_argument("--gen_path", type=str, help="Path to the generated data")
    parser.add_argument("--results_path", type=str, help="Path to the results data")
    parser.add_argument(
        "--with_imports", action="store_true", help="(Optional) Include imports"
    )
    args = parser.parse_args()
    main(args)
