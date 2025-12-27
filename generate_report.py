# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/generate_report.py

import argparse
import json
import re

from swebench_docker.swebench_utils import (
    get_eval_reports_for_dir,
    get_instances,
    get_model_eval_summary,
    get_model_report,
)
from swebench_docker.utils import get_eval_refs


def count_methods(code_str):
    """
    Counts the number of methods/functions in a given string of code.

    Args:
    code_str (str): A string containing code.

    Returns:
    int: The number of methods/functions found.
    """
    # Regular expression to find Python function definitions
    pattern = r"\bdef\b\s+\w+\s*\("
    matches = re.findall(pattern, code_str)
    return len(matches)


def get_lines_of_code(code_str):
    """
    Extracts lines of code from a given string.

    Args:
    code_str (str): A string containing code.

    Returns:
    list: A list of lines of code.
    """
    return len(code_str.strip().split("\n"))


def get_preds_report(preds_path, instances):
    with open(preds_path, "r") as f:
        preds = [json.loads(line) for line in f.readlines()]

    preds_report = {
        "loc": [],
        "num_methods": [],
        "baseline_loc": [],
        "baseline_num_methods": [],
    }
    # Track per-repo metrics
    repo_preds_report = {}

    found_full = False
    from tqdm import tqdm

    for pred in tqdm(preds):
        if "full" in pred["preds"]:
            # Get repo for this prediction
            pred_id = pred["id"]
            repo = instances[pred_id].get("repo", "unknown")

            # Initialize repo entry if needed
            if repo not in repo_preds_report:
                repo_preds_report[repo] = {
                    "loc": [],
                    "num_methods": [],
                    "baseline_loc": [],
                    "baseline_num_methods": [],
                }

            for pred_text in pred["preds"]["full"]:
                loc = get_lines_of_code(pred_text)
                num_methods = count_methods(pred_text)
                preds_report["loc"].append(loc)
                preds_report["num_methods"].append(num_methods)
                repo_preds_report[repo]["loc"].append(loc)
                repo_preds_report[repo]["num_methods"].append(num_methods)

            baseline_test = instances[pred["id"]]["preds_context"]["last"]
            if type(baseline_test) is list:
                baseline_test = baseline_test[0]

            baseline_loc = get_lines_of_code(baseline_test)
            baseline_methods = count_methods(baseline_test)
            preds_report["baseline_loc"].append(baseline_loc)
            preds_report["baseline_num_methods"].append(baseline_methods)
            repo_preds_report[repo]["baseline_loc"].append(baseline_loc)
            repo_preds_report[repo]["baseline_num_methods"].append(baseline_methods)
            found_full = True

    final_report = {}
    if found_full:
        final_report["av_pred_full_loc"] = sum(preds_report["loc"]) / len(
            preds_report["loc"]
        )
        final_report["av_pred_full_num_methods"] = sum(
            preds_report["num_methods"]
        ) / len(preds_report["num_methods"])
        final_report["av_baseline_loc"] = sum(preds_report["baseline_loc"]) / len(
            preds_report["baseline_loc"]
        )
        final_report["av_baseline_num_methods"] = sum(
            preds_report["baseline_num_methods"]
        ) / len(preds_report["baseline_num_methods"])

        # Add per-repo breakdown
        final_report["by_repo"] = {}
        for repo, repo_data in repo_preds_report.items():
            if len(repo_data["loc"]) > 0:
                final_report["by_repo"][repo] = {
                    "av_pred_full_loc": sum(repo_data["loc"]) / len(repo_data["loc"]),
                    "av_pred_full_num_methods": sum(repo_data["num_methods"])
                    / len(repo_data["num_methods"]),
                    "av_baseline_loc": sum(repo_data["baseline_loc"])
                    / len(repo_data["baseline_loc"]),
                    "av_baseline_num_methods": sum(repo_data["baseline_num_methods"])
                    / len(repo_data["baseline_num_methods"]),
                    "num_predictions": len(repo_data["loc"]),
                }
    return final_report


def generate_report(
    swe_bench_tasks: str, predictions_path: str, log_dir: str, output_dir: str
):
    instances = get_eval_refs(swe_bench_tasks)
    predictions = get_instances(predictions_path)
    model_name_or_path = predictions[0]["model_name_or_path"]

    report_net = get_eval_reports_for_dir(
        log_dir, instances, verbose=False, raw_only=True, model_name=model_name_or_path
    )

    # Group report_net by repo
    repo_report_net = {}
    for task_id, metrics in report_net.items():
        repo = instances[task_id].get("repo", "unknown")
        if repo not in repo_report_net:
            repo_report_net[repo] = {}
        repo_report_net[repo][task_id] = metrics

    with open(f"{output_dir}/{model_name_or_path}_full.json", "w") as f:
        f.write(json.dumps(report_net, indent=4))

    with open(f"{output_dir}/{model_name_or_path}_full_by_repo.json", "w") as f:
        f.write(json.dumps(repo_report_net, indent=4))

    lexical_report = get_preds_report(predictions_path, instances)

    summary = get_model_eval_summary(
        predicts_path=predictions_path,
        eval_dir=log_dir,
        swe_bench_instances=instances,
        model_name=model_name_or_path,
    )

    summary.update(lexical_report)

    # Generate per-repo summaries
    unique_repos = set(inst.get("repo", "unknown") for inst in instances.values())
    repo_summaries = {}
    for repo in unique_repos:
        repo_summary = get_model_eval_summary(
            predicts_path=predictions_path,
            eval_dir=log_dir,
            swe_bench_instances=instances,
            model_name=model_name_or_path,
            repo=repo,
        )
        if repo_summary["total_predictions"] > 0:
            repo_summaries[repo] = repo_summary

    # Combine overall and per-repo summaries
    combined_summary = {"overall": summary, "by_repo": repo_summaries}

    with open(f"{output_dir}/{model_name_or_path}_summary.json", "w") as f:
        f.write(json.dumps(combined_summary, indent=4))

    report = get_model_report(
        verbose=True,
        model=model_name_or_path,
        predictions_path=predictions_path,
        log_dir=log_dir,
    )

    # Group report by repo
    repo_report = {}
    for category, task_ids in report.items():
        repo_report[category] = {}
        for task_id in task_ids:
            repo = instances.get(task_id, {}).get("repo", "unknown")
            if repo not in repo_report[category]:
                repo_report[category][repo] = []
            repo_report[category][repo].append(task_id)

    combined_report = {"overall": report, "by_repo": repo_report}

    with open(f"{output_dir}/{model_name_or_path}_report.json", "w") as f:
        f.write(json.dumps(combined_report, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_path", type=str, help="Path to predictions file", required=True
    )
    parser.add_argument(
        "--log_dir", type=str, help="Path to log directory", required=True
    )
    parser.add_argument(
        "--swe_bench_tasks",
        type=str,
        help="Path to dataset file or HF datasets name",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to output directory", required=True
    )
    args = parser.parse_args()
    generate_report(**vars(args))
