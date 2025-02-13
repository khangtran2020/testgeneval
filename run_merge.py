import os
import json
import argparse
import pandas as pd


def finalizing_data(
    groun_truth_path: str, translate_path: str, num_try: int, final_path: str
) -> None:

    assert os.path.exists(groun_truth_path)
    assert os.path.exists(translate_path)
    assert ".jsonl" in groun_truth_path
    assert ".jsonl" in translate_path

    modes = []
    for i in range(num_try):
        modes.append(f"translate_{i}")

    data = [json.loads(l) for l in open(translate_path).readlines()]
    data_truth = [json.loads(l) for l in open(groun_truth_path).readlines()]

    data = {task["id"]: task for task in data}
    data_truth = {task["id"]: task for task in data_truth}

    uuids = []
    repos = []
    src_codes = []
    test_cases = []
    branches = []

    for key in data.keys():
        for sub_key in data[key]["branches"].keys():
            data[key]["branches"][sub_key] = data_truth[key]["branches"][sub_key]

    for key in data.keys():
        for sub_key in data[key]["branches"].keys():
            for mode in modes:

                if data[key][mode][sub_key] == "":
                    continue
                if branches.append(data[key][f"branch_{mode}"][sub_key]) == []:
                    continue

                uuids.append(data[key]["id"])
                repos.append(data[key]["repo"])
                src_codes.append(data[key]["code_src"])
                test_cases.append(data[key][mode][sub_key])

                if mode == "test_cases":
                    branches.append(data[key]["branches"][sub_key])
                else:
                    branches.append(data[key][f"branch_{mode}"][sub_key])

    df = pd.DataFrame(
        {
            "uuid": uuids,
            "repo": repos,
            "src_code": src_codes,
            "test_case": test_cases,
            "branch": branches,
        }
    )

    final_data = []
    for i in range(df.shape[0]):
        final_data.append(
            {
                "uuid": df["uuid"].iloc[i],
                "repo": df["repo"].iloc[i],
                "src_code": df["src_code"].iloc[i],
                "test_case": df["test_case"].iloc[i],
                "branch": df["branch"].iloc[i],
            }
        )

    with open(final_path, "w") as f:
        for line in final_data:
            f.write(json.dumps(line) + "\n")
    print(f"Final data saved at {final_path}")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--translate_path", type=str, required=True)
    parser.add_argument("--num_try", type=int, required=True)
    parser.add_argument("--final_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    finalizing_data(
        groun_truth_path=args.ground_truth_path,
        translate_path=args.translate_path,
        num_try=args.num_try,
        final_path=args.final_path,
    )
