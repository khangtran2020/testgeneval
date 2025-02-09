# TestGenEval Dataset


**Forked from [testgeneval](https://github.com/facebookresearch/testgeneval.git) of facebook research**

TestGenEval consists of 1,210 code test file pairs from 11 large, well-maintained repositories (3,523-78,287 stars). We use these file pairs to construct two testing tasks: 1) unit test completion for the first, last and additional tests and 2) full file unit test generation. Our benchmark is easy to run and extend, as we have docker containers for each version of each repository with coverage and mutation testing dependencies installed. For both task we use execution based metrics, including pass@1, pass@5 along with code coverage improvement, and mutation score improvement compared to the gold (human written) tests. Code and test files in \benchmark are long in length (on average 782 LOC per code file and 677 LOC per test file) and high coverage (median coverage of 60.4\%).

<!-- We measure the following metrics for the test completion task:

- pass@k (k = 1, 5)
- coverage improvement (how much generated test improves existing coverage)
- coverage improvement@pass (coverage improvement averaged only over passing tests)
- average pass@5

We measure the following metrics for the test generation task:
- pass@1
- all pass@1 (all tests generated in suite pass)
- coverage (coverage of generated tests)
- coverage@pass (coverage of generated tests for passing examples)
- mutation score (mutation score of generated tests)
- mutation score@pass (mutation score of generated tests for passing examples) -->

## Datasets

### TestGenEvalLite
Docker images for testbeds used in the `TestGenEvalLite` dataset has been built and tested.

### TestGenEval
Docker images for testbeds used in the `TestGenEval` dataset has been built and tested.

## Setup

To setup the repository run
```shell
conda env create -f testgeneval.yaml
conda activate testgeneval
```

Set SWEBENCH_DOCKER_FORK_DIR to the current directory where the repository was cloned

```shell
export SWEBENCH_DOCKER_FORK_DIR=<current-directory-of-testgeneval>
```

## Building TestGenEval

To build the docker images locally (adapted from [SWEBench Docker](https://github.com/aorwall/SWE-bench-docker/tree/main/docker)) run one of these commands:

**TestGenEvalLite** - TestGenEvalLite for faster evaluation
```
make -f Makefile.testgenevallite
```

**TestGenEval** - full TestGenEval (takes hours to a full day to build)
```
make -f Makefile.testgeneval
```

## Running TestGenEval

### Get branches of the human-written test cases

```shell
python run_pipeline_testcase.py --dataset kjain14/testgenevallite \
    --results_dir ./results/ \
    --num_processes 128 \
    --repo all \
    --get_ground_truth_branch
```

### Analyze \& Translate

```shell
python run_pipeline_testcase.py --dataset kjain14/testgenevallite \
    --results_dir ./results/ \
    --num_processes 16 \
    --repo all \
    --analyze \
    --translate \
    --eval_translate \
    --port 1234 --model Qwen/Qwen2.5-Coder-32B-Instruct 
    --num_try <num-sample-for-each-testcase>
```

## Licensing

The majority of code in this repository is licensed under CC-by-NC, however the third party code/files may be subject to different licenses.
