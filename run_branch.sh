
# get preprocesss
python run_pipeline_testcase.py --dataset kjain14/testgeneval \
    --results_dir ./results/ \
    --num_processes 1024 \
    --repo all \
    --data_path "./data" \
    --get_ground_truth_branch

# get branch for glmf generated
python run_pipeline_testcase.py --dataset kjain14/testgenevallite \
    --results_dir ./results/ \
    --num_processes 16 \
    --repo all \
    --data_path "./data" \
    --glmf_generated_path "testing_Qwen2.5_7B_graph_tr_accelerate_4096_baseline.json" \
    --glmf_generated_output "7b-baseline-output.jsonl" \
    --eval_generated