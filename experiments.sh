#!/usr/bin/env zsh

# Loop through function seeds (0 to 10) and execute the script for each seed
for function_seed in {0..10}; do
    for gp_dim in {0..8}; do
        echo "Running experiment with function seed: $function_seed and gp seed: $gp_dim"
        time python -m crashpbo.synthetic_experiments.cluster_synthetic --functionseed $function_seed --function "gp" --dim $gp_dim
    done
for test_function in ("ackley" "branin" "hartmann" "cosine8"); do
    echo "Running experiment with function: $test_function"
    time python -m crashpbo.synthetic_experiments.cluster_synthetic --function $test_function
done

done
python -m crashpbo.synthetic_experiments.experiment_evaluation
