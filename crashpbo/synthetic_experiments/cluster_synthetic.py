from crashpbo.synthetic_experiments.cluster_experiment import Experiment
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cluster experiment')
    parser.add_argument('--function', type=str, default="ackley", help='Function to optimize')
    parser.add_argument('--dim', type=int, default=0, help='Dimension of the function')
    parser.add_argument('--functionseed', type=int, default=0, help='Function seed')
    args = parser.parse_args()
    test_fun = args.function
    dim = args.dim
    function_seed = args.functionseed
    if dim==0:
        dim = None
    print(f"Function: {test_fun}")
    seeds = 20
    
    start = time.time()
    algorithms = ["crashPBO", "EUBO", "MES", "ISE", "SafeOpt", "random"]
    if test_fun == "gp":
        for algo in algorithms:
            print(f"Running {algo}")
            for mode in ["compare_to_best"]:
                test_experiment = Experiment(algo=algo, tf_name=test_fun, iterations=dim*10, seeds=seeds, noise=0.01, dim=dim, feasible_percentage=0.5, function_seed=function_seed, mode=mode)
                test_experiment.run_experiment()     
    else:
        for algo in algorithms:
            print(f"Running {algo}")
            for mode in ["compare_to_best"]:
                test_experiment = Experiment(algo=algo, tf_name=test_fun, iterations=10, seeds=seeds, noise=0.01, dim=None, feasible_percentage=0.5, function_seed=0, mode=mode)
                test_experiment.run_experiment()
        
    print(f"Time taken for {test_fun}: {time.time() - start}")
