from crashpbo.interface.crashEUBO import CrashEUBO
from crashpbo.interface.pbo_experiments import PboStepResponse, PboQuanserPendulum
from datetime import datetime

import pathlib
import os
import torch
import numpy as np
import json
import time
import argparse
from crashpbo.utils.front_end_utils import *


def experiment_pair(X, experiment, comparison_dict):
    """
    Perform two initial experiments and update the comparison dictionary.

    Args:
        X (torch.Tensor): Tensor containing parameters for two experiments.
        experiment (object): Experiment object to run experiments.
        comparison_dict (list): List to store comparison data.

    Returns:
        list: Updated comparison dictionary.
    """
    params_1 = X[0, :].numpy()
    params_2 = X[1, :].numpy()
    id_1 = experiment.run_experiment(params_1)
    id_2 = experiment.run_experiment(params_2)

    comparison_dict.append({
        'id': 0,
        'Experiment 1': {'Id': id_1, 'Parameters': params_1.tolist()},
        'Experiment 2': {'Id': id_2, 'Parameters': params_2.tolist()},
        'Decision': 'none'
    })
    return comparison_dict


def experiment_compare_to_best(X, experiment, comparison_dict, best_id):
    """
    Perform an experiment comparing a new parameter set to the best experiment.

    Args:
        X (torch.Tensor): Tensor containing parameters for the new experiment.
        experiment (object): Experiment object to run experiments.
        comparison_dict (list): List to store comparison data.
        best_id (str): ID of the best experiment.

    Returns:
        list: Updated comparison dictionary.
    """
    params_2 = X[0, :].numpy()
    id_2 = experiment.run_experiment(params_2)

    comparison_dict.append({
        'id': 0,
        'Experiment 1': {'Id': best_id, 'Parameters': experiment.experiment_dict["experiments"][best_id]["Params"]},
        'Experiment 2': {'Id': id_2, 'Parameters': params_2.tolist()},
        'Decision': 'none'
    })
    return comparison_dict


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="Choose the experiment: step_response, quanser_pendulum, wheelbot, drone_backflip")
args = parser.parse_args()

compare_to_best = True
continue_from_previous = False

# Configure optimization settings based on the experiment type
if continue_from_previous:
    rel_folder_name = "step_response"
else:
    if args.experiment == "step_response":
        optimization_config = {
            "Folder Name": "results/step_response" + datetime.now().strftime('%Y%m%d%H%M%S'),
            "Enable Crashes": True,
            "Compare to best": compare_to_best,
            "Experiment Name": "PboStepResponse"
        }
    elif args.experiment == "quanser_pendulum":
        optimization_config = {
            "Folder Name": "results/quanser_pendulum" + datetime.now().strftime('%Y%m%d%H%M%S'),
            "Enable Crashes": True,
            "Compare to best": compare_to_best,
            "Experiment Name": "PboQuanserPendulum"
        }
        
    ### Add other experiments here as needed

    rel_folder_name = optimization_config["Folder Name"]

abs_path = pathlib.Path.cwd()
data_folder = os.path.join(abs_path, rel_folder_name)
db_path = os.path.join(data_folder, "data.json")
experiments_db_path = os.path.join(data_folder, "experiment_db.json")
db_path_blocked = os.path.join(data_folder, "data.json.locked")
experiments_db_path_blocked = os.path.join(data_folder, "experiment_db_blocked.json")
optimization_config_path = os.path.join(abs_path, "optimization_config.json")
problem_def_path = os.path.join(data_folder, "problem_def.json")
opt_active_cmd_path = os.path.join(data_folder, "opt_active")
rerun_right_cmd_path = os.path.join(data_folder, "rerun_right")
rerun_left_cmd_path = os.path.join(data_folder, "rerun_left")

if continue_from_previous:
    with open(os.path.join(data_folder, "optimization_config.json")) as json_file:
        optimization_config = json.load(json_file)

experiment_class = eval(optimization_config["Experiment Name"])

optimization_config["paths"] = {
    "db_path": db_path,
    "experiments_db_path": experiments_db_path,
    "db_path_blocked": db_path_blocked,
    "experiments_db_path_blocked": experiments_db_path_blocked,
    "optimization_config_path": optimization_config_path
}

# Create the data folder if it does not exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
else:
    print(f"The folder '{data_folder}' already exists.")

with open(optimization_config_path, 'w') as json_file:
    json.dump(optimization_config, json_file)

with open(os.path.join(data_folder, "optimization_config.json"), 'w') as json_file:
    json.dump(optimization_config, json_file)

comparison_dict = []

if continue_from_previous:
    experiment = experiment_class(data_folder, warm_start=True)
    optimizer = CrashEUBO(
        experiment.dim,
        data_folder,
        unscale_fun=experiment.unscale,
        compare_to_best=compare_to_best,
        warm_start=True,
        experiment=experiment,
        constraints=experiment.constraints
    )
else:
    experiment = experiment_class(data_folder)

    prob_def_json = experiment.get_problem_def_json()

    with open(problem_def_path, 'w') as json_file:
        json.dump(prob_def_json, json_file)

    while not os.path.exists(opt_active_cmd_path):
        time.sleep(0.5)

    with open(problem_def_path, 'r') as json_file:
        problem_json = json.load(json_file)

    experiment.set_problem_def_json(problem_json)

    optimizer = CrashEUBO(
        experiment.dim,
        data_folder,
        unscale_fun=experiment.unscale,
        compare_to_best=compare_to_best,
        experiment=experiment,
        constraints=experiment.constraints
    )

    init_X = torch.stack([torch.tensor(experiment.init_first), torch.tensor(experiment.init_second)], dim=0)
    comparison_dict = experiment_pair(init_X, experiment, comparison_dict)

    with open(db_path, 'w') as json_file:
        json.dump(comparison_dict, json_file)

# Running the backend continuously to handle experiments and comparisons
while True:
    time.sleep(0.5)
    comparison_dict = safely_read_db(db_path, db_path_blocked)

    if comparison_dict is not None and (os.path.exists(rerun_right_cmd_path) or os.path.exists(rerun_left_cmd_path)):
        last_comparison = comparison_dict[-1]
        if os.path.exists(rerun_left_cmd_path):
            X = np.array(last_comparison["Experiment 1"]["Parameters"])
            id_new = experiment.run_experiment(X)
            comparison_dict[-1]["Experiment 1"]["Id"] = id_new

        if os.path.exists(rerun_right_cmd_path):
            X = np.array(last_comparison["Experiment 2"]["Parameters"])
            id_new = experiment.run_experiment(X)
            comparison_dict[-1]["Experiment 2"]["Id"] = id_new

        with open(db_path_blocked, 'w'):
            pass

        with open(db_path, 'w') as json_file:
            json.dump(comparison_dict, json_file)

        os.remove(db_path_blocked)

        if os.path.exists(rerun_right_cmd_path):
            os.remove(rerun_right_cmd_path)

        if os.path.exists(rerun_left_cmd_path):
            os.remove(rerun_left_cmd_path)

    if comparison_dict is not None:
        all_params_decided = all(comp["Decision"] != "none" for comp in comparison_dict)

        if all_params_decided:
            X_unscaled, comps, idx_crash, idx_not_crash, id_best = parse_comparison_dict_to_BO_data(comparison_dict)

            if compare_to_best:
                next_x_scaled = optimizer.make_bo_step(
                    experiment.scale(X_unscaled),
                    comps,
                    experiment.scale(experiment.experiment_dict["experiments"][id_best]["Params"])
                )
            else:
                next_x_scaled = optimizer.make_bo_step(experiment.scale(X_unscaled), comps)

            next_x = experiment.unscale(next_x_scaled)

            optimizer.write_optimizer_stats()

            if compare_to_best:
                comparison_dict = experiment_compare_to_best(next_x, experiment, comparison_dict, id_best)
            else:
                comparison_dict = experiment_pair(next_x, experiment, comparison_dict)

            with open(db_path_blocked, 'w'):
                pass

            with open(db_path, 'w') as json_file:
                json.dump(comparison_dict, json_file)

            os.remove(db_path_blocked)