import json
import os
import time

import torch


def parse_comparison_dict_to_BO_data(comparison_dict):
    """
    Parse a dictionary of comparisons into Bayesian Optimization (BO) data.

    Args:
        comparison_dict (dict): Dictionary containing comparison data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, list, list, str]: 
            - X (torch.Tensor): Tensor of experiment parameters.
            - comps (torch.Tensor): Tensor of pairwise comparisons.
            - idx_crash (list): List of indices for crashed experiments.
            - idx_not_crash (list): List of indices for non-crashed experiments.
            - id_best (str): ID of the best experiment from the last comparison.
    """
    # initialize variables
    X = []
    comps = torch.empty((0, 2))
    idx_crash = []
    idx_not_crash = []
    for i in range(len(comparison_dict)):
        X, comps, idx_crash, idx_not_crash = add_single_comparison(idx_crash,idx_not_crash,X,comps,comparison_dict[i])    

    
    
    # this is used to extract the current best from the last comparison
    last_comp = comparison_dict[i]
    if last_comp["Decision"] == "1 > 2" or last_comp["Decision"]   == "2 crashed" :
        id_best = last_comp["Experiment 1"]["Id"]
    elif last_comp["Decision"] == "2 > 1" or last_comp["Decision"] == "1 crashed" :  
        id_best = last_comp["Experiment 2"]["Id"]
    else:
        id_best = last_comp["Experiment 1"]["Id"] 

    return X,comps,idx_crash,idx_not_crash,id_best 


def add_single_comparison(idx_crash, idx_not_crash , X, comps, comparison):
    """
    This is the CrashPBO mechanism here. 
    Add a single comparison to the Bayesian Optimization (BO) data.

    Args:
        idx_crash (list): List of indices for crashed experiments.
        idx_not_crash (list): List of indices for non-crashed experiments.
        X (torch.Tensor): Tensor of experiment parameters.
        comps (torch.Tensor): Tensor of pairwise comparisons.
        comparison (dict): Dictionary containing a single comparison.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, list, list]: 
            - Updated X (torch.Tensor): Tensor of experiment parameters.
            - Updated comps (torch.Tensor): Tensor of pairwise comparisons.
            - Updated idx_crash (list): List of indices for crashed experiments.
            - Updated idx_not_crash (list): List of indices for non-crashed experiments.
    """
    next_X = torch.stack((torch.Tensor(comparison["Experiment 1"]["Parameters"]),torch.Tensor(comparison["Experiment 2"]["Parameters"])))
    if len(X) == 0: 
        X = torch.empty((0,next_X.shape[-1]))
    #torch.tensor([[]]).long()
    if comparison["Decision"] == "both crashed": # borh crashed
        if len(comps.shape) == 1:
            comps = comps.unsqueeze(0)
        if len(comps.shape) == 3:     
            comps = comps.squeeze(0)
        idx_crash.append(X.shape[-2])
        idx_crash.append(X.shape[-2] + 1)
        for idx in idx_not_crash:
            comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]]).unsqueeze(0)], dim=0) 
            comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]+1]).unsqueeze(0)], dim=0)
        comps = comps.unsqueeze(0)       
    elif comparison["Decision"] == "1 crashed":
        if len(comps.shape) == 1:
            comps = comps.unsqueeze(0)
        if len(comps.shape) == 3:     
            comps = comps.squeeze(0)
        idx_crash.append(X.shape[-2])
        idx_not_crash.append(X.shape[-2]+1)
        for idx in idx_not_crash:
            comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]]).unsqueeze(0)], dim=0)
        for idx in idx_crash:
            comps = torch.cat([comps, torch.tensor([X.shape[-2]+1, idx]).unsqueeze(0)], dim=0) 
        comps = torch.cat([comps, torch.tensor([X.shape[-2] + 1, X.shape[-2]]).unsqueeze(0)], dim=0)
        comps = comps.unsqueeze(0)
    elif comparison["Decision"] == "2 crashed":
        if len(comps.shape) == 1:
            comps = comps.unsqueeze(0)
        if len(comps.shape) == 3:     
            comps = comps.squeeze(0)
        idx_not_crash.append(X.shape[-2])
        idx_crash.append(X.shape[-2] + 1)
        for idx in idx_not_crash:
            comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]+1]).unsqueeze(0)], dim=0)
        for idx in idx_crash:
            comps = torch.cat([comps, torch.tensor([X.shape[-2], idx]).unsqueeze(0)], dim=0)
        comps = torch.cat([comps, torch.tensor([X.shape[-2], X.shape[-2] + 1]).unsqueeze(0)], dim=0)
        comps = comps.unsqueeze(0)
    else:
        if len(comps.shape) == 1:
            comps = comps.unsqueeze(0)
        if len(comps.shape) == 3:     
            comps = comps.squeeze(0)
        for idx in idx_crash:
            comps = torch.cat([comps, torch.tensor([X.shape[-2] + 1, idx]).unsqueeze(0)], dim=0)
            comps = torch.cat([comps, torch.tensor([X.shape[-2], idx]).unsqueeze(0)], dim=0)
        idx_not_crash.append(X.shape[-2])
        idx_not_crash.append(X.shape[-2] + 1)

        if comparison["Decision"] == "1 > 2":
            # first experiment preferred
            next_comps = torch.tensor([[0, 1]])
        elif comparison["Decision"] == "2 > 1":
            next_comps = torch.tensor([[1, 0]])
        else:
            raise Exception("Unknown decision, sorry") 

        #next_comps = generate_comparisons(next_y, n_comp=q_comp, noise=noise)
        comps = torch.cat([comps, next_comps + X.shape[-2]], dim=0)
   
    X = torch.cat([X, next_X])
    if len(comps.shape) == 1:
        comps = comps.unsqueeze(0)
    if len(comps.shape) == 3:     
        comps = comps.squeeze(0)

    comps = torch.unique(comps,dim = 0)
    
    return X, comps, idx_crash, idx_not_crash


def safely_read_db(data_base_path,data_base_path_blocked):
    """
    Safely read a database file, ensuring no concurrent access.

    Args:
        data_base_path (str): Path to the database file.
        data_base_path_blocked (str): Path to the blocked file.

    Returns:
        dict or None: Parsed comparison dictionary if successful, otherwise None.
    """
    if os.path.exists(data_base_path) and not os.path.exists(data_base_path_blocked):
        with open(data_base_path_blocked, 'w'): pass
        print("results file found")
        succesfull_read = False
        with open(data_base_path) as json_file:
            comparison_dict = json.load(json_file)
            print(comparison_dict)
            succesfull_read = True
        try:
            os.remove(data_base_path_blocked)
        except:
            time.pause(0.1)
            os.remove(data_base_path_blocked) 
        if succesfull_read: 
            return comparison_dict 
        else:
            return None
    else:
        return None