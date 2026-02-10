# Standard library imports
import os
import pickle
import random
import warnings
from itertools import combinations

# Third-party imports
import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf

# Local / project imports
from crashpbo.utils.utility_functions import TestFunction
from crashpbo.synthetic_experiments.baselines.ise.gaussian_processes.gaussian_noise_gp import GaussianNoiseGP
from crashpbo.synthetic_experiments.baselines.ise.acquisitions.ise_acquisition import IseAcquisition
from crashpbo.synthetic_experiments.baselines.ise.acquisitions.ise_line_bo_acquisition import IseLineBoAcquisition
from crashpbo.synthetic_experiments.baselines.ise.acquisitions.safe_opt_acquisition import SafeOptAcquisition
from crashpbo.synthetic_experiments.baselines.ise.acquisitions.safe_opt_line_bo_acquisition import SafeOptLineBoAcquisition

# Suppress noisy botorch UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="botorch")

def seed_everything(seed: int):
    """
    Seed everything for reproducibility.

    Args:
        seed (int): The seed value for random number generators.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# force double precision
torch.set_default_dtype(torch.float64)

class Experiment:
    def __init__(self, algo, tf_name, iterations=30, seeds=20, noise=0.01, dim=None, feasible_percentage=None, function_seed=0, mode="compare_to_best"):
        """
        Initialize the Experiment class.

        Args:
            algo (str): Algorithm to use (e.g., "EUBO", "crashPBO", etc.).
            tf_name (str): Name of the test function.
            iterations (int): Number of iterations for the experiment.
            seeds (int): Number of random seeds for trials.
            noise (float): Noise level for utility function.
            dim (int or None): Dimensionality of the test function.
            feasible_percentage (float or None): Percentage of feasible points.
            function_seed (int): Seed for the test function.
            mode (str): Mode of comparison ("compare_to_best", "compare_to_last", "two_new_parameters").

        Returns:
            None
        """
        self.tf = TestFunction(tf_name, seed=function_seed, dim=dim, feasible_percentage=feasible_percentage)
        if tf_name == "gp":
            self.tf_name = f"{tf_name}_seed_{function_seed}"
            self.covar_module = self.tf.objective.kernel
        else:
            self.tf_name = tf_name
            iterations = self.tf.objective.dim * 10
        self.seeds = seeds
        self.noise = noise
        self.algo = algo
        self.mode = mode # compare_to_best, compare_to_last, two_new_parameters
        if mode == "two_new_parameters" and self.algo != "random" and self.algo != "sobolRandom":
            self.iterations = iterations // 2
        else:
            self.iterations = iterations 
        
        self.feasible_percentage = feasible_percentage
        self.results_dict = {}
        print(f"Running experiment with {self.algo} on {self.tf_name} with {self.iterations} iterations, {self.seeds} seeds, noise={self.noise}, dim={dim}, mode={self.mode}")
    
    def noise_free_utility(self, X):
        """
        Compute the noise-free utility for given input points.

        Args:
            X (torch.Tensor): Input points.

        Returns:
            torch.Tensor: Noise-free utility values.
        """
        # y is weighted sum of X, with weight sqrt(i) imposed on dimension i
        y = self.tf(X)
        return y
    
    def utility(self, X):
        """
        Compute the utility with added noise for given input points.

        Args:
            X (torch.Tensor): Input points.

        Returns:
            torch.Tensor: Utility values with noise.
        """
        # y is weighted sum of X, with weight sqrt(i) imposed on dimension i
        y = self.tf(X) + torch.randn(X.shape[0], device=X.device, dtype=X.dtype) * self.noise
        return y
        
    def generate_comparisons(self, y, n_comp, replace=False):
        """
        Generate pairwise comparisons with noise.

        Args:
            y (torch.Tensor): Latent utility values.
            n_comp (int): Number of comparisons to generate.
            replace (bool): Whether to sample with replacement.

        Returns:
            torch.Tensor: Pairwise comparison indices.
        """
        # generate all possible pairs of elements in y
        all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
        # randomly select n_comp pairs from all_pairs
        comp_pairs = all_pairs[
            np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
        ]
        # add gaussian noise to the latent y values
        c0 = y[comp_pairs[:, 0]] 
        c1 = y[comp_pairs[:, 1]]
        reverse_comp = (c0 < c1).numpy()
        comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
        comp_pairs = torch.tensor(comp_pairs).long()

        return comp_pairs
    
    def make_new_data(self, X, next_X, comps, q_comp):
        """
        Generate new data points and comparisons.

        Args:
            X (torch.Tensor): Existing input points.
            next_X (torch.Tensor): New input points.
            comps (torch.Tensor): Existing comparisons.
            q_comp (int): Number of new comparisons to generate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Updated data and comparisons.
        """
        if len(comps.shape) == 1:
            comps = comps.unsqueeze(0)
        if len(comps.shape) == 3:     
            comps = comps.squeeze(0)
        # next_X is float by default; cast it to the dtype of X (i.e., double)
        next_X = next_X.to(X)
        next_y = self.utility(next_X)
        next_comps = self.generate_comparisons(next_y, n_comp=q_comp)
        comps = torch.cat([comps, next_comps + X.shape[-2]])
        X = torch.cat([X, next_X])
        return next_X, next_y, X, comps
    
    def make_new_crash_data(self, X, next_X, comps, q_comp, idx_crash, idx_not_crash, threshold):
        """
        Our data generation mechanism for the crashPBO algorithm.

        Args:
            X (torch.Tensor): Existing input points.
            next_X (torch.Tensor): New input points.
            comps (torch.Tensor): Existing comparisons.
            q_comp (int): Number of new comparisons to generate.
            idx_crash (list): Indices of crash points.
            idx_not_crash (list): Indices of non-crash points.
            threshold (float): Threshold for crash determination.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, list]: Updated data, comparisons, and crash indices.
        """
        # next_X is float by default; cast it to the dtype of X (i.e., double)
        next_X = next_X.to(X)
        next_y = self.utility(next_X)
        next_y_crash = torch.where(next_y < threshold, torch.nan, next_y)
        
        if len(comps.shape) == 1:
            comps = comps.unsqueeze(0)
        if len(comps.shape) == 3:     
            comps = comps.squeeze(0)

        if torch.isnan(next_y_crash[0]) and torch.isnan(next_y_crash[1]):
            idx_crash.append(X.shape[-2])
            idx_crash.append(X.shape[-2] + 1)
            for idx in idx_not_crash:
                comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]]).unsqueeze(0)], dim=0) 
                comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]+1]).unsqueeze(0)], dim=0)
            comps = comps.unsqueeze(0)       
        elif torch.isnan(next_y_crash[0]):
            idx_crash.append(X.shape[-2])
            idx_not_crash.append(X.shape[-2]+1)
            for idx in idx_not_crash:
                comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]]).unsqueeze(0)], dim=0)
            for idx in idx_crash:
                comps = torch.cat([comps, torch.tensor([X.shape[-2]+1, idx]).unsqueeze(0)], dim=0) 
            comps = torch.cat([comps, torch.tensor([X.shape[-2] + 1, X.shape[-2]]).unsqueeze(0)], dim=0)
            comps = comps.unsqueeze(0)
        elif torch.isnan(next_y_crash[1]):
            idx_not_crash.append(X.shape[-2])
            idx_crash.append(X.shape[-2] + 1)
            for idx in idx_not_crash:
                comps = torch.cat([comps, torch.tensor([idx, X.shape[-2]+1]).unsqueeze(0)], dim=0)
            for idx in idx_crash:
                comps = torch.cat([comps, torch.tensor([X.shape[-2], idx]).unsqueeze(0)], dim=0)
            comps = torch.cat([comps, torch.tensor([X.shape[-2], X.shape[-2] + 1]).unsqueeze(0)], dim=0)
            comps = comps.unsqueeze(0)
        else:
            for idx in idx_crash:
                comps = torch.cat([comps, torch.tensor([X.shape[-2] + 1, idx]).unsqueeze(0)], dim=0)
                comps = torch.cat([comps, torch.tensor([X.shape[-2], idx]).unsqueeze(0)], dim=0)
            idx_not_crash.append(X.shape[-2])
            idx_not_crash.append(X.shape[-2] + 1)
            next_comps = self.generate_comparisons(next_y, n_comp=q_comp)
            comps = torch.cat([comps, next_comps + X.shape[-2]], dim=0)
            
        X = torch.cat([X, next_X])
        return next_X, next_y, X, comps, idx_crash, idx_not_crash


    def init_and_fit_model(self, X, comp):
        """
        Initialize and fit a pairwise GP model.

        Args:
            X (torch.Tensor): Input points.
            comp (torch.Tensor): Pairwise comparisons.

        Returns:
            Tuple[gpytorch.mlls.MarginalLogLikelihood, botorch.models.PairwiseGP]: Fitted model and marginal log likelihood.
        """
        if  "gp" in self.tf_name:
            # set output scale to 1/noise
            model = PairwiseGP(
                X,
                comp,
                input_transform=Normalize(d=X.shape[-1]),
                covar_module=ScaleKernel(base_kernel=self.covar_module, outputscale_constraint=Interval(1/self.noise-1e-5, 1/self.noise+1e-5))
            )
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        else:
            model = PairwiseGP(
                X,
                comp,
                input_transform=Normalize(d=X.shape[-1])
            )
            #model.covar_module.lengthscale_constraint = Interval(1e-4, 0.5)
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except:
                print("Error in fitting")
        return mll, model
    
    def init_and_fit_standard_model(self, X, Y):
        """
        Initialize and fit a standard GP model for standard BO baselines. 

        Args:
            X (torch.Tensor): Input points.
            Y (torch.Tensor): Observed utility values.

        Returns:
            Tuple[gpytorch.mlls.MarginalLogLikelihood, botorch.models.SingleTaskGP]: Fitted model and marginal log likelihood.
        """ 
        if "gp" in self.tf_name:
            model = SingleTaskGP(
                X,
                Y,
                input_transform=Normalize(d=X.shape[-1]),
                covar_module=self.covar_module,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)    
        else:
            model = SingleTaskGP(
                X,
                Y,
                input_transform=Normalize(d=X.shape[-1]),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            #try:
            fit_gpytorch_mll(mll)
            
            #except:
            #    print("Error in fitting")
        return mll, model
    
    def run_experiment(self):
        """
        Run the experiment using the specified algorithm and mode.

        Args:
            None

        Returns:
            None
        """
        dim = self.tf.objective.dim
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        threshold = self.tf.threshold
        
        NUM_RESTARTS = 10
        RAW_SAMPLES = 512 
        i = 0
        random_seed = 0
        # average over multiple trials
        while i < self.seeds:
            
            seed_everything(random_seed)
            print(f"Running trial {i}")       
            
            # set up the results dictionary
            
            if dim is None:
                dim = self.tf.objective.dim
            
            self.tf_path = f"{self.tf_name}_{dim}D_feasiblep_{self.tf.feasible_percentage}"
            # Create initial data
            try:
                init_X, init_y = self.tf.generate_random_points(2)
            except ValueError:
                random_seed += 1
                print("Error in generating initial data")
                continue
            
            # add init_y to Y
            Y = init_y
            X = init_X
            
            comparisons = self.generate_comparisons(init_y, 1)             

            if self.algo == "crashPBO":
                idx_crash = []
                idx_not_crash = []
                if init_y[0] < threshold:
                    idx_crash.append(0)
                else:
                    idx_not_crash.append(0)
                if init_y[1] < threshold:
                    idx_crash.append(1)
                else:
                    idx_not_crash.append(1)
            
            data = (X, comparisons)
            
            
            if self.algo == "EUBO" or self.algo == "crashPBO":
                mll, model = self.init_and_fit_model(init_X, comparisons)
                # run the optimization loop
                for j in range(1, self.iterations + 1):
                    X, comps = data
                    if len(comps.shape) == 3:
                        comps = comps.squeeze(0)
                        
                    # Extract the winner from the last comparison in comps
                    last_comparison = comps[-1]
                    winner_index = last_comparison[0] # The winner is the first index in the pair
                    # ensure best_x is a 2D tensor
                    best_x = X[winner_index].unsqueeze(0)
                    
                    if self.mode == "compare_to_best":
                        q = 1  # number of points per query
                        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model, previous_winner=best_x)
                    elif self.mode == "compare_to_last":
                        q = 1
                        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model, previous_winner=X[-1].unsqueeze(0))
                    else:
                        q = 2
                        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
                    # optimize and get new observation
                    next_X, acq_val = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=q,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                    )
                    # update data
                    X, comps = data
                    if self.mode == "compare_to_best":
                        # make next x as best x and add it to the next_X
                        next_X = torch.stack((best_x.squeeze(0), next_X.squeeze(0)))
                    elif self.mode == "compare_to_last":	
                        # append the last x to the next_X
                        next_X = torch.stack((X[-1], next_X.squeeze(0)))
                        
                    if self.algo == "EUBO":
                        next_X, next_y, X, comps = self.make_new_data(X, next_X, comps, 1)
                    elif self.algo == "crashPBO":
                        next_X, next_y, X, comps, idx_crash, idx_not_crash = self.make_new_crash_data(X, next_X, comps, 1, idx_crash, idx_not_crash, threshold)
                    
                    Y = torch.cat([Y, next_y])
                    data = (X, comps)

                    # refit models
                    _, model = self.init_and_fit_model(X, comps)
                
            elif self.algo == "EI":
                mll, model = self.init_and_fit_standard_model(X, Y.unsqueeze(-1))
                for j in range(1, self.iterations + 1):
                    print(f"Running iteration {j}")
                    acq_func = qLogExpectedImprovement(model, best_f=Y.max())
                    if self.mode == "two_new_parameters":
                        q = 2
                    else:
                        q = 1
                    next_X, acq_val = optimize_acqf(acq_function=acq_func, bounds=bounds, q=q, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
                    # optimize and get new observation
                    next_y = self.utility(next_X)
                    X = torch.cat([X, next_X], dim=0)
                    Y = torch.cat([Y, next_y])
                    # refit models
                    _, model = self.init_and_fit_standard_model(X, Y.unsqueeze(-1))
            
            elif self.algo == "MES":
                mll, model = self.init_and_fit_standard_model(X, Y.unsqueeze(-1))
                for j in range(1, self.iterations + 1):
                    candidate_set = torch.rand(1000, dim)
                    acq_func = qMaxValueEntropy(model, candidate_set=candidate_set)
                    if self.mode == "two_new_parameters":
                        q = 2
                        next_X, acq_val = optimize_acqf(acq_function=acq_func, bounds=bounds, q=q, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES, sequential=True)
                    else:
                        q = 1
                        next_X, acq_val = optimize_acqf(acq_function=acq_func, bounds=bounds, q=q, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
                    # optimize and get new observation
                    next_y = self.utility(next_X)
                    X = torch.cat([X, next_X], dim=0)
                    Y = torch.cat([Y, next_y])
                    # refit models
                    _, model = self.init_and_fit_standard_model(X, Y.unsqueeze(-1))
                        
                
            elif self.algo == "ISE":
                safe_mask = Y >= threshold
                if not torch.any(safe_mask):
                    print("Skipping trial: ISE baseline requires at least one safe seed.")
                    random_seed += 1
                    continue
                # use both initial points if both are safe, otherwise use the single safe one
                safe_indices = torch.where(safe_mask)[0]
                max_safe_idx = safe_indices[torch.argmax(Y[safe_mask])]
                safe_seed = X[max_safe_idx].unsqueeze(0)    # shape: (1, dim)
                safe_seed_y = Y[max_safe_idx].unsqueeze(0)  # shape: (1,)
                safe_seed_obs = safe_seed_y - threshold        # shape: (1,)
                safe_seed_obs = safe_seed_y - threshold

                print("Safe seed:", safe_seed)
                print("Safe seed observation:", safe_seed_obs)

                domain = [(0.0, 1.0) for _ in range(dim)]
                kernel = RBFKernel()
                
                if "gp" in self.tf_name:
                    lengthscale_value = self.covar_module.lengthscale.detach().clone()
                    dim = self.tf.objective.dim
                else:
                    dim = self.tf.objective.dim
                    lengthscale_value = torch.tensor(0.1 * self.tf.objective.dim, dtype=torch.float64)

                noise_variance = max(float(self.noise ** 2), 1e-6)
                gp_hyperparameters = {
                    'covar_module.base_kernel.lengthscale': lengthscale_value,
                    'covar_module.outputscale': torch.tensor(1.0, dtype=torch.float64),
                    'mean_module.constant': torch.tensor(0.0, dtype=torch.float64),
                    'likelihood.noise_covar.noise': torch.tensor(noise_variance, dtype=torch.float64),
                }
                gp_model = GaussianNoiseGP(
                    gp_hyperparameters,
                    beta_squared=4.0,
                    kernel=kernel,
                    safe_seed=safe_seed,
                    safe_seed_observation=safe_seed_obs,
                )
                if dim < 4:
                    acquisition = IseAcquisition(
                        gp_model=gp_model,
                        safe_seed=safe_seed,
                        domain=domain,
                        learning_rate=0.01,
                        learning_epochs=100,
                        number_of_samples=max(1000, 200 * dim),
                    )
                else:
                    acquisition = IseLineBoAcquisition(
                        gp_model=gp_model,
                        safe_seed=safe_seed,
                        domain=domain,
                    )
                requested_points = 2 if self.mode == "two_new_parameters" else 1
                for j in range(1, self.iterations + requested_points):
                    print(f"Running iteration {j}")
                    for _ in range(requested_points):
                        try:
                            next_X, acquisition_value = acquisition.optimize()
                            next_y = self.utility(next_X)
                            gp_model.add_observations(next_X, (next_y - threshold).unsqueeze(-1))
                            X = torch.cat([X, next_X], dim=0)
                            Y = torch.cat([Y, next_y])
                        except Exception as e:
                            print(f"Acquisition optimization failed: {e}")
                            # append safe seed for the remaining iterations
                            X = torch.cat([X, safe_seed.repeat((self.iterations - j)*requested_points, 1)], dim=0)
                            Y = torch.cat([Y, safe_seed_y.repeat((self.iterations - j)*requested_points)], dim=0)
                            break

                    if X.shape[0] >= self.iterations + requested_points:
                        break
            
            elif self.algo == "SafeOpt":
                safe_mask = Y >= threshold
                if not torch.any(safe_mask):
                    print("Skipping trial: SafeOpt baseline requires at least one safe seed.")
                    random_seed += 1
                    continue
                # use both initial points if both are safe, otherwise use the single safe one
                safe_count = int(safe_mask.sum().item())
                # use only the safe seed with the maximum Y value
                safe_indices = torch.where(safe_mask)[0]
                max_safe_idx = safe_indices[torch.argmax(Y[safe_mask])]
                safe_seed = X[max_safe_idx].unsqueeze(0)    # shape: (1, dim)
                safe_seed_y = Y[max_safe_idx].unsqueeze(0)  # shape: (1,)
                safe_seed_obs = safe_seed_y - threshold

                print("Safe seed:", safe_seed)
                print("Safe seed observation:", safe_seed_obs)

                domain = [(0.0, 1.0) for _ in range(dim)]
                kernel = RBFKernel()
                
                if "gp" in self.tf_name:
                    lengthscale_value = self.covar_module.lengthscale.detach().clone()
                    dim = self.tf.objective.dim
                else:
                    dim = self.tf.objective.dim
                    lengthscale_value = torch.tensor(0.1 * self.tf.objective.dim, dtype=torch.float64)

                noise_variance = max(float(self.noise ** 2), 1e-6)
                gp_hyperparameters = {
                    'covar_module.base_kernel.lengthscale': lengthscale_value,
                    'covar_module.outputscale': torch.tensor(1.0, dtype=torch.float64),
                    'mean_module.constant': torch.tensor(0.0, dtype=torch.float64),
                    'likelihood.noise_covar.noise': torch.tensor(noise_variance, dtype=torch.float64),
                }
                gp_model = GaussianNoiseGP(
                    gp_hyperparameters,
                    beta_squared=4.0,
                    kernel=kernel,
                    safe_seed=safe_seed,
                    safe_seed_observation=safe_seed_obs,
                )
                if dim < 4:
                    acquisition = SafeOptAcquisition(
                    gp_model,
                    gp_model,
                    safe_seed,
                    domain,
                    self.tf.lipschitz_constant,
                    100,
                    grid_domain=False)
                else:
                    acquisition = SafeOptLineBoAcquisition(
                        gp_model, gp_model, safe_seed, domain, self.tf.lipschitz_constant)

                requested_points = 2 if self.mode == "two_new_parameters" else 1
                for j in range(1, self.iterations + requested_points):
                    print(f"Running iteration {j}")
                    for _ in range(requested_points):
                        try:
                            next_X, acquisition_value = acquisition.optimize()
                            next_y = self.utility(next_X)
                            gp_model.add_observations(next_X, (next_y - threshold).unsqueeze(-1))
                            X = torch.cat([X, next_X], dim=0)
                            Y = torch.cat([Y, next_y])
                        except Exception as e:
                            print(f"Acquisition optimization failed: {e}")
                            # append safe seed for the remaining iterations
                            X = torch.cat([X, safe_seed.repeat((self.iterations - j)*requested_points, 1)], dim=0)
                            Y = torch.cat([Y, safe_seed_y.repeat((self.iterations - j)*requested_points)], dim=0)
                            break

                    if X.shape[0] >= self.iterations + requested_points:
                        break    

            elif self.algo == "random":
                # generate random points
                new_X = torch.rand(self.iterations, dim)
                X = torch.cat([X, new_X], dim=0)
                Y = torch.cat([Y, self.utility(new_X)], dim=0)   
            
            elif self.algo == "sobolRandom":
                # generate sobol points
                sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=random_seed)
                new_X = sobol.draw(self.iterations).squeeze(0)
                X = torch.cat([X, new_X], dim=0)
                Y = torch.cat([Y, self.utility(new_X)], dim=0)
                          
            #save results
            df = pd.DataFrame()
            for d in range(self.tf.objective.dim):
                df[f"X{d}"] = X[:, d].detach().numpy()
            df["y"] = Y.detach().numpy()
            df["crash"] = torch.tensor([1 if y < threshold else 0 for y in df.y])
            
            self.results_dict[i] = df
            i += 1
            random_seed += 1
        self.save_results()
    
    def save_results(self):
        """
        Save the results of the experiment to the results directory.

        Args:
            None

        Returns:
            None
        """
        if not os.path.exists(f"results/{self.tf_path}"):
            os.makedirs(f"results/{self.tf_path}")
        with open(f"results/{self.tf_path}/results_{self.tf_path}_{self.algo}_{self.mode}.pkl", 'wb') as f:
            pickle.dump(self.results_dict, f)  # Save only the dictionary for the current algorithm
    

if __name__ == "__main__":
    # example usage
    for tf_name in ["cosine8", "gp", "branin", "ackley"]:
        #for dim in [2]:
            #dim = 6
            for algo in ["ISE", "crashPBO", "EUBO", "EI", "MES", "ISE", "SafeOpt", "random", "sobolRandom"]:
                print(f"Running {algo}")
                for mode in ["compare_to_best"]:
                    if tf_name == "gp":
                        dim = 8
                        test_experiment = Experiment(algo=algo, tf_name=tf_name, iterations=3, seeds=1, noise=0.01, dim=dim, feasible_percentage=0.5, function_seed=6, mode=mode)
                    else:
                        test_experiment = Experiment(algo=algo, tf_name=tf_name, iterations=30, seeds=1, noise=0.01, feasible_percentage=0.5, function_seed=6, mode=mode)
                    test_experiment.run_experiment()
                    # get length of results_dict
                    print(algo, mode)
                    print(f"Length of results_dict: {len(test_experiment.results_dict)}")
