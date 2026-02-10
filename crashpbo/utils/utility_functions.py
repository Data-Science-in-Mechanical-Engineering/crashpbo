from torch.func import grad
import torch
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from botorch.test_functions.synthetic import SyntheticTestFunction, Hartmann, Ackley, Branin, Cosine8, Rosenbrock, SixHumpCamel
from botorch.utils.transforms import unnormalize
from botorch.models import SingleTaskGP
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths  
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
import os
import scipy
import csv
import warnings
warnings.filterwarnings(
    "ignore",
    "To copy construct from a tensor, it is recommended to use sourceTensor.clone",
    module="botorch.test_functions.base",
)
torch.set_default_dtype(torch.float64)

# Define the GPFunction class
class GPFunction(SyntheticTestFunction):
    def __init__(self, dim, seed=30, noise_std=1e-5, negate=False, lengthscale=0.3):
        torch.manual_seed(seed)

        self.dim = dim
        self.seed = seed
        self.lengthscale = lengthscale
        self.continuous_inds = list(range(dim))
        self.discrete_inds   = []
        self.categorical_inds = []
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)], dim=-1)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self._initialize_gp()
        self.sample = self.generate_samples(1)  # Generate a GP sample
        self.lipschitz_constant = 1/self.lengthscale * 1.1
    
    def search_optimal_value(self):
        x = torch.rand(10000, self.dim)
        y = self(x)
        return y.max().item()

    def _initialize_gp(self):
        # Create an empty GP model
        self.kernel = RBFKernel()
        self.likelihood = GaussianLikelihood()

        # Initialize the model without training data
        self.model = SingleTaskGP(train_X=torch.empty((0, self.dim)),
                                train_Y=torch.empty((0, 1)),
                                covar_module=self.kernel,
                                likelihood=self.likelihood,
                                outcome_transform=None,        # <— disable StandardizeY
                                input_transform=None,          # (optional) disable input transforms too
                        )

        # Set hyperparameters for the GP
        self.kernel.lengthscale = self.lengthscale
        self.kernel.outputscale = 1.0
        self.likelihood.noise = torch.tensor([0.1])

    def generate_samples(self, n_samples):
        # Get GP samples using the provided function from your code
        gp_samples = draw_kernel_feature_paths(self.model,sample_shape=torch.Size([1]))
        return gp_samples

    def _evaluate_true(self, X):
        """
        Returns the true value of the GP-sampled function without noise.
        """
        return self.sample(X).squeeze(-1).detach()
    
    def evaluate_true(self, X):
        """
        Returns the true value of the GP-sampled function without noise.
        """
        return self.sample(X).squeeze(-1)
    
    def __call__(self, X):
        return self.sample(X).squeeze(0).detach()

class TestFunction:
    def __init__(self, name, dim=None, seed=30, feasible_percentage=None):
        self.seed = seed
        self.name = name
        if name in ("hartmann", "hartmann6"):
            self.objective = Hartmann(dim=6, negate=True)
            self.threshold = 0
            self.lipschitz_constant = 10
        elif name == "ackley":
            self.objective = Ackley(negate=True)
            self.threshold = -15
            self.lipschitz_constant = 50
        elif name == "branin":
            self.objective = Branin(negate=True)
            self.threshold = -80
            self.lipschitz_constant = 160
        elif name == "cosine8":
            self.objective = Cosine8(negate=True)
            self.threshold = 0
            self.lipschitz_constant = 10
        elif name == "rosenbrock":
            self.objective = Rosenbrock(negate=True)
            self.threshold = -100
            self.lipschitz_constant = 20
        elif name == "sixhumpcamel":
            self.objective = SixHumpCamel(negate=True)
            self.threshold = -15
            self.lipschitz_constant = 20
            
        elif name == "gp":
            if dim is None:
                dim = 2
            self.objective = GPFunction(dim=dim, seed=seed)
            self.lipschitz_constant = self.objective.lipschitz_constant
            self.threshold = 0
        else:
            raise ValueError("Unknown test function")
        if feasible_percentage is not None:
            self.threshold = self.generate_threshold(feasible_percentage)
            self.feasible_percentage = feasible_percentage
        else:
            self.feasible_percentage = "default"

    def __call__(self, X):
        X = unnormalize(X, bounds=self.objective.bounds)
        y = self.objective(X)
        return y
    
    def generate_random_points(self, n):
        """
        Generate n random points with at least one feasible point.

        Parameters:
        - n (int): Total number of random points to generate.

        Returns:
        - X (torch.Tensor): Tensor of shape (n, d) containing the random points.
        """
        bounds = self.objective.bounds  # Assuming bounds is a tensor of shape (2, d)
        d = bounds.shape[1]

        # Initialize lists to store points
        feasible_X = None
        infeasible_X_list = []
        crashes = 0
        # Generate one feasible point
        while feasible_X is None:
            x = torch.rand(1, d)  # Single random point in [0, 1]^d
            y = self(x)
            if y >= self.threshold:
                feasible_X = x
            if crashes == 10:
                raise ValueError("Cannot find feasible point")
                break
            crashes += 1        
        # Generate n - 1 infeasible points
        num_infeasible = n - 1
        batch_size = max(100, num_infeasible)  # Adjust batch size for efficiency

        while len(infeasible_X_list) < num_infeasible:
            x = torch.rand(batch_size, d)
            y = self(x)
            #infeasible_mask = (y < self.threshold).squeeze()
            new_infeasible_X = x
            infeasible_X_list.append(new_infeasible_X)

            # Truncate if we have enough infeasible points
            total_infeasible = sum([xi.shape[0] for xi in infeasible_X_list])
            if total_infeasible >= num_infeasible:
                break

        # Concatenate infeasible points and truncate to desired size
        infeasible_X = torch.cat(infeasible_X_list, dim=0)[:num_infeasible]

        # Combine the feasible and infeasible points
        X = torch.cat([feasible_X, infeasible_X], dim=0)
        # Shuffle the points to ensure random ordering
        indices = torch.randperm(n)
        X = X[indices]

        return X, self(X)
    
    def generate_threshold(self, percentage):
        if self.objective.dim == 2:
            x_grid = torch.linspace(0, 1, 100)
            X = torch.meshgrid(x_grid, x_grid, indexing="ij")
            X_flat = torch.stack([X[0].flatten(), X[1].flatten()], dim=-1)
            y = self(X_flat)
            threshold = torch.quantile(y, 1 - percentage)
            return threshold.item()
        elif self.objective.dim == 1:
            x = torch.linspace(0, 1, 100).unsqueeze(-1)
            y = self(x)
            threshold = torch.quantile(y, 1 - percentage)
            return threshold.item()
        else:
            x = torch.rand(10000*self.objective.dim, self.objective.dim)
            y = self(x)
            threshold = torch.quantile(y, 1 - percentage)
            return threshold.item()
        
    def __str__(self):
        return self.objective.name
    
    def plot_function(self, path=None, return_axes=False):
        """
        Plots the objective function based on its dimensionality.

        This method generates a plot for the objective function. It supports both 1D and 2D functions. 
        For 1D functions, it plots the function values against a grid of x values and includes a threshold line. 
        For 2D functions, it creates a filled contour plot along with a contour line representing the threshold.

        Parameters:
            path (str, optional): The file path where the plot will be saved. If None, the plot will not be saved.
            return_axes (bool, optional): If True, returns the axes object of the plot. Defaults to False.

        Raises:
            ValueError: If the dimensionality of the objective function is greater than 2.

        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
            matplotlib.axes.Axes: The axes object if return_axes is True; otherwise, None.
        """
        if self.objective.dim > 2:
            raise ValueError("Can only plot 2D functions")
        elif self.objective.dim == 1:
            x_grid = torch.linspace(0, 1, 100)
            y = self(x_grid.unsqueeze(-1))
            fig, ax = plt.subplots()
            ax.plot(x_grid, y)
            ax.plot(x_grid, self.threshold * torch.ones_like(x_grid), "--", color="red")
            ax.set_title(f"{self.name} Seed: {self.seed}")
            if path:
                plt.savefig(path)
            if return_axes:
                return fig, ax
        else:
            x_grid = torch.linspace(0, 1, 100)
            X = torch.meshgrid(x_grid, x_grid,indexing="ij")
            X_flat = torch.stack([X[0].flatten(), X[1].flatten()], dim=-1)
            y = self(X_flat)
            y = y.reshape(100, 100)
            fig, ax = plt.subplots()
            contour = ax.contourf(X[0].numpy(), X[1].numpy(), y.detach().numpy())
            plt.colorbar(contour, ax=ax)
            # plot threshold
            ax.contour(X[0].numpy(), X[1].numpy(), y.detach().numpy(), levels=[self.threshold], colors="red")
            ax.set_title(f"{self.name} Seed: {self.seed}")
            if path:
                plt.savefig(path)
            if return_axes:
                return fig, ax
    
    def write_to_file(self, path):
        path = f"{path}/{self.name}_seed_{self.seed}_{self.objective.dim}D_feasiblep_{self.feasible_percentage}.csv"
        # create path if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # make dict out of the data
        data = {"Function": self.name, "Seed": self.seed, "Dimension": self.objective.dim}
        # write dict to file
        with open(path, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in data.items():
                writer.writerow([key, value])
        

        
        

    
    

    
if __name__ == "__main__":
    tf = TestFunction("gp", dim=2, seed=1, feasible_percentage=0.5)
    
    # Example usage for minimization
    test_function = TestFunction(name="branin", dim=2, seed=1, feasible_percentage=0.5)
    test_function.write_to_file("results")
    
    X = torch.tensor([[0.5, 0.5], [1, 1], [0.7, 0.7]])
    X = torch.tensor([[0.5, 0.5]])
    print(test_function(X))
    test_function = TestFunction(name="gp", dim=2, seed=1, feasible_percentage=0.5)
    print(test_function(X))
    
    # Example usage for plotting the function
    test_function.plot_function(path="test_new_branin.png")
   
