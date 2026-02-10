import copy
import io
import json
import math
import os
from itertools import product

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf

# set torch to double precision
torch.set_default_dtype(torch.float64)


class CrashEUBO:
    """
    Class for performing Bayesian optimization using Expected Utility of Best Option (EUBO) with crash constraints.
    The data generation mechanism actually happens in the front end utils/frontend_utils.py file, add_single_comparison(...)
    where the pairwise comparisons are made and passed to this class, this here is just standard EUBO. 

    Attributes:
        problem_dim (int): Dimensionality of the optimization problem.
        data_folder (str): Path to the folder for saving data and stats.
        unscale_fun (callable or None): Function to unscale data points.
        compare_to_best (bool): Whether to compare to the best option.
        warm_start (bool): Whether to initialize from previous stats.
        experiment (object or None): Experiment object containing parameter names.
        constraints (list or None): Inequality constraints for optimization.
    """
    def __init__(self,problem_dim,data_folder,unscale_fun = None,compare_to_best = False, warm_start = False, experiment = None, constraints=None):        
        """
        Initialize the CrashEUBO class.

        Args:
            problem_dim (int): Dimensionality of the optimization problem.
            data_folder (str): Path to the folder for saving data and stats.
            unscale_fun (callable or None): Function to unscale data points.
            compare_to_best (bool): Whether to compare to the best option.
            warm_start (bool): Whether to initialize from previous stats.
            experiment (object or None): Experiment object containing parameter names.
            constraints (list or None): Inequality constraints for optimization.

        Returns:
            None
        """
        self.model = None
        self.problem_dim = problem_dim
        self.bounds = torch.stack([torch.zeros(self.problem_dim), torch.ones(self.problem_dim)])
        self.iteration = 0
        self.data_folder = data_folder
        self.stats_path         = os.path.join(data_folder, "optimizer_stats.json")
        self.stats_path_blocked = os.path.join(data_folder, "optimizer_stats.json.blocked")
        self.compare_to_best = compare_to_best 
        self.unscale_fun     = unscale_fun 
        self.experiment = experiment
        self.constraints = constraints
        # initialize stats:
        self.stats = pd.DataFrame(columns=['estimated optimum', 'estimated optimum unscaled','iteration'])
        
        if warm_start:
            with open(self.stats_path) as json_file:
                stats_dict = json.load(json_file)
                pass
            self.stats = pd.read_json(io.StringIO(stats_dict[-1]["Optimizer Stats"]))
            self.iteration = self.stats.iloc[-1]["iteration"]
 
    def init_and_fit_model(self, X, comp):
        """
        Initialize and fit a pairwise Gaussian process model.

        Args:
            X (torch.Tensor): Input points.
            comp (torch.Tensor): Pairwise comparisons.

        Returns:
            Tuple[PairwiseLaplaceMarginalLogLikelihood, PairwiseGP]: Marginal log likelihood and fitted model.
        """
        X = X.to(dtype = torch.float64)
        self.model = PairwiseGP(
            X,
            comp,
            input_transform=Normalize(d=X.shape[-1]),
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(self.model.likelihood, self.model)
        
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Error in fitting: {e}")
        return mll, self.model
    
    def make_bo_step(self, X, comps, best_x = None):
        """
        Perform a single Bayesian optimization step, here any PBO acquisition function could be implemented. 

        Args:
            X (torch.Tensor): Input points.
            comps (torch.Tensor): Pairwise comparisons.
            best_x (torch.Tensor or None): Best point from previous iterations.

        Returns:
            torch.Tensor: Next points to evaluate.
        """
        NUM_RESTARTS = 10
        RAW_SAMPLES = 512 

        # Set bounds based on X dim
        dim = X.shape[1]
        
        # Refit models
        _, model = self.init_and_fit_model(X, comps)
        
        # Create the acquisition function object
        if self.compare_to_best:
            q = 1  # number of points per query
            acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model, previous_winner=torch.tensor(best_x,dtype=torch.double).unsqueeze(0))
        else:
            q = 2  # number of points per query
            acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        
        # Optimize and get new observation
        #try:
        if self.constraints is None:
            next_X, acq_val = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=q,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
        else: 
            print("Constraints", self.constraints)
            next_X, acq_val = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=q,
                inequality_constraints=self.constraints,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )

        self.iteration += 1

        return next_X

    ################ All functions for interface and logging ################

    def get_posterior_best(self,):
        """
        Get the optimum estimated by the current model for logging. 

        Args:
            None

        Returns:
            torch.Tensor: Estimated optimum point.
        """
        # get optimum estimated by current model
        mean = PosteriorMean(model=self.model)
        x_gpopt, acq_value = optimize_acqf(
            acq_function=mean,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        return x_gpopt.squeeze(0)

    def write_optimizer_stats(self):
        """
        Write optimizer statistics to a JSON file and save posterior plots.

        Args:
            None

        Returns:
            None
        """
        if self.iteration  == 1:
            stats_dict = []
        else:
            with open(self.stats_path ) as json_file:
                stats_dict = json.load(json_file)

        BO_state_name = os.path.join(self.data_folder,"BO_State_iteration_"+str(self.iteration)+".html")
        
        levels = 2

        lower_bound_np = self.bounds[0].numpy()
        upper_bound_np = self.bounds[1].numpy()

        levels = [np.linspace(lb, ub, num=levels) for lb, ub in zip(lower_bound_np, upper_bound_np)]
        full_factorial_design = torch.Tensor(np.array(list(product(*levels))))

        with torch.no_grad():
            posterior = self.model.posterior(full_factorial_design)
            mean = posterior.mean.numpy()
            stddev = posterior.stddev.numpy()

        full_factorial_unscaled = self.unscale_fun(full_factorial_design)

        columns = copy.deepcopy(self.experiment.param_names)
        columns.append('mean')
        columns.append('std')

        data = [full_factorial_unscaled[i].tolist() +  [mean[i][0].tolist()] + [stddev[i].tolist()    ] for i in range(full_factorial_design.shape[0])] 

        posterior_df = pd.DataFrame(data = data, columns= columns)
        
        posterior_csv_name = os.path.join(self.data_folder,"GP_posterior_"+str(self.iteration)+".csv")
        posterior_df.to_csv(posterior_csv_name , index=False)
        
        
        x_est_opt = self.get_posterior_best()
        if not (self.unscale_fun is  None):
            x_est_opt_scaled = self.unscale_fun( x_est_opt)
        else:
            x_est_opt_scaled = x_est_opt

        self.stats.loc[self.iteration] = [x_est_opt.numpy(),x_est_opt_scaled.numpy(),self.iteration]
        
        htmlheight = 400
        if self.problem_dim == 1:   
            htmlwidth = 800
        elif self.problem_dim == 2:  
            htmlwidth = 1200
        elif self.problem_dim > 2:
            htmlwidth = 1200
            n_rows = math.floor(self.problem_dim/3) + 1
            htmlheight = n_rows*400       
        stats_dict.append({"Figure Path":BO_state_name,'config':{'figwidth': htmlwidth,'figheight': htmlheight},"Numeric Stats":{"X Best Unscaled": x_est_opt_scaled.numpy().tolist()},"Optimizer Stats":self.stats.to_json()})

        # plotting if problem_dim == 1
        if self.problem_dim == 1: 
            combined_fig = make_subplots(rows=1, cols=2, subplot_titles=("Posterior Plot", "Progress Plot"))
            posterior_fig = self.plot_posterior(show = False)
            progress_fig = self.plot_x_est_best_progress(show = False, dim = 0)

            # Add traces from the first figure to the combined figure
            for trace in posterior_fig.data:
                combined_fig.add_trace(trace, row=1, col=1)

             # Add traces from the second figure to the combined figure
            for trace in progress_fig.data:
                combined_fig.add_trace(trace, row=1, col=2)
            
            combined_fig.update_yaxes(range=[self.unscale_fun([0.0])[0],  self.unscale_fun([1.0])[0]],title = self.experiment.param_names[0], row=1, col=2)
            combined_fig.update_xaxes(title = "Iteration", row=1, col=2)

        # plotting if problem_dim == 2
        if self.problem_dim == 2: 
            combined_fig = make_subplots(rows=1, cols=3
            ,specs=[[{"type": "surface"}, {"type": "xy"},{"type": "xy"}]])
            posterior_fig = self.plot_posterior(show = False)
            progress_fig_1 = self.plot_x_est_best_progress(show = False, dim = 0)
            progress_fig_2 = self.plot_x_est_best_progress(show = False, dim = 1)

            # Add traces from the first figure to the combined figure
            for trace in posterior_fig.data:
                combined_fig.add_trace(trace, row=1, col=1)

             # Add traces from the second figure to the combined figure
            for trace in progress_fig_1.data:
                combined_fig.add_trace(trace, row=1, col=2)

             # Add traces from the second figure to the combined figure
            for trace in progress_fig_2.data:
                combined_fig.add_trace(trace, row=1, col=3)
            combined_fig.update_yaxes(range=[self.unscale_fun([0.0])[0],  self.unscale_fun([1.0])[0]],title = self.experiment.param_names[0], row=1, col=2)
            combined_fig.update_yaxes(range=[self.unscale_fun([0.0])[1],  self.unscale_fun([1.0])[1]],title = self.experiment.param_names[1], row=1, col=3)
            combined_fig.update_xaxes(title = "Iteration", row=1, col=2)
            combined_fig.update_xaxes(title = "Iteration", row=1, col=3)

        # plotting for higher problem dimensions
        if self.problem_dim > 2: 
            n_rows = math.floor(self.problem_dim/3) + 1
            n_cols = 3
            combined_fig = make_subplots(rows=n_rows, cols=n_cols)
                   
            for i in range(self.problem_dim):
                curr_row = math.floor(i/3)+1
                curr_col = i - math.floor(i/3)*3 +1
                print(curr_row)
                print(curr_col)
                progress_fig = self.plot_x_est_best_progress(show = False, dim = i)
                for trace in progress_fig.data:
                    combined_fig.add_trace(trace, row=curr_row, col=curr_col)
                combined_fig.update_yaxes(range=[self.unscale_fun([0.0])[i],  self.unscale_fun([1.0])[i]],
                                                title = self.experiment.param_names[i], row=curr_row, col=curr_col)
                combined_fig.update_xaxes(title = "Iteration", row=curr_row, col=curr_col)
        # Update layout
        combined_fig.update_layout(width= htmlwidth, height= htmlheight,showlegend=False)

        # Save the figure as an HTML file
        combined_fig.write_html(BO_state_name)

        with open(self.stats_path, 'w') as json_file:
            json.dump(stats_dict,json_file)


    def plot_posterior(self,show = False): 
        """
        Plot the posterior mean and uncertainty.

        Args:
            show (bool): Whether to display the plot.

        Returns:
            plotly.graph_objects.Figure: Posterior plot.
        """
        if self.problem_dim == 1:

            # Generate prediction points
            X_test = torch.linspace( self.bounds[0][0], self.bounds[1][0], 100).unsqueeze(-1)  # Points to predict

            X_plot = self.unscale_fun(X_test)

            # Get posterior mean and covariance (you may need to adjust based on your library version)
            with torch.no_grad():
                posterior = self.model.posterior(X_test)
                mean = posterior.mean.numpy()
                stddev = posterior.stddev.numpy()

            # Create Plotly figure
            fig = go.Figure()

            # Add Posterior Mean Line
            fig.add_trace(go.Scatter(x=X_plot.numpy().flatten(), y=mean.flatten(), mode='lines', name='Posterior Mean'))

            # Add Uncertainty (Standard Deviation) Area
            upper_bound = mean.flatten() + 1.96 * stddev.flatten()
            lower_bound = mean.flatten() - 1.96 * stddev.flatten()
            fig.add_trace(go.Scatter(
                x=np.concatenate([X_plot.numpy().flatten(), X_plot.numpy().flatten()[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Uncertainty Interval'
            ))

            # Update layout for better visualization
            fig.update_layout(xaxis_title="X",
                            yaxis_title="Mean",
                            showlegend=True)
        
        elif self.problem_dim == 2:
            # Show figure
        
            # Create a grid of points in 2D space (from 0 to 1)
            x1 = np.linspace(self.bounds[0][0], self.bounds[1][0], 100)  # Range for first dimension
            x2 = np.linspace(self.bounds[0][1], self.bounds[1][1], 100)  # Range for second dimension
            X1, X2 = np.meshgrid(x1, x2) # Create meshgrid for plotting

            # Flatten the grid for evaluation
            X_test = torch.tensor(np.vstack([X1.ravel(), X2.ravel()]).T).float()

            # Evaluate the GP model without gradient tracking
            with torch.no_grad():
                posterior = self.model.posterior(X_test)
                mean = posterior.mean.numpy().reshape(X1.shape)  # Reshape back to grid shape

            x_unscaled = self.unscale_fun([[x1[i],x2[i]] for i in range(len(x1))]).numpy()
            

            # Create a surface plot using Plotly
            fig = go.Figure(data=[go.Surface(z=mean, x=x_unscaled[:,1], y=x_unscaled[:,1])])

            # Update layout for better visualization
            fig.update_layout(
                title='Predicted Mean from Gaussian Process Model',
                scene=dict(
                    xaxis_title='X1',
                    yaxis_title='X2',
                    zaxis_title='Mean Prediction'
                ),
            )
        
        else:
            fig = None
        
        
        if show:
            fig.show()

        return fig#, fig.axes[0]

    def plot_x_est_best_progress(self,show = False,dim = 0): 
        """
        Plot the progress of the estimated optimum over iterations.

        Args:
            show (bool): Whether to display the plot.
            dim (int): Dimension to plot.

        Returns:
            plotly.graph_objects.Figure: Progress plot.
        """
        fig = go.Figure()

        yvals = [self.stats["estimated optimum unscaled"].to_numpy()[i][dim] for i in range(self.stats.shape[0])]
        
        fig.add_trace(go.Scatter(y= yvals, mode='lines'))
        
        fig.update_layout(xaxis_title="Iterations",
                    yaxis_title="Estimated Optimum",
                    showlegend=True)
        
        # Show figure
        if show:
            fig.show()

        return fig

   