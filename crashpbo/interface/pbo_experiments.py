import os
import json
import time
import shutil
import numpy as np
import torch
from botorch.utils import transforms
from abc import ABC, abstractmethod
from scipy.integrate import odeint
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set torch to double precision
torch.set_default_dtype(torch.float64)

# Parent class for all pbo experiments
class PboExperiment(ABC):
    """
    Abstract base class for Preferential Bayesian Optimization (PBO) experiments.

    Attributes:
        data_folder (str): Folder to save experiment data.
        param_names (list): Names of the parameters for the experiment.
        htmlwidth (int): Width of the HTML plots.
        htmlheight (int): Height of the HTML plots.
        warm_start (bool): Whether to initialize from previous experiment data.
    """
    def __init__(self, data_folder, param_names, htmlwidth=None, htmlheight=None, warm_start=False):
        """
        Initialize the PboExperiment class.

        Args:
            data_folder (str): Folder to save experiment data.
            param_names (list): Names of the parameters for the experiment.
            htmlwidth (int, optional): Width of the HTML plots. Defaults to None.
            htmlheight (int, optional): Height of the HTML plots. Defaults to None.
            warm_start (bool, optional): Whether to initialize from previous experiment data. Defaults to False.
        """
        self.experiment_counter = 0
        self.data_folder = data_folder # folder to save the data to
        self.experiment_dict = None
        # Setting up all paths
        self.db_path                     = os.path.join(data_folder, "data.json")
        self.experiments_db_path         = os.path.join(data_folder, "experiment_db.json")
        self.db_path_blocked             = os.path.join(data_folder, "data.json.locked") 
        self.experiments_db_path_blocked = os.path.join(data_folder, "experiment_db_blocked.json")
        self.param_names = param_names
        self.htmlwidth = htmlwidth 
        self.htmlheight = htmlheight

        if warm_start:
            with open(self.experiments_db_path) as json_file:
                self.experiment_dict = json.load(json_file)
            self.experiment_counter = len(self.experiment_dict["experiments"])
        else:
            for f in os.listdir("comm/experiment_to_pbo"):
                os.remove(os.path.join("comm/experiment_to_pbo", f))
            for f in os.listdir("comm/pbo_to_experiment"):
                os.remove(os.path.join("comm/pbo_to_experiment", f))
            
    @abstractmethod
    def run_experiment(self, params):
        """
        Abstract method to run an experiment.

        Args:
            params (list): Parameters for the experiment.

        Returns:
            int: Experiment ID.
        """
        pass
    
    def save_experiment(self,params,fig,numeric_results_json):
        """
        Save experiment results to the database.

        Args:
            params (list): Parameters used in the experiment.
            fig (plotly.graph_objects.Figure or str): Plot of the experiment results.
            numeric_results_json (dict): Numeric results of the experiment.

        Returns:
            int: Experiment ID.
        """
        id = self.experiment_counter
        self.experiment_counter += 1

        if id == 0:
            experiment_dict = {'config':{'figwidth': self.htmlwidth,'figheight': self.htmlheight},'experiments':[]}
        else:
            with open(self.experiments_db_path) as json_file:
                experiment_dict = json.load(json_file)
        if not(fig is None):
            experiment_html_name = os.path.join(self.data_folder,"experiment_results_id_"+str(id)+".html")
        else:
            experiment_html_name = ""

        experiment_dict["experiments"].append({"Experiment Id":id,"Numeric Results":numeric_results_json,"Figure Path":experiment_html_name,"Params":params.tolist()})
        
        self.experiment_dict = experiment_dict 
        with open(self.experiments_db_path, 'w') as json_file:
            json.dump(experiment_dict,json_file)

        # Save the figure as an HTML file
        if not(fig is None):
            if isinstance(fig,str):
                shutil.copy2(fig, experiment_html_name)    
            else:
                fig.write_html(experiment_html_name)
                print(f"Plot saved as {experiment_html_name}")
        
        return  id

    def unscale(self,X):
        """
        Unscale normalized parameters to their original range.

        Args:
            X (torch.Tensor or list): Normalized parameters.

        Returns:
            torch.Tensor: Unscaled parameters.
        """
        if not X is torch.Tensor:
            X = torch.Tensor(X)
        X_unnorm = transforms.unnormalize(X, torch.Tensor([self.lb, self.ub])) 
        return X_unnorm
        
    def scale(self,X):
        """
        Scale parameters to a normalized range.

        Args:
            X (torch.Tensor or list): Original parameters.

        Returns:
            torch.Tensor: Normalized parameters.
        """
        if not X is torch.Tensor:
            X = torch.Tensor(X)
        X_norm = transforms.normalize(X,  torch.Tensor([self.lb, self.ub])) 
        return X_norm

    def get_problem_def_json(self):
        """
        Get the problem definition as a JSON object.

        Returns:
            dict: Problem definition containing parameter details.
        """
        problem_def_dict = {"Parameters": []}
        for i in range(self.dim):
            problem_def_dict["Parameters"].append({
                "name": self.param_names[i],
                "lower_bound": self.lb[i],
                "upper_bound": self.ub[i],
                "init_first": self.init_first[i],
                "init_second": self.init_second[i]
            })
        return problem_def_dict

    def set_problem_def_json(self,problem_def):
        """
        Set the problem definition from a JSON object.

        Args:
            problem_def (dict): Problem definition containing parameter details.
        """
        params = problem_def["Parameters"]
        self.param_names = []
        self.lb = []
        self.ub = []
        self.init_first = []
        self.init_second = []
        self.dim = len(params)
        for i in range(self.dim):
            param = params[i]
            self.param_names.append(param["name"])
            self.lb.append(param["lower_bound"])
            self.ub.append(param["upper_bound"])
            self.init_first.append(param["init_first"])
            self.init_second.append(param["init_second"])
            


class PboStepResponse(PboExperiment):
    """
    PBO experiment for step response optimization.

    Attributes:
        dim (int): Dimensionality of the problem.
        lb (list): Lower bounds of the parameters.
        ub (list): Upper bounds of the parameters.
        init_first (list): Initial values for the first experiment.
        init_second (list): Initial values for the second experiment.
        m (float): Mass (kg).
        k (float): Spring constant (N/m).
        b (float): Damping coefficient (Ns/m).
        initial_conditions (list): Initial conditions for the simulation.
        simulation_time (float): Total simulation time (s).
        sample_time (float): Sampling time (s).
        input_constraints (list): Constraints on the input force.
        constraints (None): Placeholder for more complex input constraints.
    """
    def __init__(self,data_folder,warm_start = False):
        """
        Initialize the PboStepResponse class.

        Args:
            data_folder (str): Folder to save experiment data.
            warm_start (bool, optional): Whether to initialize from previous experiment data. Defaults to False.
        """
        param_names    = ["log10(kp)","log10(ki)"]
        htmlheight = 400
        htmlwidth = 600
        super().__init__(data_folder,param_names,htmlwidth = htmlwidth, 
                            htmlheight = htmlheight,warm_start = warm_start)
        self.dim = 2
        self.lb  =  np.log10([0.05, 0.01]).tolist()   # lower bound of the parameters
        self.ub  = np.log10([5, 3]).tolist()  # upper bound of the parameters
        self.init_first  = [0.0, -1.0] 
        self.init_second = [-1.0, -0.5] 

        self.m = 1.0  # mass (kg)
        self.k = 0.5  # spring constant (N/m)
        self.b = 1.0   # damping coefficient (Ns/m)
        self.initial_conditions =  [0, 0]
        self.simulation_time = 20
        self.sample_time = 0.1
        self.input_constraints = [-30, 30]
        self.constraints = None

    def run_experiment(self,params):
        """
        Run the step response experiment.

        Args:
            params (list): Parameters for the experiment.

        Returns:
            int: Experiment ID.
        """
        sim_data = self.run_simulation(params)
        fig = self.plot_episode(sim_data)
        numeric_results_json = {"None":0}
        return super().save_experiment(params,fig,numeric_results_json) 


    def run_simulation(self,params):
        """
        Simulate the step response based on the given parameters.

        Args:
            params (list): Parameters for the simulation.

        Returns:
            dict: Simulation data containing time, position, velocity, input force, and reference position.
        """
        #initial condition
        xkm1 = self.initial_conditions

        # time points
        t = np.arange(0, self.simulation_time, self.sample_time)
        n = int(self.simulation_time / self.sample_time)

        # step input
        u = np.zeros_like(t)
        # change to 2.0 at time = 5.0

        # store solution
        x1 = np.empty_like(t)
        x2 = np.empty_like(t)

        kp = 10**params[0]
        
        #ki = 0.2
        ki = 10**params[1]
        

        # record initial conditions
        x1[0] = xkm1[0]
        x2[0] = xkm1[1]
        integral_position_error = 0.0
        
        reference = np.zeros_like(t)
        reference[10:-1] =    2.0  # first step response
        # solve ODE
        for i in range(1, n):
            # span for next time step
            tspan = [t[i - 1], t[i]]

            # solve for next step
            xk = odeint(self.model, xkm1, tspan, args=(u[i-1],))

            # store solution for plotting
            x1[i] = xk[1][0]
            x2[i] = xk[1][1]
 
            # next initial condition
            xkm1 = xk[1]

            position_error = xkm1.copy()[0] - reference[i]
            integral_position_error += position_error*self.sample_time
            
                            # next control input
            next_u = -(kp* position_error + ki*integral_position_error)
            u[i] = np.clip(next_u, self.input_constraints[0], self.input_constraints[1])

        sim_data = {"t":t,"x1":x1,"x2":x2,"u":u,"x1ref":reference}
        return sim_data


    def plot_episode(self,sim_data,show_plot = False):
        """
        Plot the simulation results.

        Args:
            sim_data (dict): Simulation data containing time, position, velocity, input force, and reference position.
            show_plot (bool, optional): Whether to display the plot. Defaults to False.

        Returns:
            plotly.graph_objects.Figure: Plot of the simulation results.
        """
        t   = sim_data["t"]
        x1  = sim_data["x1"]
        x2  = sim_data["x2"]
        x1ref  = sim_data["x1ref"]
        u   = sim_data["u"] 
        
        fig = make_subplots(rows=2, cols=1)

        # Add traces for each subplot
        fig.add_trace(go.Scatter(x=t, y=x1, mode='lines', name='Actual'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=x1ref, mode='lines', name='Reference'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=u, name = "input", mode='lines'), row=2, col=1)

        fig.update_xaxes(title_text="Time (s)", row=2, col=1)  # X-axis label for the last subplot only
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)  # Y-axis label for first subplot
        fig.update_yaxes(title_text="Force (N)", row=2, col=1)    # Y-axis label for second subplot

        # Update layout for better visualization
        fig.update_layout(height=self.htmlheight, width=self.htmlwidth,
                        showlegend=True)

        if show_plot:
            # Show the figure
            fig.show()  

        return fig


    def model(self,x, t, force):
        """
        Define the system model for the simulation.

        Args:
            x (list): State variables (position and velocity).
            t (float): Time.
            force (float): Input force.

        Returns:
            list: Derivatives of the state variables.
        """
        position, velocity = x

        m = self.m    # mass (kg)
        k = self.k    # spring constant (N/m)
        b = self.b    # damping coefficient (Ns/m)


        dxdt = [velocity, -k/m * position - b/m * velocity + force]
        return dxdt
       
class PboQuanserPendulum(PboExperiment):
    """
    PBO experiment for optimizing the Quanser Pendulum. 
    This is an example for a hardware experiment, where the parameters are exchanged with the hardware experiment via a file system.

    Attributes:
        dim (int): Dimensionality of the problem.
        lb (list): Lower bounds of the parameters.
        ub (list): Upper bounds of the parameters.
        init_first (list): Initial values for the first experiment.
        init_second (list): Initial values for the second experiment.
        refresh_time (int): Time interval for refreshing the experiment.
        param_exchange_path (str): Path for exchanging parameters with the experiment.
        flag_path (str): Path for the flag indicating experiment completion.
        figure_path (str): Path for saving the experiment plot.
    """
    
    def __init__(self,data_folder,warm_start = False):
        """
        Initialize the PboQuanserPendulum class.

        Args:
            data_folder (str): Folder to save experiment data.
            warm_start (bool, optional): Whether to initialize from previous experiment data. Defaults to False.
        """
        param_names    = ["mu",  "ref_energy", "switch_point"]
        htmlheight = 1000
        htmlwidth = 600
        super().__init__(data_folder,param_names,htmlwidth = htmlwidth, 
                            htmlheight = htmlheight,warm_start = warm_start)
        self.dim =  3 # problem dimension = 1
        self.lb  =  [30, 20/1000 , 1] # lower bound of the parameters
        self.ub   = [100, 100/1000, 90] # upper bound of the parameters
        self.init_first  = [43, 0.05, 36]
        self.init_second = [51, 0.03, 9]
        self.refresh_time = 1
        self.param_exchange_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"comm", "pbo_to_experiment" , "params.json")
        self.flag_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"comm", "experiment_to_pbo" , "experiment_done.txt")
        self.figure_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"comm", "experiment_to_pbo" , "plot.html")
        

    def run_experiment(self,params):
        """
        Run the Quanser Pendulum experiment.
        For hardware experiments, this function will save the parameters to a file and wait for the experiment to finish and write a result file.

        Args:
            params (list): Parameters for the experiment.

        Returns:
            None
        """
        params_json = {}
        for i in range(len(self.param_names)):
            params_json[self.param_names[i]] = round(params[i].tolist(),4)

        with open(self.param_exchange_path, 'w') as json_file:
            json.dump(params_json,json_file)

        while True: 
            if os.path.exists(self.flag_path):
                break
            else:
                time.sleep(self.refresh_time)
                print("waiting for experiment done file")
        
        os.remove(self.flag_path)

        numeric_results_json = {"objVal":0}
        return super().save_experiment(params,self.figure_path,numeric_results_json)

if __name__ == '__main__':
    exp = PboStepResponse('dummy_folder')
    sim_data = exp.run_simulation([-0.2691398859024048, -0.6456849575042725])
    fig = exp.plot_episode(sim_data,show_plot = True)

    

