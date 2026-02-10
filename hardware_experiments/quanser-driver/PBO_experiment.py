#### qube_base_env.py in gym_brt -> env -> MAX_MOTOR_VOLTAGE = 18 auf 0 #####

import csv
import gym
import time
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import json

from gym_brt.envs import QubeSwingupEnv
from gym_brt.control import QubeFlipUpControl

import os
from datetime import datetime
                
optimization_steps = 100
flag_path = os.path.join("comm/experiment_to_pbo", "experiment_done.txt")
params_path = os.path.join("comm/pbo_to_experiment" , "params.json")
plot_path = os.path.join("comm/experiment_to_pbo" , "plot.html")
data_path = "results/"+datetime.now().strftime('%Y%m%d%H%M%S')+"_quanser_pendulum"

if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"The folder '{data_path}' was created.")

refresh_time = 1

for i in range(optimization_steps): 
    header = ["mu","time","alpha", "theta", "alpha_dot", "theta_dot"]
    df = pd.DataFrame(columns = header)
    while True: 
        if os.path.exists(params_path):
            break
        else:
            time.sleep(refresh_time)
            print("waiting for user feedback!")
        
    with open(params_path, "r") as f:
        params = json.load(f)

    mu = params["mu"]
    ref_energy = params["ref_energy"]
    #ref_energy = 30/1000
    switch_point = params["switch_point"]

    print("Testing mu ", mu)       
    with QubeSwingupEnv(use_simulator=False, frequency=250) as env:
        controller = QubeFlipUpControl(sample_freq=250, env=env, mu=mu, switch_point=switch_point) # TODO: Add ref_energy (20/1000 - 40/1000 und switch_point 10-30 degree)
        state = env.reset()
        t_start = time.time() 
        for step in range(2048):
            
            action = controller.action(state)
            state, reward, done, info = env.step(action)
            alpha_dot = info['alpha_dot']
            theta_dot = info['theta_dot']
            alpha = info['alpha']
            theta = info['theta']
            df.loc[len(df)] = [mu, time.time()-t_start, alpha, theta, alpha_dot, theta_dot]

        df.to_csv(f"{data_path}/experiment_{i}_mu_{mu}_ref_en_{ref_energy}_switch_{switch_point}")

    print(df)

    fig = make_subplots(rows=2, cols=1)
    t = df.time.to_list()
    alpha = df.alpha.to_list()
    theta = df.theta.to_list()
    # Add traces for each subplot
    fig.add_trace(go.Scatter(x=t, y=alpha, mode='lines', name='alpha'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=theta, name = 'theta', mode='lines'), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)  # X-axis label for the last subplot only
    fig.update_yaxes(title_text="Angle (rad)", row=1, col=1)  # Y-axis label for first subplot
    fig.update_yaxes(title_text="Angle (rad)", row=2, col=1)    # Y-axis label for second subplot

    # Update layout for better visualization
    fig.update_layout(height=400, width=600,
                    showlegend=True)

    fig.write_html(plot_path)

    # delete params_path
    os.remove(params_path) 

    with open(flag_path, 'w') as f:
        f.write("Experiment done")
    print(f"Experiment done")