from crashpbo.interface.pbo_experiments import PboQuanserPendulum
import numpy as np
import os
import datetime

weight = "5g"

seed = 1
np.random.seed(seed)
ft = "%Y%m%dT%H%M%S"
tz = datetime.timezone.utc
t = datetime.datetime.now(tz=tz).strftime(ft)

data_folder = f"results/random/{t}_random_quanser_seed_{seed}_weight_{weight}"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"The folder '{data_folder}' was created.")
    
experiment = PboQuanserPendulum(data_folder = data_folder) 

iterations = 20

# generate random values in bounds: 
for i in range(iterations):
    params = np.random.uniform(experiment.lb, experiment.ub)
    print("i", i)
    print(params)
    plot = experiment.run_experiment(params)
