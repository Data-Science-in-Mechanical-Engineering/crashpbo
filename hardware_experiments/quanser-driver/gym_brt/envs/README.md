# OpenAI Gym Environments

These are the core integrated environments utilizing OpenAI Gym.

The base class for all environments can be found in `qube_base_env.py`, whereas all subclasses derived from this environment are in the residual python files and in the directory [reinforcementlearning_extensions](./reinforcementlearning_extensions). The base class can and should not be instantiated but rather the subclasses like `QubeSwingupEnv`.

Mujoco and PyBullet implementations of the Qube can be found in [simulation](./simulation) and can be used interchangeably with the ODE simulation by correctly specifying the arguments `use_simulator` and `simulation_mode` in the constructor of the Qube instance.

The directory [rendering](./rendering) contains the rendering of the ODE simulation.


## Environments

### QubeBaseEnv
The base class for Qube environments.
Info:

- Reset starts the pendulum from the bottom (at rest).
- Has no reward function.


### QubeBeginDownEnv
Info:

- Reset starts the pendulum from the bottom (at rest).
- The task is to flip up the pendulum and hold it upright.
- Episode ends once the theta angle is greater than 90 degrees.
- Reward is a function of the angles theta (arm angle) and alpha (pendulum), and the alpha angular velocity.
    - Encourages the the arm to stay centered, the pendulum to stay upright, and to stay stationary.


### QubeBeginUprightEnv
Info:

- Reset starts the pendulum from the top (flipped up/inverted).
- The task is to hold the pendulum upright.
- Episode ends once the alpha angle is greater the 20 degrees or theta angle is greater than 90 degrees.
- Reward is a function of the angles theta (arm angle) and alpha (pendulum), and the alpha angular velocity.
    - Encourages the the arm to stay centered, the pendulum to stay upright, and to stay stationary.
