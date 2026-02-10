# Vision-based Furuta Pendulum based on the Quanser Qube-Servo 2

The **[Quanser Qube-Servo 2](https://www.quanser.com/products/qube-servo-2/)** is a Furuta pendulum that can be used to apply various control and learning algorithms. Encoder values of the rotary arm and the pendulum can be measured and a input voltage can be applied. If the pendulum is used the angle of the rotary arm is limited to less than one turn due to the connection of the encoder.

Introductory literature on the dynamics of the system and the implementation with MATLAB can be found [here](https://www.quanser.com/products/qube-servo-2/) under Simulink Courseware.

We added a camera setup to the Furuta Pendulum for vision-based learning research. Simulations in Mujoco and PyBullet can be used as well. 

<p align="center" float="left">
  <img src="hardware_setup/pictures/overview.jpg" height="200" />
  <img src="hardware_setup/pictures/qube.jpg" height="200" />
</p>

Note: This repository is still under development, it is used and maintained by the Max-Planck-Institute for Intelligent Systems, ICS group and the DSME Institute at RWTH Aachen. Other useful tools and classes might be available sometime later.

Note: In order to not damage the hardware limitations on voltage and safety mechanisms have to be implemented in software (and hardware) as described below.


## Setup
The *Qube-Servo 2* is a furuta pendulum and can be used for various control and learning tasks. One way of communicating with the device is through a Python Interface. Quanser itself does not support a Python interface but Blue River Tech provides one on their [GitHub repository](https://github.com/BlueRiverTech/quanser-openai-driver). This repository is an extension to the development of Blue River Tech. The interface uses the HIL SDK from Quanser and provides an OpenAI Gym like API to communicate with the Qube Servo 2. Furthermore it comes with various implemented controllers and simulations (more of those below). To start with this repository it is recommended to start with the [Blue River Tech repository]((https://github.com/BlueRiverTech/quanser-openai-driver)) and their corresponding [whitepaper](https://arxiv.org/abs/2001.02254) to get an overview of the implemented code.


## Hardware Setup (Vision-based Furuta Pendulum)

Instructions for a robust hardware setup and a standardized lab environment for vision-based experiments on the Furuta Pendulum can be found [here](./hardware_setup/instructions.md).

## Software Setup
The setup is described in the Blue River Techs GitHub repository and was tested in Python 3.6.7. Both repositories, the Blue River Tech repository on GitHub and this one have working ODE simulators but they are different. Both ODE simulation types seem to represent the dynamics of the pendulum and finding a better simulator is a matter of parameter tuning. In addition, this repository includes a new Mujoco simulation. More on the internal simulators can be found below. 

It is recommended to use [conda](https://anaconda.org) as environment and repository management tool.

Some naming conventions in the BlueRiverTech repository are not consistent with the ones in their documentation. The naming conventions described here work for the internal repository and are consistent. 

### Prerequisites
#### Driver installation: HIL SDK from Quanser [Hardware only]
If the **hardware version** should be used, the HIL SDK from Quanser must be installed. At the moment, the hardware version of this repository only works on machines with Linux.
A mirror is available at https://github.com/quanser/hil_sdk_linux_x86_64. Follow the instructions there. If you are using a virtual environment (i.e. conda or virtualenv), it is best to install the driver outside even if the installation should not affect the virtual environment directly.

You also must have _pip_ installed (in most cases conda did this already for you so this step might not be needed if conda is used):
```
$ sudo apt-get install python3-pip 
```
This repository requires a version of the HIL SDK that supports buffer overwrite on overflow (circular buffers). (The mirror posted above supports buffer overflow.)

#### Camera setup (Blackfly Flir)

For vison-based hardware experiments the [Flir Blackfly S](https://www.flir.de/products/blackfly-s-usb3/) is required. The repository can of course be extended to support other cameras as well. The Flir Blackfly S camera is a high speed camera with 522 Hz that allows to run fast sequential experiments. It allows other steps of the control loop to be more resource heavy.

Install the non-python [Spinnaker SDK](https://www.flir.de/products/spinnaker-sdk/) and its requirements for connecting to the Blackfly Camera. Test the camera with their provided GUI first before you continue. After that use the whl file provided (Version must match operating system and your Python version!) by running

```
$ pip install package/blackfly/spinnaker_python-1.23.0.27-cp36-cp36m-linux_x86_64.whl
```

if you use Python 3.6 and your computer fulfills the hardware requirements of the provided version. Alternatively, download the appropriate whl file for your Python version and operating system under https://www.flir.de/products/spinnaker-sdk/.

We found that dependent on the USB port used at our computer the camera had a time delay of over 2 seconds. Make sure to chose a port that does not add a time delay.


#### Repository installation
You can install the driver by cloning and pip-installing the repository this repository. After changing your working directory into the folder run:
```
$ pip install -e .
```

If the needed packages are not installed automatically via `setup.py`, install the following also:
```
$ pip install numpy scipy gym matplotlib numba vpython
```

> **Instructions for using Mujoco:** If the Mujoco simulator should be used, the last installation command changes to `$ pip3 install -e . [mujoco]`. [Mujoco](http://www.mujoco.org/index.html) itself should be installed separately and before the installation of this repository since the [mujoco-py](https://github.com/openai/mujoco-py) package relies on an existing installation of Mujoco. Furthermore, [mujoco-py](https://github.com/openai/mujoco-py) only works on Linux and MacOS operating systems.

If the command `pip` is not available try to use `pip3` instead.

> **_Important:_** Every user should adapt the file [`configuration.py`](./gym_brt/data/config/configuration.py) in [`gym_brt/data/config`](./gym_brt/data/config/) to their own needs and must follow the specific ReadMe in this folder.

Once you have that setup: Run `example_code.py` (ensure the *Qube* is connected to your computer). Touch the pendulum if it does not start moving.
```
$ python example_code.py
```

While using the hardware errors can occurs which often can be solved by unplugging the device or restarting the script.

### Usage
#### OpenAI Gym interface
Usage is very similar to most [OpenAI Gym](https://github.com/openai/gym) environments. Important and different to normal OpenAI Gym is that if you use the hardware classes it requires that you close the environment when finished (See WARNING). This can be done with context managers using a `with` statement:
```python
import gym
from gym_brt.envs import QubeSwingupEnv
from gym_brt.control import QubeFlipUpControl

with QubeSwingupEnv(use_simulator=False, frequency=250) as env:
    controller = QubeFlipUpControl(sample_freq=250, env=env)
    for episode in range(3):
        state = env.reset()
        for step in range(2048):
            action = controller.action(state)
            state, reward, done, info = env.step(action)
```

It can also be closed manually by using explicitly calling `env.close()` in a `try-finally` statement.
 
> **WARNING:** Forgetting to close the environment or incorrectly closing the env leads to several possible issues. The worst including segfaults.
The most common case when the env was not properly closed: you can not reopen the env and you get:

> Error 1073: QERR_BOARD_ALREADY_OPEN: The HIL board is already opened by another process.  The board does not support access from more than one process at the same time.

> The fastest solution (no restart required) is to remove the semaphore of the board:
> 1. Find the location of the semaphore by ls /dev/shm.
>   - It will start with sem.qube_servo2_usb$xxxxxxxxxxxxx (with the 'x's being some alphanumeric sequence).
> 2. Use the rm command to remove it.
>   - Ex: rm /dev/shm/sem.qube_servo2_usb$xxxxxxxxxxxxx

The normal `QubeBaseEnv` should not be initialized solely. There exist multiple subclasses like `QubeSwingupEnv` and `QubeBalanceEnv` that implement specific control problems and their corresponding observation types. Furthermore, this repository contains multiple extensions (wrapper, more subclasses, etc. to change observation type, reward functions and more) which can be found in the directory [gym_brt/envs/reinforcementlearning_extensions](./gym_brt/envs/reinforcementlearning_extensions).

It is recommended to use the `QubeSwingupEnv` or its subclasses since the encoder values of the environment are set to 0 at the beginning which means that either the pendulum in its initial condition is interpreted as a angle of $\pi$ or $0$ respectively. Therefore, inaccuracy in the initial conditions lead to wrong states in the environment. Especially the simulators are just tested for the `QubeSwingupEnv`.

Interacting with the real pendulum or a simulator is nearly identical. The state contains information about the angles of the pendulum alpha and of the rotary arm theta. They can be accessed as following:
```python
theta, alpha, theta_dot, alpha_dot = state
```
The upright equilibrium point is defined as $`\alpha = 0`$ and the downright equilibrium point as $`\alpha = \pm \pi`$ . Looking from the top the positive direction of $`\theta`$ is defined counterclockwise, looking from the front the positive direction of $`\alpha`$ is counterclockwise as well.
The variable `action` refers to the motor voltage and has a limit of $`\pm 18 V`$. This can be additionally limited at 8 V by the environment to prevent hardware damage. 

By manipulating the `use_simulator` variable switching between hardware and simulation can be achieved. The variable `simulation_mode` defines the concrete simulation (i.e. `simulation_mode="mujoco"`).
The two standard controllers (`QubeFlipUpControl`, `QubeHoldControl`) for the pendulum are already implemented in the environment and can be used in a OpenAI Gym style as following:

```python
ctrl_sys = QubeFlipUpControl(env, frequency=frequency)
action = ctrl_sys.action(state)
```

> Usage of the angle $`\alpha`$ is not consistent. Used as above the environment treats $`\alpha = 0`$ as the upward equilibrium point and $`\alpha = \pm \pi`$ as the downward equilibrium point and the implemented controllers use the same convention. Especially if working with the simulator the convention is not guaranteed and the transformation to the encoder values has to be considered.

## Structure
The main part of the repository can be found in the [gym_brt](./gym_brt) directory which includes the OpenAI Gym environments (with the needed data resources), the interface to the hardware and some simple controllers.

The directory [simulator_tuning](./simulator_tuning) contains scripts to optimize the parameters for the different simulations to match their behavior with the hardware.

In [test](./tests) you find simple test scripts and might be a playground to test your implementations.

### Simulator
This repository includes two types of simulators: A simulation calculated with explicit ODEs and simulations which use higher-level software like Mujoco or PyBullet (which also use ODEs but not in such an explicit form). The first type of simulation can be found beside the files of the interface to the hardware version in [gym_brt/quanser/](./gym_brt/quanser). The Mujoco and PyBullet version can be found in the environments directory [gym_brt/envs/simulation](./gym_brt/envs/simulation). The different simulations can be accessed with the constructor arguments `use_simulator` and `simulation_mode` of any Qube instance.

##### ODE
The ODE simulation in [gym_brt/quanser/qube_simulator.py](./gym_brt/quanser/qube_simulator.py) and [gym_brt/quanser/qube_interfaces.py](./gym_brt/quanser/qube_interfaces.py) seems to model the physics quite reasonable but does not provide a precise simulation of the reality. The equations were provided by Peter [peter.martin@quanser.com](peter.martin@quanser.com) from Quanser. They implemented a SIMULINK simulation with a visualization for the Qube Servo2 (accessible in the internal Setup for Matlab/Simulink repository as a zip file: SimulinkSimulationModel_Quanser.zip). The equations seem to be right but the values for the damping coefficients `br` and `bp` had to be negated. Some parameters were tuned heuristically and the simulator is very sensitive to changes in parameters, e.g. the damping coefficients. For further investigation, non-linear and linear modeling approaches for the Furuta pendulum can be found in the [Simulink Courseware on the Quanser website](https://www.quanser.com/products/qube-servo-2/) (dynamic equations contradict each other sometimes, simulator equations in Python and Matlab seem to model the physics quite reasonable).

To get a more precise simulation the parameters have to be adjusted. A team at Bosch invested more time in developing a better physical model, information might be available there.

All simulations can be visualized by using the command `env.render()` in every step but it slows down the simulation noticeably.

##### Mujoco
The Mujoco simulator in [gym_brt/envs/simulation/mujoco/](./gym_brt/envs/simulation/mujoco) uses the Mujoco rigid-body dynamics simulation software to approximate the dynamics of the Qube. This simulation needs an activated [Mujoco installation](http://www.mujoco.org/index.html) and the Python interface [mujoco-py](https://github.com/openai/mujoco-py) from OpenAI. The necessary meshes and configuration files can be found in [gym_brt/data](./gym_brt/data). The simulation works similar to [other robotic environments](https://github.com/openai/gym/tree/master/gym/envs/robotics) of OpenAI Gym.

Since this simulation was build with a simple 3D model provided by Quanser (and not with a CAD model) it is quiet inaccurate as the ODE damping coefficients do not hold anymore and other variables which cannot be measured easily have to be considered additionally (especially the cable influence). Those variables might be tuned by the file [simulator_tuning/simulator_tuning_mujoco.py](./simulator_tuning/simulator_tuning_mujoco.py) if needed and enough time and computational power is available. Further investigation of this problem is definitely needed!

Even if the simulator uses an external software package it is generally faster than the ODE simulation and thus training reinforcement learning agents can be trained with less time. Furthermore rendering is much more realistic than the ODE simulation.

This simulation can be used with the [Simulation Modification Framework]() to apply Domain Randomization. An alternative to this could be OpenAI's [Robogym](https://github.com/openai/robogym) which was not available during the development of the first named framework.

##### PyBullet
A PyBullet simulation is still under development but should be similar to the Mujoco simulation.

##### Other
TBD: Nvidia Isaac, Dart, Unreal + ODE (maybe with help from Quanser)

### Reinforcement Learning Extensions
Extension modules for the `QubeBaseEnv` of the Quanser driver can be found in *[reinforcementlearning_extensions](./gym_brt/envs/reinforcementlearning_extensions)*. The subclasses work as normal OpenAI Gym interfaces like the initially integrated environment classes. I.e. the observed states of the class `QubeBeginDownEnv` are defined as `state = [cos(theta), sin(theta), cos(alpha), sin(alpha), theta_velocity, alpha_velocity]`. 

Furthermore, instead of using subclasses the behavior (observation, reward function, etc.) of an environment can be overwritten with wrapper which should work with most hardware classes and simulated classes (the version where a wrapper does not work should be obvious). Wrapper can be found in _wrapper.py_. Reward functions for this are provided in _rl_reward_functions.py_. 

## Extras (Internal Lab Equipment of DSME)
To change the dynamics of the system, additional weights can be applied. A setup to to change the orientation of the whole Qube is also available.

## Contribution
If you contribute to this repository (this means if you change and want to push something which can affect other projects), you should prepare your contribution correctly:

* Code should be consistent so that it works with the simulation and the hardware integrations of the Qube or it should be obvious that the new integration does only work with specific versions.
* It should be clear that your new implementation does not break other code (i.e. if you add an argument to the constructor of the QubeBaseEnv class, this argument should be optional and the behaviour should be the same without your new integration if the argument is not given).
* The Python code conforms to the [PEP8](https://www.python.org/dev/peps/pep-0008/) standard. Please read and understand it in detail.
* Python files should provide docstrings for necessary public methods which follow [PEP257](https://www.python.org/dev/peps/pep-0257/) docstring conventions and [Google](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstring formatting. A good docstring example can be found [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
* Non-Python files (including XML, HTML, CSS, JS, and Shell Scripts) should follow the [Google Style Guide](https://github.com/google/styleguide) for that language. YAML files should use 2 spaces for indentation.
* If possible, integrate tests.
* Remove `print(...)` statements if not needed.

## Copyright

### License

Copyright © 2017, Max Planck Insitute for Intelligent Systems.

Authors: Steffen Bleher, Moritz Schneider

Released under the [MIT License](LICENSE).
