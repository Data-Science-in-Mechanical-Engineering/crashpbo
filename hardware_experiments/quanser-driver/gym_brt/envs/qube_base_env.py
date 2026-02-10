"""Base code for all OpenAI Gym environments of the Qube.

This base class defines the general behavior and variables of all Qube environments.

Furthermore, this class defines if a simulation or the hardware version of the Qube should be used by initialising the
specific corresponding Qube class at the variable `qube`.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np

from gym import spaces
from gym.utils import seeding

# For other platforms where it's impossible to install the HIL SDK
try:
    from gym_brt.quanser import QubeHardware
except ImportError:
    print("Warning: Can not import QubeHardware in qube_base_env.py")


MAX_MOTOR_VOLTAGE = 18
ACT_MAX = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
# OBS_MAX = [params, alpha, theta_dot, alpha_dot]
OBS_MAX = np.asarray([np.pi / 2, np.pi, np.inf, np.inf], dtype=np.float64)

def normalize_angle(angle):
    return angle/np.pi


class QubeBaseEnv(gym.Env):
    """Base class for all qube-based environments.

    This base class cannot be instantiated since the methods `_reward` and `_isdone` are not defined in this base
    code. For this a subclass like `QubeSwingupEnv` should be used.

    The subclasses of this base class determine the starting point of the task (i.e. pole starts upwards), the end
    points of the task (i.e. a specific angle threshold of the pole in the balance task) and also the reward
    structure of the task.

    Each of those subclasses holds a specific qube instantiation defined at `self.qube`. This qube instantiation
    might be a instance of the hardware interface, the ODE simulation or the Mujoco simulation (PyBullet currently
    not supported).
    Every qube instantiation should implement the same methods to ensure a common behavior so that those classes are
    interchangeable.

    Each subclass of this base class should be used with a `with` statement to ensure that the environment is
    closed correctly:
    ```python
    import gym
    from gym_brt.envs import QubeBeginDownEnv

    with QubeSwingupEnv(use_simulator=False, frequency=250) as env:
        controller = QubeFlipUpControl(sample_freq=frequency, env=env)
        for episode in range(2):
            state = env.reset()
            for step in range(5000):
                action = controller.action(state)
                state, reward, done, info = env.step(action)
    ```
    Instead it can also be closed manually by using explicitly calling `env.close()` inside a try-finally statement.
    """

    def __init__(self, frequency=250, batch_size=2048, use_simulator=False, simulation_mode='ode',
                 integration_steps=10, encoder_reset_steps=int(1e8),):
        """Starting point for the creation of new instances of a Qube (both simulation and hardware).

        Args:
            frequency: Sample frequency
            batch_size: Number of timesteps of a single episode
            use_simulator: Specifies if a simulator should be used instead of the hardware
            simulation_mode: If `use_simulator=True` this specifies the used simulator; either `ode`, `mujoco` or
                            `bullet`; does not affect the hardware classes
            integration_steps: Number of integration steps of the simulation during a single timestep; does not affect
                                the hardware classes
            encoder_reset_steps: Number of timesteps to be done after the hardware encoders should be reinitialized
        """
        self.observation_space = spaces.Box(-OBS_MAX, OBS_MAX, dtype=np.float64)
        self.action_space = spaces.Box(-ACT_MAX, ACT_MAX, dtype=np.float64)
        self.reward_range = (-float(0.), float(1.))

        self._frequency = frequency
        # Ensures that samples in episode are the same as batch size
        # Reset every batch_size steps (2048 ~= 8.192 seconds)
        self._max_episode_steps = batch_size
        self._episode_steps = 0
        self._encoder_reset_steps = encoder_reset_steps
        if not self._encoder_reset_steps:
            self._encoder_reset_steps = int(1e8)
        self._steps_since_encoder_reset = 0
        self._target_angle = 0

        self._theta, self._alpha, self._theta_dot, self._alpha_dot = 0, 0, 0, 0
        self._dtheta, self._dalpha = 0, 0

        # Open the Qube: This means create the appropriate interface (simulation or hardware)
        if use_simulator:
            if simulation_mode == 'ode' or simulation_mode == 'euler':
                # TODO: Check assumption: ODE integration should be ~ once per ms
                from gym_brt.quanser import QubeSimulator
                #integration_steps = int(np.ceil(1000 / self._frequency))
                self.qube = QubeSimulator(
                    forward_model=simulation_mode,
                    frequency=self._frequency,
                    integration_steps=integration_steps,
                    max_voltage=MAX_MOTOR_VOLTAGE,
                )
                self._own_rendering = True
            elif simulation_mode == 'mujoco':
                from gym_brt.envs.simulation.mujoco import QubeMujoco
                #integration_steps = int(np.ceil(1000 / self._frequency))
                self.qube = QubeMujoco(
                    frequency=self._frequency,
                    integration_steps=integration_steps,
                    max_voltage=MAX_MOTOR_VOLTAGE,
                )
                self._own_rendering = False
            elif simulation_mode == 'bullet':
                from gym_brt.envs.simulation.pybullet import QubeBullet
                self.qube = QubeBullet(
                    frequency=self._frequency,
                    integration_steps=integration_steps,
                    max_voltage=MAX_MOTOR_VOLTAGE,
                )
                self._own_rendering = False
            else:
                raise ValueError(f"Unsupported simulation type '{simulation_mode}'. "
                                 f"Valid ones are 'ode', 'mujoco' and 'bullet'.")
        else:
            self.qube = QubeHardware(frequency=self._frequency, max_voltage=MAX_MOTOR_VOLTAGE)
            self._own_rendering = True
        self.qube.__enter__()

        self.seed()
        self._viewer = None

        self._episode_reward = 0

    @property
    def frequency(self):
        return self._frequency

    @property
    def sim(self):
        try:
            return self.qube.sim
        except AttributeError:
            raise AttributeError(f"'{self.qube}' object has no attribute 'sim'")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close(type=type, value=value, traceback=traceback)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        led = self._led()

        action = np.clip(np.array(action, dtype=np.float64), -ACT_MAX, ACT_MAX)
        state = self.qube.step(action, led=led)

        self._dtheta = state[0] - self._theta
        self._dalpha = state[1] - self._alpha
        self._theta, self._alpha, self._theta_dot, self._alpha_dot = state

    def reset(self):
        self._episode_reward = 0
        self._episode_steps = 0
        # Occasionaly reset the encoders to remove sensor drift
        if self._steps_since_encoder_reset >= self._encoder_reset_steps:
            self.qube.reset_encoders()
            self._steps_since_encoder_reset = 0
        action = np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype)
        self._step(action)
        return self._get_state()

    def _reset_up(self):
        self.qube.reset_up()
        action = np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype)
        self._step(action)
        return self._get_state()

    def _reset_down(self):
        self.qube.reset_down()
        action = np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype)
        self._step(action)
        return self._get_state()

    def _get_state(self):
        return np.array(
            [self._theta, self._alpha, self._theta_dot, self._alpha_dot],
            dtype=np.float64,
        )

    def _next_target_angle(self):
        return 0

    def _reward(self):
        raise NotImplementedError

    def _isdone(self):
        raise NotImplementedError

    def _led(self):
        led = [0.0, 0.0, 0.0]
        # if self._isdone():  # Doing reset
        #     led = [1.0, 1.0, 0.0]  # Yellow
        # else:
        #     if abs(self._alpha) > (20 * np.pi / 180):
        #         led = [1.0, 0.0, 0.0]  # Red
        #     elif abs(self._theta) > (90 * np.pi / 180):
        #         led = [1.0, 0.0, 0.0]  # Red
        #     else:
        #         led = [0.0, 1.0, 0.0]  # Green
        return led

    def step(self, action):
        self._step(action)
        state = self._get_state()
        reward = self._reward()
        done = self._isdone()
        self._episode_reward += reward
        info = {
            "theta": self._theta,
            "alpha": self._alpha,
            "theta_dot": self._theta_dot,
            "alpha_dot": self._alpha_dot,
        }

        self._episode_steps += 1
        self._steps_since_encoder_reset += 1
        self._target_angle = self._next_target_angle()

        return state, reward, done, info

    def render(self, mode="human", width=1024, height=1024):
        # TODO: Different modes
        if self._own_rendering:
            if self._viewer is None:
                    from gym_brt.envs.rendering import QubeRenderer
                    self._viewer = QubeRenderer(self._theta, self._alpha, self._frequency)
            return self._viewer.render(self._theta, self._alpha)
        else:
            return self.qube.render(mode=mode, width=width, height=height)

    def close(self, type=None, value=None, traceback=None):
        # Safely close the Qube (important on hardware)
        self.qube.close(type=type, value=value, traceback=traceback)
        if self._viewer is not None:
            self._viewer.close()
