from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from gym_brt.envs import QubeBeginDownEnv, convert_state
from gym_brt.control import QubeFlipUpControl
from gym_brt.data.config.configuration import FREQUENCY


def optimize_controller():
    frequency = FREQUENCY

    with QubeBeginDownEnv(use_simulator=False, frequency=frequency) as env:
        controller = QubeFlipUpControl(sample_freq=frequency, env=env)
        for episode in range(2):
            state = env.reset()
            state, reward, done, info = env.step(np.array([0], dtype=np.float64))
            # print("Started episode {}".format(episode))
            for step in range(1000):
                # print("step {}.{}".format(episode, step))
                action = controller.action(convert_state(state))
                state, reward, done, info = env.step(action)


if __name__ == '__main__':
    optimize_controller()