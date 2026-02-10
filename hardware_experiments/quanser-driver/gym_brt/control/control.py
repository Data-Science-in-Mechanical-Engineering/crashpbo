from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import linalg
from scipy import signal

from gym_brt import configuration as config

# Set the motor saturation limits for the Aero and Qube
AERO_MAX_VOLTAGE = 15.0


def _convert_state(state):
    state = np.asarray(state)
    if state.shape == (4,):
        return state
    if state.shape == (5,):
        return state
    elif state.shape == (6,):
        # Get the angles
        theta_x, theta_y = state[0], state[1]
        alpha_x, alpha_y = state[2], state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        theta_dot, alpha_dot = state[4], state[5]
        return theta, alpha, theta_dot, alpha_dot
    else:
        raise ValueError(
            "State shape was expected to be one of: (4,1) or (6,1), found: {}".format(
                state.shape
            )
        )


def dampen_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # Alt alpha angle: -pi to +pi, where 0 is the pendulum facing down (at rest)
    alt_alpha = (alpha + 2 * np.pi) % (2 * np.pi) - np.pi
    if np.abs(alpha) < (20.0 * np.pi / 180.0) and np.abs(theta) < (np.pi / 4):
        kp_theta = -2
        kp_alpha = 35
        kd_theta = -1.5
        kd_alpha = 3
        if alpha >= 0:
            action = (
                    -theta * kp_theta
                    + (np.pi - alpha) * kp_alpha
                    + -theta_dot * kd_theta
                    + -alpha_dot * kd_alpha
            )
        else:
            action = (
                    -theta * kp_theta
                    + (-np.pi - alpha) * kp_alpha
                    + -theta_dot * kd_theta
                    + -alpha_dot * kd_alpha
            )
    else:
        action = 0

    action = np.clip(action, -3.0, 3.0)
    return np.array([action], dtype=np.float64)


class Control(object):
    def __init__(self, env=None, action_shape=None, *args, **kwargs):
        if env:
            self.action_shape = env.action_space.sample().shape
        elif action_shape:
            self.action_shape = action_shape
        else:
            raise ValueError('Either env or action_shape must be passed.')

    def action(self, state):
        raise NotImplementedError


class NoControl(Control):
    '''Output motor voltages of 0'''

    def __init__(self, env, *args, **kwargs):
        super(NoControl, self).__init__(env)
        self._action_space = env.action_space

    def action(self, state):
        return 0. * self._action_space.sample()


class RandomControl(Control):
    '''Output motor voltages smapling from the action space (from env).
    '''

    def __init__(self, env, *args, **kwargs):
        super(RandomControl, self).__init__(env)
        self._action_space = env.action_space

    def action(self, state):
        return self._action_space.sample()


class QubeFlipUpControl(Control):
    """Classical controller to hold the pendulum upright whenever the
    angle is within 20 degrees, and flips up the pendulum whenever
    outside 20 degrees.
    """

    start = True

    def __init__(self, env=None, action_shape=(1,), sample_freq=1000, mu=50 , ref_energy =30/1000 , switch_point = 20,
                 **kwargs):
        super(QubeFlipUpControl, self).__init__(env=env, action_shape=action_shape)
        self.sample_freq = sample_freq
        self.kp_theta, self.kp_alpha, self.kd_theta, self.kd_alpha = self._get_optimized_gains()
        self.mu = mu
        self.ref_energy = ref_energy
        self.switch_point = switch_point

    def _get_optimized_gains(self):
        kp_theta, kp_alpha, kd_theta, kd_alpha = self._calculate_lqr(self.sample_freq / 1.2)
        kp_theta, _, kd_theta, _ = self._calculate_lqr(self.sample_freq * 1.1)

        # Parameters tuned by hand
        if self.sample_freq < 120:
            kp_theta = -2.3  # -2.1949203339944114
            kd_theta = -1.13639961510041033
        elif self.sample_freq < 80:
            print("Critical sample frequency! LQR not tuned for frequencies below 80 Hz.")
        return kp_theta, kp_alpha, kd_theta, kd_alpha

    def _flip_up(self, theta, alpha, theta_dot, alpha_dot):
        """Implements a energy based swing-up controller"""
        mu = self.mu  # 50 in m/s/J
        ref_energy = self.ref_energy #30.0 / 1000.0  Er in joules

        max_u = config.QUBE_MAX_VOLTAGE  # Max action is 6m/s^2

        # System parameters
        jp = config.JP
        lp = config.LP
        lr = config.LR
        mp = config.MP
        mr = config.MR
        rm = config.RM
        g = config.G
        kt = config.KT

        multiplicator = config.VOLTAGE_MULTIPLICATOR

        pend_torque = (1 / 2) * mp * g * lp * (1 + np.cos(alpha))
        energy = pend_torque + (jp / 2.0) * alpha_dot * alpha_dot

        u = mu * (energy - ref_energy) * np.sign(-1 * np.cos(alpha) * alpha_dot)
        u = multiplicator * u
        u = np.clip(u, -max_u, max_u)
        torque = (mr * lr) * u
        voltage = (rm / kt) * torque
        return -voltage

    def _dlqr(self, A, B, Q, R):
        """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # first, solve the ricatti equation
        P = np.array(linalg.solve_discrete_are(A, B, Q, R))
        # compute the LQR gain
        K = np.array((linalg.inv(B.T.dot(P).dot(B) + R)).dot(B.T.dot(P).dot(A)))
        return K

    def _calculate_lqr(self, freq=None):
        if freq is None:
            freq = self.sample_freq
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 149.2751, -0.0104, 0], [0, 261.6091, -0.0103, 0]])
        B = np.array([[0], [0], [49.7275], [49.1493]])
        C = np.array([[1, 0, 0, 0]])
        D = np.array([[0]])
        (Ad, Bd, Cd, Dd, dt) = signal.cont2discrete((A, B, C, D), 1 / freq, method='zoh')

        Q = np.eye(4)
        Q[0, 0] = 12
        Q[1, 1] = 5
        Q[2, 2] = 1
        R = np.array([[1]]) * 1

        K = self._dlqr(Ad, Bd, Q, R)
        kp_theta = K[0, 0]
        kp_alpha = K[0, 1]
        kd_theta = K[0, 2]
        kd_alpha = K[0, 3]
        return kp_theta, kp_alpha, kd_theta, kd_alpha

    def _action_hold(self, theta, alpha, theta_dot, alpha_dot):
        # multiply by proportional and derivative gains
        action = \
            theta * self.kp_theta + \
            alpha * self.kp_alpha + \
            theta_dot * self.kd_theta + \
            alpha_dot * self.kd_alpha
        return action

    def action(self, state):

        theta, alpha, theta_dot, alpha_dot = state

        # If pendulum is within 20 degrees of upright, enable balance control
        if np.abs(alpha) <= (self.switch_point * np.pi / 180.0):
            action = self._action_hold(theta, alpha, theta_dot, alpha_dot)
        else:
            action = self._flip_up(theta, alpha, theta_dot, alpha_dot)

        voltages = np.array([action], dtype=np.float64)

        # set the saturation limit to +/- the Qube saturation voltage
        np.clip(voltages, -config.QUBE_MAX_VOLTAGE, config.QUBE_MAX_VOLTAGE, out=voltages)
        assert voltages.shape == self.action_shape
        return voltages


class QubeHoldControl(QubeFlipUpControl):
    """Classical controller to hold the pendulum upright whenever the
    angle is within 20 degrees. (Same as QubeFlipUpControl but without a
    flip up action)
    """

    def __init__(self, env, sample_freq=1000, **kwargs):
        super(QubeHoldControl, self).__init__(
            env, sample_freq=sample_freq)

    def _flip_up(self, theta, alpha, theta_dot, alpha_dot):
        return 0

# # No input
# def zero_policy(state, **kwargs):
#     return np.array([0.0])
#
#
# # Constant input
# def constant_policy(state, **kwargs):
#     return np.array([3.0])
#
#
# # Rand input
# def random_policy(state, **kwargs):
#     return np.asarray([np.random.randn()])
#
#
# # Square wave, switch every 85 ms
# def square_wave_policy(state, step, frequency=FREQUENCY, **kwargs):
#     steps_until_85ms = int(85 * (frequency / 300))
#     state = _convert_state(state)
#     # Switch between positive and negative every 85 ms
#     mod_170ms = step % (2 * steps_until_85ms)
#     if mod_170ms < steps_until_85ms:
#         action = 3.0
#     else:
#         action = -3.0
#     return np.array([action])
#
#
# # Flip policy
# def energy_control_policy(state, **kwargs):
#     state = _convert_state(state)
#     # Run energy-based control to flip up the pendulum
#     params, alpha, theta_dot, alpha_dot = state
#     # alpha_dot += alpha_dot + 1e-15
#
#     """Implements a energy based swing-up controller"""
#     mu = 50.0  # in m/s/J
#     ref_energy = 30.0 / 1000.0  # Er in joules
#
#     max_u = 6  # Max action is 6m/s^2
#     # max_u = 0.85  # Max action is 6m/s^2
#
#     # System parameters
#     jp = 3.3282e-5
#     lp = 0.129
#     lr = 0.085
#     mp = 0.024
#     mr = 0.095
#     rm = 8.4
#     g = 9.81
#     kt = 0.042
#
#     pend_torque = (1 / 2) * mp * g * lp * (1 + np.cos(alpha))
#     energy = pend_torque + (jp / 2.0) * alpha_dot * alpha_dot
#
#     u = mu * (energy - ref_energy) * np.sign(-1 * np.cos(alpha) * alpha_dot)
#     u = np.clip(u, -max_u, max_u)
#
#     torque = (mr * lr) * u
#     voltage = (rm / kt) * torque
#     return np.array([-voltage])
#
#
# # Hold policy
# def pd_control_policy(state, **kwargs):
#     state = _convert_state(state)
#     params, alpha, theta_dot, alpha_dot = state
#     # multiply by proportional and derivative gains
#     kp_theta = -2.0
#     kp_alpha = 35.0
#     kd_theta = -1.5
#     kd_alpha = 3.0
#
#     # If pendulum is within 20 degrees of upright, enable balance control, else zero
#     if np.abs(alpha) <= (20.0 * np.pi / 180.0):
#         action = (
#             params * kp_theta
#             + alpha * kp_alpha
#             + theta_dot * kd_theta
#             + alpha_dot * kd_alpha
#         )
#     else:
#         action = 0.0
#     action = np.clip(action, -3.0, 3.0)
#     return np.array([action])
#
#
# # Flip and Hold
# def flip_and_hold_policy(state, **kwargs):
#     state = _convert_state(state)
#     params, alpha, theta_dot, alpha_dot = state
#
#     # If pendulum is within 20 degrees of upright, enable balance control
#     if np.abs(alpha) <= (20.0 * np.pi / 180.0):
#         action = pd_control_policy(state)
#     else:
#         action = energy_control_policy(state)
#     return action
#
#
# # Square wave instead of energy controller flip and hold
# def square_wave_flip_and_hold_policy(state, **kwargs):
#     state = _convert_state(state)
#     params, alpha, theta_dot, alpha_dot = state
#
#     # If pendulum is within 20 degrees of upright, enable balance control
#     if np.abs(alpha) <= (20.0 * np.pi / 180.0):
#         action = pd_control_policy(state)
#     else:
#         action = square_wave_policy(state, **kwargs)
#     return action
#
#
# # Hold policy
# def pd_tracking_control_policy(state, **kwargs):
#     state = _convert_state(state)
#     params, alpha, theta_dot, alpha_dot, theta_target = state
#     # multiply by proportional and derivative gains
#     kp_theta = -2.0
#     kp_alpha = 35.0
#     kd_theta = -1.5
#     kd_alpha = 3.0
#
#     kp_theta = -2.28
#     kp_alpha = 26.9
#     kd_theta = -1.18
#     kd_alpha = 2.32
#
#     # If pendulum is within 20 degrees of upright, enable balance control, else zero
#     if np.abs(alpha) <= (20.0 * np.pi / 180.0):
#         action = (
#             (params - theta_target) * kp_theta
#             + alpha * kp_alpha
#             + theta_dot * kd_theta
#             + alpha_dot * kd_alpha
#         )
#     else:
#         action = 0.0
#     action = np.clip(action, -3.0, 3.0)
#     return np.array([action])
