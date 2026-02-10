# Control

This directory includes simple control policies to swing-up and balance the Qube. Those can be used to test the Hardware (some of them might not be usable in simulation or does not reach the intended behavior).

Furthermore this folder contains calibration methods and classes in `calibration.py` to reset the hardware-based Qube to specific starting points. A wrapper version of this can be found in [`gym_brt/envs/reinforcementlearning_extensions/wrapper.py`](../envs/reinforcementlearning_extensions/wrapper.py).

## Controllers

### Control
Control base class.

### NoControl
Applies no action (sets voltages to zero).

### RandomControl
Applies a random action (samples from the action space).

### AeroControl
Uses PID control to minimize (pitch - reference pitch) + (yaw - reference yaw), where reference is the original position.

### QubeFlipUpControl
Uses a mixed mode controller that uses gains found from LQR to do the flip up when the pendulum angle is over than 20 degrees off upright, and uses PID control and filtering to hold the pendulum upright when under 20 degrees.

### QubeHoldControl
Holding control uses PID with filtering, and outside of 20 degrees use no control.
