# Quanser Interfaces
Quanser wrapper is a hardware wrapper that allows the Qube Servo 2 to be used directly in OpenAI Gym.
Quanser ODE-Simulator simulates the Qube Servo 2 only.

## Quanser Wrapper
This directory includes the interface to the hardware and exploits the official hardware API from Quanser to communicate with the board of the Qube (might also work with other hardware from Quanser i.e. Aero).

It is an implementation of a Python wrapper around Quanser's C-based HIL SDK written in Cython. All changes to this are recompiled if you install the _gym_brt_ package with Cython installed.

## Qube ODE-Simulator
The file [qube_simulator.py](./qube_simulator.py) contains the code for ODE simulation.
Similar to the Mujoco simulation it is intended to be used as a simulator directly in the Qube environments instead of using the real hardware.
