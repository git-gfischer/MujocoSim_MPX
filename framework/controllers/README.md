# Changing the controller

Quadruped simulators get their controller from `make_controller` in `framework/controllers/__init__.py`. The default name is `mpc`. That builds `MujocoMpxConnector`, which runs the MPX model-predictive controller and returns joint torques.

The simulator still owns contacts, the velocity command, terrain, the motor model, and the physics step.

## Default

Leave `name` unset. These are the calls the simulators use:

```python
from framework.controllers import make_controller

controller = make_controller(config, "locomotion")
controller = make_controller(config, "balance", feet=3)
controller = make_controller(config, "balance", feet=4)
```

`make_controller(config, "locomotion")` is the same as `make_controller(config, "locomotion", name="mpc")`.

To make every simulator use another registered controller, change `DEFAULT_CONTROLLER` in `framework/controllers/__init__.py`. To change one call, pass `name=` there.

## Add a controller

1. Add a package `framework/controllers/<name>/` with a MuJoCo connector. It needs `make_carry()`, `step(...)`, and `reset(...)`, the same calls as `TorqueController` in `framework/controllers/mpx/connectors/base.py`. `step` returns `(carry, tau)`.
2. Register it in `make_controller` with a branch next to the `mpc` branch.
3. Select it with `name="<name>"`, or set `DEFAULT_CONTROLLER` to that name.

An IsaacLab policy is the next controller to add. It is not in the tree yet. Until it is registered, any name other than `mpc` raises `ValueError`.
