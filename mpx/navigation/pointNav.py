"""Point-to-point navigation for the locomotion MPC.

This module turns an XY goal in the world into the 7D locomotion command
``[vx, vy, 0, 0, 0, wz, robot_height]`` expected by the MPC (body-frame planar
velocities + yaw rate, see ``KeyboardVelocityCommand.mpc_input``).

Two ways to set the goal are supported:

* **Random goal** – :meth:`PointNavigator.sample_goal` picks a random XY point at
  a configurable distance from the robot. With ``auto_resample`` a fresh goal is
  drawn automatically every time the current one is reached.
* **User-pointed goal** – in the MuJoCo passive viewer, double-click a spot on
  the ground (this re-centres the camera ``lookat`` there) and press the commit
  key. :meth:`PointNavigator.handle_key` / :meth:`PointNavigator.update` read
  ``viewer.cam.lookat`` and use it as the goal.

Typical usage in a passive-viewer loop::

    nav = PointNavigator(robot_height=config.robot_height)
    nav.reset(data.qpos)                       # sample first goal near the robot

    # in key_callback(key):
    nav.handle_key(key)

    # in the controller step:
    command = jnp.asarray(nav.mpc_input(data.qpos))

    # in the render loop (has access to the viewer):
    nav.update(data.qpos, viewer)
"""

from __future__ import annotations

import numpy as np

try:  # glfw is optional (only needed for the interactive key bindings).
    import glfw
except Exception:  # pragma: no cover - headless environments
    glfw = None

from mpx.utils.simulation_utils import sim_utils


def _yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    """Return the world yaw angle [rad] from a MuJoCo ``[w, x, y, z]`` quaternion."""

    w, x, y, z = (float(v) for v in quat_wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PointNavigator:
    """Drive the locomotion MPC toward a 2D goal point.

    The navigator is deliberately framework-light (pure NumPy) so it can be
    dropped into any MuJoCo example with a couple of lines. It owns the goal,
    converts the robot pose into a body-frame velocity command, and can render
    the goal in a passive viewer.
    """

    def __init__(
        self,
        robot_height: float,
        *,
        goal_distance: tuple[float, float] = (2.0, 4.0),
        goal_tolerance: float = 0.25,
        max_speed: float = 0.5,
        max_yaw_rate: float = 0.8,
        kp_yaw: float = 1.5,
        slowdown_radius: float = 0.8,
        auto_resample: bool = True,
        ground_z: float = 0.0,
        seed: int | None = None,
        commit_key: str = "G",
        resample_key: str = "N",
    ):
        self.robot_height = float(robot_height)
        self.goal_distance = (float(goal_distance[0]), float(goal_distance[1]))
        self.goal_tolerance = float(goal_tolerance)
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.kp_yaw = float(kp_yaw)
        self.slowdown_radius = max(float(slowdown_radius), 1e-3)
        self.auto_resample = bool(auto_resample)
        self.ground_z = float(ground_z)

        self._rng = np.random.default_rng(seed)
        self.goal_xy = np.zeros(2, dtype=np.float64)
        self._has_goal = False

        # Viewer rendering state.
        self._goal_geom_id = -1

        # Deferred key actions (key_callback has no access to the viewer).
        self._commit_key = self._keycode(commit_key)
        self._resample_key = self._keycode(resample_key)
        self._commit_pending = False
        self._resample_pending = False

    # ------------------------------------------------------------------ goals
    @staticmethod
    def _robot_xy(qpos: np.ndarray) -> np.ndarray:
        return np.asarray(qpos[:2], dtype=np.float64)

    def set_goal(self, xy) -> None:
        """Set the goal to an explicit world XY point."""

        self.goal_xy = np.asarray(xy, dtype=np.float64).reshape(2)
        self._has_goal = True

    def sample_goal(self, qpos: np.ndarray) -> np.ndarray:
        """Pick a random XY goal at a configured distance from the robot."""

        robot_xy = self._robot_xy(qpos)
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        radius = self._rng.uniform(*self.goal_distance)
        self.goal_xy = robot_xy + radius * np.array([np.cos(angle), np.sin(angle)])
        self._has_goal = True
        return self.goal_xy.copy()

    def set_goal_from_lookat(self, lookat) -> None:
        """Use the viewer camera ``lookat`` point (user pointing) as the goal."""

        self.set_goal(np.asarray(lookat, dtype=np.float64)[:2])

    def reset(self, qpos: np.ndarray) -> None:
        """Sample a fresh random goal relative to the current robot pose."""

        self.sample_goal(qpos)

    def distance_to_goal(self, qpos: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_xy - self._robot_xy(qpos)))

    def reached(self, qpos: np.ndarray) -> bool:
        return self._has_goal and self.distance_to_goal(qpos) <= self.goal_tolerance

    # ---------------------------------------------------------------- command
    def planar_command(self, qpos: np.ndarray) -> np.ndarray:
        """Return the body-frame ``[vx, vy, wz]`` command toward the goal."""

        if not self._has_goal:
            return np.zeros(3, dtype=np.float64)

        robot_xy = self._robot_xy(qpos)
        yaw = _yaw_from_quat(qpos[3:7])

        to_goal = self.goal_xy - robot_xy
        distance = float(np.linalg.norm(to_goal))
        if distance <= self.goal_tolerance:
            return np.zeros(3, dtype=np.float64)

        # Yaw control: rotate to face the goal.
        desired_yaw = float(np.arctan2(to_goal[1], to_goal[0]))
        yaw_error = _wrap_to_pi(desired_yaw - yaw)
        wz = float(np.clip(self.kp_yaw * yaw_error, -self.max_yaw_rate, self.max_yaw_rate))

        # Linear control: project the world direction into the body frame and
        # damp it as the robot approaches the goal or faces away from it.
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        body_dir = np.array(
            [
                cos_y * to_goal[0] + sin_y * to_goal[1],
                -sin_y * to_goal[0] + cos_y * to_goal[1],
            ]
        )
        body_dir /= max(np.linalg.norm(body_dir), 1e-9)

        speed = self.max_speed * min(1.0, distance / self.slowdown_radius)
        heading_gain = max(0.0, float(np.cos(yaw_error)))  # don't push forward facing away
        vx, vy = speed * heading_gain * body_dir
        return np.array([vx, vy, wz], dtype=np.float64)

    def mpc_input(self, qpos: np.ndarray, robot_height: float | None = None) -> np.ndarray:
        """Return the 7D locomotion command consumed by the MPC."""

        vx, vy, wz = self.planar_command(qpos)
        height = self.robot_height if robot_height is None else float(robot_height)
        return np.array([vx, vy, 0.0, 0.0, 0.0, wz, height], dtype=np.float64)

    # ----------------------------------------------------------- interaction
    @staticmethod
    def _keycode(key: str | int | None) -> int | None:
        if key is None or glfw is None:
            return None
        if isinstance(key, int):
            return key
        return getattr(glfw, f"KEY_{key.upper()}", None)

    def handle_key(self, key: int) -> bool:
        """Handle navigation keys from a passive-viewer ``key_callback``.

        Returns ``True`` if the key was consumed. The actual goal update is
        deferred to :meth:`update`, which runs where the viewer is available.
        """

        if self._resample_key is not None and key == self._resample_key:
            self._resample_pending = True
            return True
        if self._commit_key is not None and key == self._commit_key:
            self._commit_pending = True
            return True
        return False

    def update(self, qpos: np.ndarray, viewer=None) -> None:
        """Per-frame housekeeping: apply pending key actions, auto-resample, render.

        Call this once per viewer frame (it needs the viewer for the user-pointed
        goal and for rendering).
        """

        if self._resample_pending:
            self.sample_goal(qpos)
            self._resample_pending = False

        if self._commit_pending:
            if viewer is not None:
                self.set_goal_from_lookat(viewer.cam.lookat)
            self._commit_pending = False

        if self.auto_resample and self.reached(qpos):
            self.sample_goal(qpos)

        self.render(viewer)

    # --------------------------------------------------------------- rendering
    def render(self, viewer=None) -> None:
        """Draw the goal as a sphere in the passive viewer."""

        if viewer is None or not self._has_goal:
            return

        position = np.array([self.goal_xy[0], self.goal_xy[1], self.ground_z + 0.05])
        self._goal_geom_id = sim_utils.render_sphere(
            viewer,
            position=position,
            diameter=2.0 * self.goal_tolerance,
            color=np.array([0.1, 0.8, 0.2, 0.6]),
            geom_id=self._goal_geom_id,
        )

    def overlay_text(self) -> tuple[str, str]:
        """Short viewer text describing the controls and goal."""

        return (
            "Nav: dbl-click ground + G to set goal | N: new random goal",
            f"goal ({self.goal_xy[0]:+.2f}, {self.goal_xy[1]:+.2f})",
        )
