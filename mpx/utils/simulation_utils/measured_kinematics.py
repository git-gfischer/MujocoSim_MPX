"""
What the robot can actually measure: IMU reads, FK on the measured joint state,
and terrain geometry queries.

Two leaks in the v3 dataset motivate this module.

**IMU frame.** v3 logged ``data.qacc[:3]`` — world-frame, gravity-free base
acceleration — under the name ``imu_acc``. A real IMU measures specific force in
its own frame, gravity included, and no base orientation was stored, so the
conversion could not be undone afterwards. The Go2 model already carries
``accelerometer``, ``gyro`` and ``framequat`` sensors on the ``imu`` site, so
:class:`ImuReader` reads them directly. A site-mounted accelerometer in MuJoCo
already includes both gravity and the lever-arm term for a site offset from the
body origin, which is why no manual ``ω × (ω × r)`` correction appears here.

**Forward kinematics on clean state.** v3 injected encoder noise into
``joint_pos`` *after* computing foot positions from the simulator's own geom
poses, so ``foot_pos_base`` was a noise-free readout that no onboard estimator
could reproduce — worth ~0.0043 m of unexplained residual against the logged
joint angles. :class:`MeasuredKinematics` writes the measured joint vector into
a scratch ``MjData`` and runs MuJoCo's own kinematics on it, so the logged foot
pose is exactly the function of the logged joint angles that the robot would
compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import mujoco
import numpy as np

# Terrain and floor geoms live in group 0; robot collision geoms are group 3 and
# visuals group 2, so a ray masked to group 0 sees only the ground.
_TERRAIN_GEOM_GROUP = 0

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix (world ← body) from a wxyz quaternion."""
    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    return flat.reshape(3, 3)


def specific_force_body(a_world: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Specific force as an IMU measures it: ``R_wb^T (a_w - g)``.

    Fallback for models without an ``accelerometer`` sensor; prefer
    :class:`ImuReader`, which reads the sensor and so also picks up the lever arm
    of an offset IMU site.
    """
    rotation = quat_to_rotmat(quat_wxyz)
    return rotation.T @ (np.asarray(a_world, dtype=np.float64).reshape(3) - GRAVITY)


def _sensor_slice(model: mujoco.MjModel, name: str) -> slice | None:
    """Index range of a named sensor in ``data.sensordata``, or None if absent."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        return None
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + int(model.sensor_dim[sensor_id]))


@dataclass
class ImuReader:
    """
    Reads the base IMU from MuJoCo sensors, with a documented fallback.

    ``acc`` is specific force in the IMU frame **including gravity** — a level,
    stationary robot reads ``≈ +9.81 m/s²`` on Z, not zero. ``quat`` is the base
    orientation as wxyz.
    """

    acc_slice: slice | None
    gyro_slice: slice | None
    quat_slice: slice | None
    site_id: int
    site_offset_base: Tuple[float, float, float]
    using_sensors: bool

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        acc_sensor: str = "accelerometer",
        gyro_sensor: str = "gyro",
        quat_sensor: str = "orientation",
        site: str = "imu",
    ) -> "ImuReader":
        acc = _sensor_slice(model, acc_sensor)
        gyro = _sensor_slice(model, gyro_sensor)
        quat = _sensor_slice(model, quat_sensor)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        offset = (
            tuple(float(v) for v in model.site_pos[site_id])
            if site_id >= 0
            else (0.0, 0.0, 0.0)
        )
        return cls(
            acc_slice=acc,
            gyro_slice=gyro,
            quat_slice=quat,
            site_id=int(site_id),
            site_offset_base=offset,  # type: ignore[arg-type]
            using_sensors=acc is not None and gyro is not None,
        )

    def base_quat(self, data: mujoco.MjData) -> np.ndarray:
        """Base orientation, wxyz, world→base."""
        if self.quat_slice is not None:
            return np.asarray(data.sensordata[self.quat_slice], dtype=np.float64)
        return np.asarray(data.qpos[3:7], dtype=np.float64)

    def read(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        """
        Sample the IMU for this step.

        Returns ``acc`` (specific force, IMU frame, gravity included), ``gyro``
        (IMU frame) and ``quat`` (wxyz).
        """
        quat = self.base_quat(data)
        if self.using_sensors:
            acc = np.asarray(data.sensordata[self.acc_slice], dtype=np.float64)
            gyro = np.asarray(data.sensordata[self.gyro_slice], dtype=np.float64)
        else:
            # No sensor in the model: reconstruct from the base state and say so.
            acc = specific_force_body(data.qacc[:3], quat)
            gyro = np.asarray(data.qvel[3:6], dtype=np.float64)
        return {"acc": acc, "gyro": gyro, "quat": quat}

    def to_metadata(self) -> Dict[str, object]:
        return {
            "source": "mujoco_sensors" if self.using_sensors else "reconstructed_from_qacc",
            "site_offset_base": list(self.site_offset_base),
            "convention": "specific force in IMU frame, gravity included; quat wxyz",
        }


def measured_joint_torque(data: mujoco.MjData, n_joints: int) -> np.ndarray:
    """Applied actuator torque after MuJoCo saturation, not the MPC command.

    Prefers ``data.actuator_force`` (post-``forcerange`` clamp). Falls back to
    the actuated-joint slice of ``data.qfrc_actuator`` when the actuator vector
    is empty. Returns a length-``n_joints`` float64 copy.
    """
    n = int(n_joints)
    force = np.asarray(data.actuator_force, dtype=np.float64).reshape(-1)
    if force.size >= n:
        return force[:n].copy()
    return np.asarray(data.qfrc_actuator[6 : 6 + n], dtype=np.float64).copy()


class MeasuredKinematics:
    """
    Forward kinematics and Jacobians evaluated on the **measured** joint state.

    Holds a scratch ``MjData`` whose joint coordinates are overwritten with the
    noisy encoder reading each control step. Running MuJoCo's own
    ``mj_kinematics``/``mj_comPos`` on it yields exactly the foot pose an onboard
    estimator would compute from the same encoders, so ``foot_pos_base`` becomes
    a true function of the logged ``joint_pos``.

    The scratch state carries an identity base pose, so the result is in the
    **body frame**: ``R_base^T (p_foot_w - p_base_w)``. That is the honest frame
    for an ``input`` channel — it needs no attitude estimate at all, only the
    encoders. A gravity-aligned variant is a *different* channel
    (:func:`gravity_align`) and, as an input, must be built from an **estimated**
    attitude rather than ground truth.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        foot_site_names: Sequence[str],
        n_joints: int = 12,
    ) -> None:
        self.model = model
        self.n_joints = int(n_joints)
        self._scratch = mujoco.MjData(model)
        self.site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in foot_site_names
        ]
        missing = [
            name for name, sid in zip(foot_site_names, self.site_ids) if sid < 0
        ]
        if missing:
            raise ValueError(f"Foot sites not found in the model: {missing}")
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, model.nv), dtype=np.float64)

    def __call__(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Foot positions and velocities in the yaw-aligned base frame.

        Both are computed purely from ``joint_pos`` / ``joint_vel``: positions by
        forward kinematics, velocities by ``J(q) @ dq``. Returns flat
        ``(3 * n_feet,)`` arrays in FL FR RL RR order.
        """
        scratch = self._scratch
        # Identity base pose — the result is base-relative, so world pose cancels.
        scratch.qpos[:3] = 0.0
        scratch.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        scratch.qpos[7 : 7 + self.n_joints] = np.asarray(
            joint_pos, dtype=np.float64
        ).reshape(-1)[: self.n_joints]
        scratch.qvel[:6] = 0.0
        scratch.qvel[6 : 6 + self.n_joints] = np.asarray(
            joint_vel, dtype=np.float64
        ).reshape(-1)[: self.n_joints]

        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)

        n_feet = len(self.site_ids)
        positions = np.empty((n_feet, 3), dtype=np.float64)
        velocities = np.empty((n_feet, 3), dtype=np.float64)
        for i, site_id in enumerate(self.site_ids):
            positions[i] = scratch.site_xpos[site_id]
            mujoco.mj_jacSite(self.model, scratch, self._jacp, self._jacr, site_id)
            velocities[i] = self._jacp @ scratch.qvel

        return positions.reshape(-1), velocities.reshape(-1)


def _yaw_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation about Z only, from a wxyz quaternion."""
    rotation = quat_to_rotmat(quat_wxyz)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def gravity_align(
    foot_vectors: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """
    Rotate body-frame foot vectors into the gravity-aligned (yaw-removed) frame.

    ``R_yaw^T R_base`` removes the base yaw and leaves roll and pitch expressed
    against gravity, so the result is the "yaw_base" convention: X forward,
    Y left, Z up.

    Which quaternion goes in decides the channel's role. Pass the **estimated**
    attitude for an ``input`` column and the **true** attitude only for a
    ``privileged`` one: on hardware the attitude comes from an estimator, and
    feeding ground truth through this function would smuggle a privileged signal
    into the network's input vector.
    """
    vectors = np.asarray(foot_vectors, dtype=np.float64).reshape(-1, 3)
    rotation = _yaw_rotation(quat_wxyz).T @ quat_to_rotmat(quat_wxyz)
    return (vectors @ rotation.T).reshape(-1)


def true_foot_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_site_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Noise-free foot kinematics in the **body frame**, from the simulator state.

    This is the noise-free twin of ``foot_pos_base`` / ``foot_vel_base``: same
    frame, same formula, same time alignment, differing only by the sensor noise
    model. v4.0 computed it in the yaw-aligned frame instead, which put a base
    roll/pitch rotation between the two columns — a 1-3 degree offset in clean
    episodes and 23 degrees in a crash — while both claimed ``frame: yaw_base``.
    Any ablation across that pair measured the frame convention, not the noise.
    """
    base_pos = np.asarray(data.qpos[:3], dtype=np.float64)
    base_vel = np.asarray(data.qvel[:3], dtype=np.float64)
    # Body frame: undo the full base orientation, not just its yaw.
    rot_t = quat_to_rotmat(data.qpos[3:7]).T

    n_feet = len(foot_site_names)
    positions = np.empty((n_feet, 3), dtype=np.float64)
    velocities = np.empty((n_feet, 3), dtype=np.float64)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    for i, name in enumerate(foot_site_names):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        positions[i] = rot_t @ (
            np.asarray(data.site_xpos[site_id], dtype=np.float64) - base_pos
        )
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        velocities[i] = rot_t @ (jacp @ data.qvel - base_vel)

    return positions.reshape(-1), velocities.reshape(-1)


def foot_world_positions(
    data: mujoco.MjData,
    foot_site_ids: Sequence[int],
) -> np.ndarray:
    """Foot site positions in world, ``(n_feet, 3)``."""
    return np.asarray(
        [data.site_xpos[int(sid)] for sid in foot_site_ids], dtype=np.float64
    )


class TerrainProbe:
    """
    Terrain elevation under a point, by ray cast.

    Casts downward from well above the query point with the geom-group mask
    restricted to terrain (group 0), so the robot's own collision geoms (group 3)
    cannot be hit. Falls back to ``ground_z`` when the ray misses, which happens
    off the edge of a finite terrain patch.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        ray_start_height: float = 5.0,
        ground_z: float = 0.0,
    ) -> None:
        self.model = model
        self.ray_start_height = float(ray_start_height)
        self.ground_z = float(ground_z)
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[_TERRAIN_GEOM_GROUP] = 1
        self._geomid = np.zeros(1, dtype=np.int32)
        self._down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def height_at(self, data: mujoco.MjData, xy: Sequence[float]) -> float:
        """Terrain elevation [m] directly beneath ``(x, y)``."""
        origin = np.array(
            [float(xy[0]), float(xy[1]), self.ray_start_height], dtype=np.float64
        )
        distance = mujoco.mj_ray(
            self.model, data, origin, self._down,
            self._geomgroup, 1, -1, self._geomid,
        )
        if distance < 0.0:
            return self.ground_z
        return float(origin[2] - distance)

    def heights_under(self, data: mujoco.MjData, points: np.ndarray) -> np.ndarray:
        """Terrain elevation beneath each row of ``points`` ``(n, 3)``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return np.asarray(
            [self.height_at(data, p[:2]) for p in pts], dtype=np.float64
        )
