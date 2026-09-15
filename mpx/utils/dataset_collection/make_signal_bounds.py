"""
Build ``signal_bounds.json`` from the robot, not from a collected dataset.

Joint position and torque come from the MJCF ranges (plus a sensing/limit
margin). Joint speed is the actuator capability. Foot position and velocity
are the kinematic envelope of those limits. IMU scales come from the robot's
PI constraint sheet when one exists, otherwise from conservative locomotion
envelopes. Trajectories are never fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import mujoco
import numpy as np

from mpx.config.sim_config.config_sensor_noise import sensor_noise_config
from mpx.utils.dataset_collection.dataset_schema import FOOT_ORDER, JOINT_ORDER

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAVITY = 9.81
SIGNAL_BOUNDS_VERSION = 3
SCHEMA_VERSION = 1

# Extra room past a hard joint stop: MuJoCo limit softness (~0.06 rad observed)
# plus encoder noise. Kept as a robot/sim property, not a dataset percentile.
JOINT_POS_LIMIT_SLACK_RAD = 0.05

# Unitree Go2 rated joint speed. The PI YAML lists 4 rad/s, which is a gait
# preference, not the actuator.
# Boston Dynamics Spot published hip / leg speeds (rad/s).
ROBOTS: Dict[str, Dict[str, Any]] = {
    "go2": {
        "model_path": REPO_ROOT / "mpx" / "data" / "go2" / "go2_mjx.xml",
        "joint_vel_rad_s": {"HAA": 30.1, "HFE": 30.1, "KFE": 30.1},
        "q0": (0.0, 0.9, -1.8) * 4,
        "constraints_yaml": (
            REPO_ROOT
            / "mpx"
            / "addons"
            / "ProprioceptiveImage"
            / "config"
            / "robots"
            / "go2_constrains.yaml"
        ),
    },
    "spot": {
        "model_path": REPO_ROOT
        / "mpx"
        / "data"
        / "boston_dynamics_spot"
        / "spot.xml",
        "joint_vel_rad_s": {"HAA": 8.72, "HFE": 12.57, "KFE": 12.57},
        "q0": (0.0, 1.04, -1.8) * 4,
        "constraints_yaml": None,
    },
}


def _pair(low: float, high: float) -> List[float]:
    lo, hi = float(low), float(high)
    if lo > hi:
        lo, hi = hi, lo
    return [lo, hi]


def _widen(low: float, high: float, margin: float) -> List[float]:
    return _pair(low - margin, high + margin)


def _joint_family(name: str) -> str | None:
    n = name.lower()
    if any(tag in n for tag in ("calf", "_kn", "kfe", "knee")):
        return "KFE"
    if any(tag in n for tag in ("thigh", "_hy", "hfe")):
        return "HFE"
    if any(tag in n for tag in ("hip", "_hx", "haa", "abduction")):
        return "HAA"
    return None


def _hinge_joints(model: mujoco.MjModel) -> List[int]:
    return [
        j
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]


def _actuator_force_range(model: mujoco.MjModel, joint_id: int) -> Tuple[float, float] | None:
    for act in range(model.nu):
        if int(model.actuator_trnid[act, 0]) != joint_id:
            continue
        fr = np.asarray(model.actuator_forcerange[act], dtype=np.float64)
        cr = np.asarray(model.actuator_ctrlrange[act], dtype=np.float64)
        gear = float(np.asarray(model.actuator_gear[act]).reshape(-1)[0])
        if np.isfinite(fr).all() and not np.allclose(fr, 0.0):
            return float(fr[0] * gear), float(fr[1] * gear)
        if np.isfinite(cr).all():
            return float(cr[0] * gear), float(cr[1] * gear)
    return None


def _per_family_joint_limits(
    model: mujoco.MjModel,
) -> Dict[str, Dict[str, Any]]:
    families: Dict[str, Dict[str, Any]] = {
        name: {"pos": [], "tau": [], "qposadr": []} for name in JOINT_ORDER
    }
    for jid in _hinge_joints(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        family = _joint_family(name)
        if family is None:
            continue
        lo, hi = np.asarray(model.jnt_range[jid], dtype=np.float64)
        families[family]["pos"].append((float(lo), float(hi)))
        families[family]["qposadr"].append(int(model.jnt_qposadr[jid]))
        tau = _actuator_force_range(model, jid)
        if tau is not None:
            families[family]["tau"].append(tau)
    for family, data in families.items():
        if not data["pos"]:
            raise ValueError(f"No {family} joints found in the MJCF")
        lows, highs = zip(*data["pos"])
        data["pos_bound"] = (min(lows), max(highs))
        if data["tau"]:
            t_lo, t_hi = zip(*data["tau"])
            data["tau_bound"] = (min(t_lo), max(t_hi))
        else:
            data["tau_bound"] = (-1.0, 1.0)
    return families


def _load_yaml(path: Path | None) -> Mapping[str, Any]:
    if path is None or not path.is_file():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("Constraints", payload)


def _imu_from_constraints(constraints: Mapping[str, Any]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    lin = constraints.get("lin_acc") or {}
    ang = constraints.get("ang_vel_base") or {}
    acc_xy = 15.0
    acc_z = 15.0
    gyro = {"x": 1.5, "y": 1.5, "z": 3.0}
    if lin:
        acc_xy = max(abs(float(lin.get("max_x", acc_xy))), abs(float(lin.get("min_x", acc_xy))))
        acc_xy = max(
            acc_xy,
            abs(float(lin.get("max_y", acc_xy))),
            abs(float(lin.get("min_y", acc_xy))),
        )
        acc_z = max(abs(float(lin.get("max_z", acc_z))), abs(float(lin.get("min_z", acc_z))))
    if ang:
        gyro = {
            "x": max(abs(float(ang.get("max_x", 1.5))), abs(float(ang.get("min_x", 1.5)))),
            "y": max(abs(float(ang.get("max_y", 1.5))), abs(float(ang.get("min_y", 1.5)))),
            "z": max(abs(float(ang.get("max_z", 3.0))), abs(float(ang.get("min_z", 3.0)))),
        }
    acc_noise = 5.0 * sensor_noise_config.imu_acc_std + 5.0 * sensor_noise_config.imu_acc_bias_init_std
    gyro_noise = 5.0 * sensor_noise_config.imu_gyro_std + 5.0 * sensor_noise_config.imu_gyro_bias_init_std
    acc_bounds = {
        "x": _pair(-(acc_xy + acc_noise), acc_xy + acc_noise),
        "y": _pair(-(acc_xy + acc_noise), acc_xy + acc_noise),
        "z": _pair(GRAVITY - (acc_z + acc_noise), GRAVITY + acc_z + acc_noise),
    }
    gyro_bounds = {
        axis: _pair(-(limit + gyro_noise), limit + gyro_noise) for axis, limit in gyro.items()
    }
    return acc_bounds, gyro_bounds


def _foot_targets(model: mujoco.MjModel) -> List[Tuple[str, int]]:
    """Prefer named foot sites; Spot uses geoms ``FL``/``FR``/``HL``/``HR``."""
    targets: List[Tuple[str, int]] = []
    geom_alias = {"RL": "HL", "RR": "HR"}
    for leg in FOOT_ORDER:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
        if site >= 0:
            targets.append(("site", site))
            continue
        geom_name = geom_alias.get(leg, leg)
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom < 0:
            raise ValueError(f"No foot site or geom for {leg}")
        targets.append(("geom", geom))
    return targets


def _apply_q(model: mujoco.MjModel, data: mujoco.MjData, q_hinge: np.ndarray) -> None:
    data.qpos[:3] = 0.0
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    n = min(int(q_hinge.size), model.nq - 7)
    data.qpos[7 : 7 + n] = np.asarray(q_hinge, dtype=np.float64).reshape(-1)[:n]
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def _foot_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: Sequence[Tuple[str, int]],
    dq_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_feet = len(targets)
    pos = np.empty((n_feet, 3), dtype=np.float64)
    vel_cap = np.empty((n_feet, 3), dtype=np.float64)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    for i, (kind, index) in enumerate(targets):
        if kind == "site":
            pos[i] = data.site_xpos[index]
            mujoco.mj_jacSite(model, data, jacp, jacr, index)
        else:
            pos[i] = data.geom_xpos[index]
            mujoco.mj_jacGeom(model, data, jacp, jacr, index)
        # Identity base pose → site/geom xpos is already base-relative enough
        # for an envelope. Velocity capability is Σ |J_joint * dq_max|.
        joint_jac = jacp[:, 6 : 6 + dq_max.size]
        vel_cap[i] = np.abs(joint_jac) @ dq_max
    return pos, vel_cap


def _sample_grid(lows: np.ndarray, highs: np.ndarray, stance: np.ndarray) -> Iterable[np.ndarray]:
    yield stance.copy()
    # Per-leg 3³ grid with the other legs held at stance.
    for leg in range(4):
        sl = slice(leg * 3, leg * 3 + 3)
        for hx in (lows[sl][0], 0.5 * (lows[sl][0] + highs[sl][0]), highs[sl][0]):
            for hy in (lows[sl][1], 0.5 * (lows[sl][1] + highs[sl][1]), highs[sl][1]):
                for hz in (lows[sl][2], 0.5 * (lows[sl][2] + highs[sl][2]), highs[sl][2]):
                    q = stance.copy()
                    q[sl] = (hx, hy, hz)
                    yield q
    # All joints at min / all at max / mixed extremes.
    yield lows.copy()
    yield highs.copy()
    mixed = stance.copy()
    mixed[0::2] = lows[0::2]
    mixed[1::2] = highs[1::2]
    yield mixed


def _foot_envelope(
    model: mujoco.MjModel,
    stance: np.ndarray,
    pos_low: np.ndarray,
    pos_high: np.ndarray,
    dq_max: np.ndarray,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    data = mujoco.MjData(model)
    targets = _foot_targets(model)
    pos_min = np.full(3, np.inf)
    pos_max = np.full(3, -np.inf)
    vel_max = np.zeros(3)
    for q in _sample_grid(pos_low, pos_high, stance):
        _apply_q(model, data, q)
        pos, vel_cap = _foot_state(model, data, targets, dq_max)
        pos_min = np.minimum(pos_min, pos.min(axis=0))
        pos_max = np.maximum(pos_max, pos.max(axis=0))
        vel_max = np.maximum(vel_max, vel_cap.max(axis=0))
    pos_margin = 0.02
    vel_margin = 0.1
    pos_bounds = {
        "x": _widen(pos_min[0], pos_max[0], pos_margin),
        "y": _widen(pos_min[1], pos_max[1], pos_margin),
        "z": _widen(pos_min[2], pos_max[2], pos_margin),
    }
    vel_bounds = {
        "x": _pair(-(vel_max[0] + vel_margin), vel_max[0] + vel_margin),
        "y": _pair(-(vel_max[1] + vel_margin), vel_max[1] + vel_margin),
        "z": _pair(-(vel_max[2] + vel_margin), vel_max[2] + vel_margin),
    }
    return pos_bounds, vel_bounds


def _signal(
    unit: str,
    bounds: Mapping[str, Sequence[float]],
    stance: Mapping[str, float],
) -> Dict[str, Any]:
    return {
        "unit": unit,
        "bounds": {key: [float(v[0]), float(v[1])] for key, v in bounds.items()},
        "stance_value": {key: float(val) for key, val in stance.items()},
    }


def build_signal_bounds(robot: str) -> Dict[str, Any]:
    """Return the global PI bounds dict for ``robot`` (``go2`` or ``spot``)."""
    key = robot.strip().lower()
    if key not in ROBOTS:
        raise ValueError(f"Unknown robot {robot!r}; expected one of {sorted(ROBOTS)}")
    spec = ROBOTS[key]
    model = mujoco.MjModel.from_xml_path(str(spec["model_path"]))
    families = _per_family_joint_limits(model)

    pos_margin = JOINT_POS_LIMIT_SLACK_RAD + 5.0 * sensor_noise_config.joint_pos_std
    vel_margin = 5.0 * sensor_noise_config.joint_vel_std
    tau_scale = 1.0 + 2.0 * sensor_noise_config.joint_torque_scale_err_std
    tau_noise = 5.0 * sensor_noise_config.joint_torque_std

    q0 = np.asarray(spec["q0"], dtype=np.float64).reshape(-1)
    stance_posture = {name: float(q0[i]) for i, name in enumerate(JOINT_ORDER)}

    joint_pos_bounds = {}
    joint_vel_bounds = {}
    torque_bounds = {}
    pos_low = np.empty(12, dtype=np.float64)
    pos_high = np.empty(12, dtype=np.float64)
    dq_max = np.empty(12, dtype=np.float64)
    for leg in range(4):
        for j, name in enumerate(JOINT_ORDER):
            idx = leg * 3 + j
            lo, hi = families[name]["pos_bound"]
            pos_low[idx], pos_high[idx] = lo, hi
            speed = float(spec["joint_vel_rad_s"][name])
            dq_max[idx] = speed
            joint_pos_bounds[name] = _widen(lo, hi, pos_margin)
            joint_vel_bounds[name] = _pair(-(speed + vel_margin), speed + vel_margin)
            t_lo, t_hi = families[name]["tau_bound"]
            t_lo, t_hi = t_lo * tau_scale - tau_noise, t_hi * tau_scale + tau_noise
            torque_bounds[name] = _pair(t_lo, t_hi)

    constraints = _load_yaml(spec.get("constraints_yaml"))
    acc_bounds, gyro_bounds = _imu_from_constraints(constraints)
    foot_pos_bounds, foot_vel_bounds = _foot_envelope(
        model, q0, pos_low, pos_high, dq_max
    )

    zero3 = {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "schema_version": SCHEMA_VERSION,
        "signal_bounds_version": SIGNAL_BOUNDS_VERSION,
        "robot": key,
        "source": "robot_mjcf_and_actuator_capability",
        "model_path": str(Path(spec["model_path"]).relative_to(REPO_ROOT)),
        "stance_posture_rad": stance_posture,
        "signals": {
            "joint_pos": _signal("rad", joint_pos_bounds, stance_posture),
            "joint_vel": _signal(
                "rad/s",
                joint_vel_bounds,
                {name: 0.0 for name in JOINT_ORDER},
            ),
            "joint_torque_measured": _signal(
                "N*m",
                torque_bounds,
                {name: 0.0 for name in JOINT_ORDER},
            ),
            "imu_acc_body": _signal(
                "m/s^2",
                acc_bounds,
                {"x": 0.0, "y": 0.0, "z": GRAVITY},
            ),
            "imu_gyro_body": _signal("rad/s", gyro_bounds, zero3),
            "foot_pos_base": _signal("m", foot_pos_bounds, zero3),
            "foot_vel_base": _signal("m/s", foot_vel_bounds, zero3),
        },
    }


def write_signal_bounds(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def ensure_signal_bounds_file(
    path: str | Path | None = None,
    robot: str = "go2",
) -> Path:
    """Write ``datasets/signal_bounds.json`` from the named robot."""
    from mpx.utils.dataset_collection.signal_bounds import default_signal_bounds_path

    out = Path(path) if path is not None else default_signal_bounds_path()
    return write_signal_bounds(out, build_signal_bounds(robot or "go2"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write datasets/signal_bounds.json from robot MJCF limits."
    )
    parser.add_argument("--robot", default="go2", choices=sorted(ROBOTS))
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "datasets" / "signal_bounds.json"),
        help="Output JSON path",
    )
    args = parser.parse_args(argv)
    path = ensure_signal_bounds_file(args.out, args.robot)
    print(f"wrote {path} for robot={args.robot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
