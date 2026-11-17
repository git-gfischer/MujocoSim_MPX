"""Logistic foot-stance estimator from Camurri et al., RA-L 2017, Sec. IV.

The probability that leg ``l`` is in reliable stance is

    P(S_l = 1 | f_z) = 1 / (1 + exp(-beta * f_z - beta0))

with ``f_z`` the normal component of the ground reaction force. The paper
declares contact when that probability reaches 0.5. This is a comparison
signal: the controller and the dataset labels keep the MuJoCo contact.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from framework.config.sim_config.config_camurri_contact import (
    CamurriContactConfig,
    camurri_contact_config,
)
from framework.estimators.quad_contact_estimation import (
    estimate_foot_grf,
    foot_contact_normals,
)


def stance_probability(fz, beta: float, beta0: float) -> np.ndarray:
    """Eq. (7). ``fz`` is in newtons, one value per foot."""
    z = beta * np.asarray(fz, dtype=np.float64) + beta0
    z = np.clip(z, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def estimate_camurri_contacts(
    grf,
    cfg: CamurriContactConfig | None = None,
    normals=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stance probability and binary contact from world-frame GRF (n, 3).

    Returns ``(probability, stance)`` with stance 1 where P exceeds the threshold.
    """
    cfg = camurri_contact_config if cfg is None else cfg
    force = np.asarray(grf, dtype=np.float64).reshape(-1, 3)
    if cfg.force_axis == "contact_normal":
        if normals is None:
            raise ValueError("contact_normal needs one unit normal per foot")
        axis = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        fz = np.sum(force * axis, axis=1)
    else:
        fz = force[:, 2]
    prob = stance_probability(fz, cfg.beta, cfg.beta0)
    stance = (prob > cfg.threshold).astype(np.float32)
    return prob, stance


def upward_grf(grf) -> np.ndarray:
    """Flip ``estimate_foot_grf`` into the upward load Eq. (7) expects.

    That estimator points into the ground. On the Go2 a stance foot is about
    −50 N of world Z, which the positive-beta logistic would read as swing.
    """
    return -np.asarray(grf, dtype=np.float64)


def camurri_from_sim(
    model,
    data,
    contact_geom_ids: Sequence[int],
    cfg: CamurriContactConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (7) on the MuJoCo GRF of each foot geom. Zeros when disabled."""
    cfg = camurri_contact_config if cfg is None else cfg
    n = len(list(contact_geom_ids))
    if not cfg.enabled:
        zeros = np.zeros(n, dtype=np.float32)
        return zeros, zeros
    grf = upward_grf(estimate_foot_grf(
        model, data, contact_geom_ids, dist_threshold=cfg.dist_threshold,
    ))
    normals = None
    if cfg.force_axis == "contact_normal":
        normals = foot_contact_normals(
            model, data, contact_geom_ids, dist_threshold=cfg.dist_threshold,
        )
    return estimate_camurri_contacts(grf, cfg, normals)


def format_camurri(probability, stance) -> str:
    prob = np.asarray(probability, dtype=np.float64).reshape(-1)
    bits = np.asarray(stance, dtype=np.float64).reshape(-1)
    p = " ".join(f"{v:.2f}" for v in prob)
    s = " ".join(str(int(v)) for v in bits)
    return f"Camurri: [{s}]  P=[{p}]"


def camurri_sphere_rgba(stance, ground_truth) -> np.ndarray:
    """Green when Camurri matches ground truth, red when it does not.

    Same RGBA as the PI contact sphere in ``PIInferenceOverlay``.
    """
    if int(stance) == int(ground_truth):
        return np.array([0.15, 0.85, 0.2, 0.95])
    return np.array([0.95, 0.12, 0.12, 0.95])


class CamurriContactViewer:
    """One sphere per foot, stacked on the PI contact sphere.

    The PI sphere is centered 0.045 m above the foot with diameter 0.032 m.
    This one sits on top of it. Green means the Camurri bit matches ground
    truth; red means it does not.
    """

    _ABOVE_FOOT = np.array([0.0, 0.0, 0.081])
    _DIAMETER = 0.028

    def __init__(self) -> None:
        self._geom_ids = [-1, -1, -1, -1]

    def draw(self, viewer, foot_pos, camurri_stance, contact_gt) -> None:
        if viewer is None or camurri_stance is None or contact_gt is None:
            return
        from framework.utils.simulation_utils.sim_utils import render_sphere

        feet = np.asarray(foot_pos, dtype=np.float64).reshape(-1, 3)
        stance = np.asarray(camurri_stance, dtype=int).reshape(-1)
        truth = np.asarray(contact_gt, dtype=int).reshape(-1)
        n = min(4, feet.shape[0], stance.shape[0], truth.shape[0])
        for i in range(n):
            self._geom_ids[i] = render_sphere(
                viewer,
                position=feet[i] + self._ABOVE_FOOT,
                diameter=self._DIAMETER,
                color=camurri_sphere_rgba(stance[i], truth[i]),
                geom_id=self._geom_ids[i],
            )
