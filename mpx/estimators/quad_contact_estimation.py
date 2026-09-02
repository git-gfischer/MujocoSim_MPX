import mujoco
import numpy as np
from typing import Sequence

def estimate_contacts(
    data: mujoco.MjData,
    contact_geom_ids: Sequence[int],
    dist_threshold: float = 0.005,
    *,
    foot_positions: np.ndarray | None = None,
    ground_z: float = 0.0,
    height_threshold: float = 0.025,
) -> np.ndarray:
    """Estimate binary contact state for a set of contact geoms.

    Primary source is MuJoCo's ``data.contact`` manifold (``dist <= dist_threshold``).
    When ``foot_positions`` is provided, any foot still reading open whose world Z is
    at or below ``ground_z + height_threshold`` is treated as in contact.  This catches
    stance feet that the contact solver reports with a small positive separation margin.
    """

    contact_geom_ids = np.asarray(contact_geom_ids, dtype=np.int32)
    contact_state = np.zeros(contact_geom_ids.shape[0], dtype=np.float32)
    geom_to_contact = {int(geom_id): idx for idx, geom_id in enumerate(contact_geom_ids)}

    for idx in range(data.ncon):
        contact = data.contact[idx]
        if contact.dist > dist_threshold:
            continue
        geom1 = geom_to_contact.get(int(contact.geom1))
        geom2 = geom_to_contact.get(int(contact.geom2))
        if geom1 is not None:
            contact_state[geom1] = 1.0
        if geom2 is not None:
            contact_state[geom2] = 1.0

    if foot_positions is not None:
        foot_xyz = np.asarray(foot_positions, dtype=np.float64).reshape(-1, 3)
        n = min(foot_xyz.shape[0], contact_state.shape[0])
        z_contact = float(ground_z) + float(height_threshold)
        for i in range(n):
            if contact_state[i] < 0.5 and foot_xyz[i, 2] <= z_contact:
                contact_state[i] = 1.0

    return contact_state


def estimate_foot_grf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_geom_ids: Sequence[int],
    dist_threshold: float = 0.005,
) -> np.ndarray:
    """Estimate world-frame ground reaction force [N] at each foot contact geom.

    Returns shape ``(n_feet, 3)`` in FL, FR, RL, RR order (world frame: X, Y, Z).
    Forces are summed over all active contacts involving each foot geom.
    """
    contact_geom_ids = np.asarray(contact_geom_ids, dtype=np.int32)
    n_feet = contact_geom_ids.shape[0]
    grf = np.zeros((n_feet, 3), dtype=np.float64)
    geom_to_idx = {int(geom_id): idx for idx, geom_id in enumerate(contact_geom_ids)}

    wrench = np.zeros(6, dtype=np.float64)
    for k in range(data.ncon):
        contact = data.contact[k]
        if contact.dist > dist_threshold:
            continue

        mujoco.mj_contactForce(model, data, k, wrench)
        force_local = wrench[:3]
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = frame @ force_local

        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        idx1 = geom_to_idx.get(g1)
        idx2 = geom_to_idx.get(g2)
        # mj_contactForce: force on body1 (geom1) exerted by body2.
        if idx1 is not None:
            grf[idx1] += force_world
        if idx2 is not None:
            grf[idx2] -= force_world

    return grf

#----------------------------------------------------------------------------
def print_contact_friction(
    model: mujoco.MjModel,
    foot_geom_names: Sequence[str],
    floor_geom_name: str = "floor",
) -> None:
    """Print condim and friction coefficients for the floor and each foot geom.

    Call once after loading the model to verify contact settings.

    Example output::

        [friction] floor   condim=3  friction=[1.000 0.005 0.000]
        [friction]   FL    condim=6  friction=[0.800 0.020 0.010]
        [friction]   FR    condim=6  friction=[0.800 0.020 0.010]
        [friction]   RL    condim=6  friction=[0.800 0.020 0.010]
        [friction]   RR    condim=6  friction=[0.800 0.020 0.010]
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
    if floor_id >= 0:
        f = model.geom_friction[floor_id]
        print(f"[friction] {floor_geom_name:<8} condim={model.geom_condim[floor_id]}  "
              f"friction=[{f[0]:.3f} {f[1]:.3f} {f[2]:.3f}]")
    else:
        print(f"[friction] WARNING: geom '{floor_geom_name}' not found in model")

    for name in foot_geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            f = model.geom_friction[gid]
            print(f"[friction]   {name:<6} condim={model.geom_condim[gid]}  "
                  f"friction=[{f[0]:.3f} {f[1]:.3f} {f[2]:.3f}]")
        else:
            print(f"[friction]   WARNING: geom '{name}' not found in model")