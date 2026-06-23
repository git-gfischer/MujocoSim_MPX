import mujoco
import numpy as np
from typing import Sequence

def estimate_contacts(
    data: mujoco.MjData,
    contact_geom_ids: Sequence[int],
    dist_threshold: float = 0.0,
) -> np.ndarray:
    """Estimate binary contact state for a set of contact geoms from MuJoCo contacts."""

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

    return contact_state