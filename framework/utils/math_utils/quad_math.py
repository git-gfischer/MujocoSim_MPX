import numpy as np

#Quaternion math (MuJoCo ``qpos`` order ``[w, x, y, z]``)
#    :func:`quat_normalize_wxyz`, :func:`quat_mul_wxyz`, :func:`quat_yaw_wxyz`.

def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    """Unit-length quaternion; returns input unchanged if norm is tiny."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    return q / (n if n > 1e-12 else 1.0)


_quat_normalize_wxyz = quat_normalize_wxyz


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (compose rotations), MuJoCo order ``[w, x, y, z]``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


_quat_mul_wxyz = quat_mul_wxyz


def quat_yaw_wxyz(yaw_rad: float) -> np.ndarray:
    """Pure yaw about world +Z by ``yaw_rad`` radians (wxyz)."""
    h = 0.5 * float(yaw_rad)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_from_rpy_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )

def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    """Extract yaw angle [rad] from a wxyz quaternion."""
    return 2.0 * np.arctan2(float(quat_wxyz[3]), float(quat_wxyz[0]))

def _quat_to_axes(quat_wxyz: np.ndarray):
    """Return the three body axes (X, Y, Z) in world frame from a wxyz quaternion."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    x_axis = np.array([1-2*(y*y+z*z),   2*(x*y+w*z),   2*(x*z-w*y)])
    y_axis = np.array([  2*(x*y-w*z), 1-2*(x*x+z*z),   2*(y*z+w*x)])
    z_axis = np.array([  2*(x*z+w*y),   2*(y*z-w*x), 1-2*(x*x+y*y)])
    return x_axis, y_axis, z_axis