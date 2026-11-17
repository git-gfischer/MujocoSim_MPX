import jax
from jax import numpy as jnp
from functools import partial
from mujoco.mjx._src import math
from jax.scipy.spatial.transform import Rotation

def terrain_orientation(liftoff_pos,Ryaw):
    """
    Calculates the terrain orientation based on the liftoff positions.
    Args:
        liftoff_pos: The liftoff positions of the legs.
        Ryaw: The yaw rotation matrix.

    Returns:
        The terrain orientation quaternion.
    """
    # Calculate the vectors between the legs
    vec_front_back = (liftoff_pos[:3] + liftoff_pos[3:6] - liftoff_pos[6:9] - liftoff_pos[9:12])/2
    # vec_left_right = (liftoff_pos[:3] + liftoff_pos[6:9] - liftoff_pos[3:6] - liftoff_pos[9:12])/2
    #DO NOT ADJUST THE ROLL
    vec_left_right = Ryaw@jnp.array([0,1,0])
    # Compute the normal vector to the plane
    normal_vector = jnp.cross(vec_front_back, vec_left_right)

    # Normalize the vectors
    vec_front_back = vec_front_back / math.norm(vec_front_back)
    vec_left_right = vec_left_right / math.norm(vec_left_right)
    normal_vector = normal_vector / math.norm(normal_vector)

    # Create the rotation matrix
    rotation_matrix = Rotation.from_matrix(jnp.stack([vec_front_back, vec_left_right, normal_vector], axis=1))

    # Convert the rotation matrix to a quaternion
    quat = rotation_matrix.as_quat()

    return jnp.roll(quat,1)




