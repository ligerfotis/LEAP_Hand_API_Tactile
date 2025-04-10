import numpy as np
import random


def generate_random_middle_pose(current_pose, safe_bounds, indices=[4, 5, 6, 7]):
    """
    Generates a new pose for the middle finger within the provided bounds.

    Args:
         current_pose (np.array): The current 16-dimensional pose.
         safe_bounds (dict): Mapping from joint index to (min_angle, max_angle) in radians.
         indices (list): The indices for the middle finger joints.

    Returns:
         np.array: A new pose vector with updated middle finger joint angles.
    """
    new_pose = current_pose.copy()
    for idx in indices:
        min_angle, max_angle = safe_bounds.get(idx, (0, np.pi))
        rand_deg = random.uniform(np.rad2deg(min_angle), np.rad2deg(max_angle))
        new_pose[idx] = np.deg2rad(rand_deg)
    return new_pose
