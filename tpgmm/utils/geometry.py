import logging
from typing import List, Union
from numpy import ndarray
import numpy as np


def obtain_pick_and_place_translation(demos: ndarray, frame_idx: List[int] = [0, -1]) -> ndarray:
    """Obtain the vector frames representing the translation with respect to the global frame of the pick and place locations of each demo.

    Args:
        demos (ndarray): Datapoints from demos. Shape: (n_demos, len_demos, 3)
        frame_idx (List[int]): The indices of the demo that correspond to the reference frames of interest. Defaults to [0, -1].

    Returns:
        ndarray: Translation vector with respect to the global frame of the pick and place locations of each demo. Shape: (num_demos, len(frame_idx), 3)
    """

    pick_place_translation = demos[:, frame_idx, :]
    return pick_place_translation


def transform_into_frames(
    trajectories: Union[List[np.ndarray], np.ndarray],
    translations: np.ndarray,
    rotations: np.ndarray,
) -> Union[List[np.ndarray], np.ndarray]:
    """Transforms the given trajectories into local reference frames.

    Args:
        trajectories (Union[List[np.ndarray], np.ndarray]): Acquired trajectories. Shape (num_trajectories, num_points, 3)
        translations (ndarray): Translation into frames. Each trajectory has its own set of frames. Shape (num_trajectories, num_frames, 3)
        rotations (ndarray): Rotation matrix for reference frames. Each trajectory has its own set of frames. Shape (num_trajectories, num_frames, 3, 3)

    Returns:
        Union[List[np.ndarray], np.ndarray]: Transformed trajectories in local reference frames. Shape (num_demos, num_frames, num_points, 3)
    """
    num_trajectories = len(trajectories)
    if len(translations) != num_trajectories or len(rotations) != num_trajectories:
        logging.error(
            "inconsistent number of trajectories across trajectories, translations and rotations"
        )

    _, num_frames, _ = translations.shape
    if num_frames != rotations.shape[1]:
        logging.error("inconsistent number of frames across translations and rotations")

    local_trajectories = []
    # TODO: make this in a single numpy operation
    for trajectory, translation, rotation in zip(trajectories, translations, rotations):
        # affine coordinate transformation: F(x) = Ax + b
        local_trajectory = (np.tile(trajectory, (num_frames, 1, 1)).swapaxes(0, 1)) + translation

        # -> (num_frames, num_points, 3)
        local_trajectory = local_trajectory.swapaxes(0, 1)

        # make individual rotations
        collector = []
        for frame_data, frame_rotation in zip(local_trajectory, rotation):
            collector.append(frame_rotation @ frame_data.T)
        local_trajectory = np.stack(collector)

        local_trajectory = local_trajectory.transpose(0, 2, 1)  # shape (num_frames, num_points, 3)

        local_trajectories.append(local_trajectory)

    if isinstance(trajectories, np.ndarray):
        return np.stack(local_trajectories)

    return local_trajectories
