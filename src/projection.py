# Adapted from https://github.com/alexmelekhin/project_lidar_to_cam_ros1

from typing import Tuple
import logging

import quaternion
import numpy as np

from src.config import get_config, Config
from src.utils import build_matrix

cfg = get_config()

logger = logging.getLogger(__name__)
logger.setLevel(cfg.logging_level) # type: ignore

def project_scan_to_camera(
    points: np.ndarray, proj_matrix: np.ndarray, cam_res: Tuple[int, int], tf_config: Config, return_mask=False
):
    lidar2left_cam_q = np.quaternion(*tf_config.lidar2cam_q) # type: ignore # w, x, y, z
    lidar2left_cam_t = np.asarray(tf_config.lidar2cam_t) # type: ignore

    lidar2cam_T = build_matrix(*lidar2left_cam_t, lidar2left_cam_q) # type: ignore # left_cam

    
    if points.shape[0] != 3:
        logger.debug(f"Transposing pointcloud {points.shape} -> {points.T.shape}")
        points = points.T
    
    points = np.vstack((points, np.ones((1, points.shape[1]))))

    points = lidar2cam_T @ points

    if points.shape[0] == 3:
        points = np.vstack((points, np.ones((1, points.shape[1]))))
    
    if len(points.shape) != 2 or points.shape[0] != 4:
        raise ValueError(
            f"Wrong shape of points array: {points.shape}; expected: (4, n), where n - number of points."
        )
    if proj_matrix.shape != (3, 4):
        raise ValueError(f"Wrong proj_matrix shape: {proj_matrix}; expected: (3, 4).")
    in_image = points[2, :] > 0
    depths = points[2, :] # colors

    uvw = np.dot(proj_matrix, points)
    uv = uvw[:2, :]
    w = uvw[2, :]
    uv[0, :] /= w
    uv[1, :] /= w
    in_image = (uv[0, :] >= 0) * (uv[0, :] < cam_res[0]) * (uv[1, :] >= 0) * (uv[1, :] < cam_res[1]) * in_image
    if return_mask:
        return uv[:, in_image].astype(int), depths, in_image
    return uv[:, in_image].astype(int), depths