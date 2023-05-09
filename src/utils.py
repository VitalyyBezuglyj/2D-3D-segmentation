import os
import sys
import yaml
import logging
from typing import Tuple
from itertools import count

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import quaternion
from quaternion import as_rotation_matrix

from src.config import get_config
from src.logger import TqdmLoggingHandler

cfg = get_config()

logger = logging.getLogger(__name__)
logger.setLevel(cfg.logging_level) # type: ignore
logger.addHandler(TqdmLoggingHandler())

def _gauss(x, x0=0., sigma=1.):
    return np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

_vec_gauss = np.vectorize(_gauss)

def inverse_gaussian_kernel(distances):
    # distances = np.asarray(distances)
    # print(distances.shape)
    # print(distances)
    # input()
    # return distances
    weighted = []
    for d_vec in distances:
        if len(d_vec)  == 0:
            weighted.append(d_vec)
        else:
            weights = _gauss(d_vec)
            weights *= max(weights)
            weights = 1 - weights
            weighted.append(d_vec * weights)
    return np.asarray(weighted)


def read_calib_file(file_path: str) -> dict:
        """
        Args:
            file_path (str): path to cam calibration file (.y(a)ml).
            
        Returns:
            cam_params (dict): {'P': projection matrix, 'resolution': cam_resolution}
    """
        calib_dict = {}

        with open(file_path, 'r') as f:
            config_cameras = yaml.safe_load(f)
    
        calib_dict['P'] = np.array(config_cameras['left_rect']['P'])
        calib_dict['resolution'] =  np.array(config_cameras['left_rect']['resolution'])
    
        return calib_dict

def build_matrix(x, y, z, q):
    M = np.zeros((4,4))
    if not type(q) is quaternion.quaternion:
         q = np.quaternion(*q) # type: ignore
    M[:3, :3] = as_rotation_matrix(q)
    M[:, 3] = x, y, z, 1
    return M

def inv_T(T):
    T = np.vstack((T,np.asarray([0, 0, 0, 1])))
    T_inv = np.linalg.inv(T)
    return T_inv[:3]

def depths_to_colors(depths: np.ndarray, max_depth: int = 100, cmap: str = "hsv") -> np.ndarray:
    depths /= max_depth
    to_colormap = plt.get_cmap(cmap) # type: ignore
    rgba_values = to_colormap(depths, bytes=True)
    return rgba_values[:, :3].astype(int)

def get_points_labels_by_mask(points: np.ndarray, mask: np.ndarray):
    """
        Args:
            points (ndarray): array of 2D coordinates of projected points with shape (n, 2). Coordinates should match with cam_resolution.

            mask (ndarray): semantic mask in opencv  image format (ndarray)
            
        Returns:
            labels (ndarray): point labels taken from the mask.
    """

    labels = []
    for img_point in points.T: # points.T
        labels.append(mask[img_point[1], img_point[0]] + 2) #! Magic number
        #? Because of Unknown and Dynamic-by-Motion labels added
    
    return np.asarray(labels)

def transform_xyz(T_a_b, xyz):
    """ 
    Borrowed from pypcd lib.
    Transforms an Nx3 array xyz in frame a to frame b
    T_a_b is a 4x4 matrix s.t. xyz_b = T_a_b * xyz_a
    conversely, T_a_b is the pose a in the frame b
    """
    # xyz in frame a, homogeneous
    xyz1_a = np.vstack([xyz.T, np.ones((1, xyz.shape[0]))])
    # xyz in b frame
    xyz1_b = np.dot(T_a_b, xyz1_a)
    xyz_b = np.ascontiguousarray((xyz1_b[:3]).T) #type: ignore
    return xyz_b

def fuse_batch(batch: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Just apply segment_pointcloud func for batch of keyposes and return merged cloud, labels, etc 

        Args:
            batch  ( [dataset[i], dataset[i+1], ...] ): list of tuples with scan pose, scan, seg_masks for cams, such tuple could be obtained using dataset.__get_item__ (dataset[idx]).

        Returns:
            scan_batch (np.ndarray): Array of points (shape (N,3), dtype=np.float32), where N = n1 + n2 + ..., ni - num of points in i-th element of batch.

            point_labels (np.ndarray): Array of labels for points (shape (N,), dtype=np.uint16), where N same as for scan_batch.

            uncertanities (np.ndarray): Array of uncertanities for labels (shape (N,), dtype=np.float32). Calculated as 1/depth, where depth is the distance from the optical center of the camera to the point.
    """
    scan_batch = np.empty((0,3), dtype='float32')
    batch_labels = np.empty((0), dtype='int16')
    batch_uncert = np.empty((0), dtype='float32')

    for pose, scan, scan_labels, scan_uncert in tqdm(batch, "batch processing"):
        if scan is None or scan_labels is None:
            logger.warning("[Fusing batch] Skip pose from batch!")
            continue

        # scan_labels = np.zeros(len(scan), dtype=np.uint16)
        logger.warning(f"Labels shape: {scan_labels.shape}")

        scan_t = transform_xyz(pose, scan)
        scan_batch = np.append(scan_batch, scan_t, axis=0)
        batch_labels = np.append(batch_labels, scan_labels, axis=0)
        batch_uncert = np.append(batch_uncert, scan_uncert, axis=0) # type: ignore

        # Normalize
        # batch_uncert = (batch_uncert - min(batch_uncert)) / (max(batch_uncert) - min(batch_uncert))

    logger.warning(f"Batch shapes:")
    logger.warning(f"Scans: {scan_batch.shape}")
    logger.warning(f"Labels: {batch_labels.shape}")
    logger.warning(f"Confidence: {batch_uncert.shape}")
    return scan_batch, batch_labels, batch_uncert

def stack_size(size=2):
    """Get stack size for caller's frame.
    """
    frame = sys._getframe(size)

    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size - 2