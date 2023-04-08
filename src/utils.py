import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

import quaternion
from quaternion import as_rotation_matrix

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
        labels.append(mask[img_point[1], img_point[0]] + 1) #? because of Unknown label added
    
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
