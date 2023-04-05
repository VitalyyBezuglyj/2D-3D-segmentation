
## Adopted from https://github.com/PRBonn/vdbfusion 

import os
import yaml
import glob
import logging

import cv2
import numpy as np
from typing import Union

from src.utils import read_calib_file, build_matrix

class MIPT_Campus_Dataset:
    def __init__(self, config):
        """Simple DataLoader to provide a ready-to-run example.

        Heavily inspired in PyLidar SLAM
        """
        # Config stuff
        self.cfg = config

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.cfg.logging_level)

        self.zed_left_dir = os.path.join(self.cfg.dataset_root_dir,
                                         'unpacked_'+ self.cfg.sequence, 
                                         'zed_node_left_image_rect_color_compressed')
        
        self.zed_right_dir = os.path.join(self.cfg.dataset_root_dir, 
                                          'unpacked_'+ self.cfg.sequence, 
                                          'zed_node_right_image_rect_color_compressed')
        
        self.realsense_dir = os.path.join(self.cfg.dataset_root_dir, 
                                          'unpacked_realsense_'+ self.cfg.sequence, 
                                          'realsense_back_color_image_raw_compressed')
        
        self.velodyne_dir = os.path.join(self.cfg.dataset_root_dir, 
                                         'unpacked_'+ self.cfg.sequence, 
                                         'velodyne_points')
        
        self._getitem_set = {'images': False,
                             'scan': True,
                             'pose': True,
                             'segmentation_masks': True}

        # Read stuff
        #? self.zed_calibration = read_calib_file(os.path.join(self.cfg.dataset_root_dir, 
        #?                                                          "zed_calib.yml"))
        #? self.realsense_calibration = read_calib_file(os.path.join(self.cfg.dataset_root_dir, 
        #?                                                                "realsense_calib.yml"))

        self.load_poses(os.path.join(self.cfg.dataset_root_dir, 
                                     self.cfg.sequence + '_lol_map', 
                                     'gt_poses.tum'))

        self.zed_left_ts = self.load_ts(os.path.join(self.zed_left_dir, "timestamps_demo.txt"))
        self.logger.debug(f"Zed ts {self.zed_left_ts.shape}")

        self.zed_right_ts = self.load_ts(os.path.join(self.zed_right_dir, "timestamps_demo.txt"))
        self.realsense_ts = self.load_ts(os.path.join(self.realsense_dir, "timestamps_demo.txt"))
        self.scan_ts = self.load_ts(os.path.join(self.velodyne_dir, "timestamps_demo.txt"))

        self.zed_left_files = sorted(glob.glob(self.zed_left_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.zed_left_files)} Zed left images")

        self.zed_right_files = sorted(glob.glob(self.zed_right_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.zed_right_files)} Zed right images")

        self.realsense_files = sorted(glob.glob(self.realsense_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.realsense_files)} Realsense images")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "/clouds/*.bin"))
        self.logger.debug(f"Found {len(self.scan_files)} Scan files")

        #TODO Add right segmentation masks (wider viewing angle)
        self.zed_seg_files = sorted(glob.glob(self.zed_left_dir + "/anno_demo/*.png"))
        self.logger.debug(f"Found {len(self.zed_seg_files)} Zed left masks")

        self.realsense_seg_files = sorted(glob.glob(self.realsense_dir + "/anno_demo/*.png"))
        self.logger.debug(f"Found {len(self.zed_left_files)} Realsense masks")


        assert(len(self.zed_left_ts) == len(self.zed_left_files))
        # self.logger.error(f"Len files: {len(self.zed_right_files)} -- Len ts: {len(self.zed_right_ts)}")
        # assert(len(self.zed_right_ts) == len(self.zed_right_files))
        assert(len(self.scan_ts) == len(self.scan_files))

        self.big_ts_diff = False


    def __getitem__2(self, idx):
        if idx < 0:
            idx = len(self.scan_files) + idx
        elif idx > len(self.scan_files):
            idx -= len(self.scan_files)

        pose_idx = self.align_by_ts(idx, self.poses_ts) # type: ignore

        if self._getitem_set['images'] or self._getitem_set['segmentation_masks']:
            realsense_idx = self.align_by_ts(idx, self.realsense_ts)
            zed_left_idx =  self.align_by_ts(idx, self.zed_left_ts)
            zed_right_idx = self.align_by_ts(idx, self.zed_right_ts)
        
        return_list = []

        if self._getitem_set['scan']:
            return_list.append(self.poses(pose_idx)) # type: ignore
        
        if self._getitem_set['scan']:
            return_list.append(self.scans(idx))

        if self._getitem_set['images']:
            return_list.append(self.read_zed_img(zed_left_idx)) # type: ignore
            return_list.append(self.read_zed_img(zed_right_idx)) # type: ignore
            return_list.append(self.read_realsense_img(realsense_idx)) # type: ignore
        
        if self._getitem_set['segmentation_masks']:
            return_list.append(self.read_zed_mask(zed_left_idx)) # type: ignore
            #? return_list.append(return_list.append(self.read_zed_mask(zed_left_idx))) # type: ignore
            return_list.append( self.read_realsense_mask(zed_left_idx)) # type: ignore

        return tuple(return_list)
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self.poses) + idx
        elif idx > len(self.poses):
            idx -= len(self.poses)

        scan_idx = self.align_by_ts(idx, self.scan_ts)

        if self._getitem_set['images'] or self._getitem_set['segmentation_masks']:
            realsense_idx = self.align_by_ts(idx, self.realsense_ts)
            zed_left_idx = self.align_by_ts(idx, self.zed_left_ts)
            zed_right_idx = self.align_by_ts(idx, self.zed_right_ts)

        self.logger.debug(f"Pose_idx: {idx}")
        self.logger.debug(f"scan_idx: {scan_idx}")
        self.logger.debug(f"zed_idx: {zed_left_idx}") # type: ignore
        self.logger.debug(f"rs_idx: {realsense_idx}") # type: ignore
        # input()
        return_list = []

        if self._getitem_set['pose']:
            return_list.append(self.poses[idx])
        
        if self._getitem_set['scan']:
            return_list.append(self.scans(scan_idx))

        if self._getitem_set['images']:
            return_list.append(self.read_zed_img(zed_left_idx)) # type: ignore
            #? return_list.append(self.read_zed_img(zed_right_idx)) # type: ignore
            return_list.append(self.read_realsense_img(realsense_idx)) # type: ignore
        
        if self._getitem_set['segmentation_masks']:
            return_list.append(self.read_zed_mask(zed_left_idx)) # type: ignore
            #? return_list.append(return_list.append(self.read_zed_mask(zed_left_idx))) # type: ignore
            return_list.append( self.read_realsense_mask(realsense_idx)) # type: ignore

        return tuple(return_list)

    # def get_pose(self, idx):
    #     return self.poses[idx]

    def align_by_ts2(self, lidar_idx: int, ts_list: np.ndarray) -> np.intp:
        lidar_ts = self.scan_ts[lidar_idx]

        timestamps_diff = np.asarray(abs(ts_list - abs(lidar_ts))) #-ts0
        idx = np.argmin(timestamps_diff)
        if timestamps_diff[idx]> self.cfg.max_ts_delay:
            self.logger.info(f'Big diff: {timestamps_diff[idx]} -- idx: {idx} -- scan_id: {lidar_idx}') 
            self.logger.info(f"Scan ts {lidar_ts}")
            self.logger.info(f"Found ts {ts_list[idx]}")

        return idx
    
    def align_by_ts(self, pose_idx: int, ts_list: np.ndarray) -> Union[np.intp, None]:
        pose_ts = self.poses_ts[pose_idx]

        timestamps_diff = np.asarray(abs(ts_list - abs(pose_ts))) #-ts0
        idx = np.argmin(timestamps_diff)
        if timestamps_diff[idx]> self.cfg.max_ts_delay:
            self.logger.warning(f'Big diff: {timestamps_diff[idx]} -- idx: {idx} -- pose_id: {pose_idx}') 
            self.logger.warning(f"Pose ts {pose_ts}")
            self.logger.warning(f"Found ts {ts_list[idx]}")
            return None

        return idx

    def __len__(self):
        return len(self.poses)

    def scans(self, idx):
        if idx is None:
            return None
        return self.read_point_cloud(idx, self.scan_files[idx])

    def read_point_cloud(self, idx: int, apply_pose=False):
        points = np.fromfile(self.scan_files[idx], dtype=np.float32).reshape((-1, 4))[:, :-1]
        in_range_idx = np.all(np.logical_and(self.cfg.lidar.lower_bounds <= points, 
                                             points <= self.cfg.lidar.upper_bounds), axis=1)
        points = points[in_range_idx]
        #TODO Add poses
        # points = self.transform_points(points, self.poses[idx]) if apply_pose else points
        return points

    def read_zed_mask(self, idx: np.intp):
        if idx is None:
            return None
        return cv2.imread(self.zed_seg_files[idx], cv2.IMREAD_UNCHANGED)
    
    def read_realsense_mask(self, idx: np.intp):
        if idx is None:
            return None
        return cv2.imread(self.realsense_seg_files[idx], cv2.IMREAD_UNCHANGED)
    
    def read_zed_img(self, idx: np.intp):
        if idx is None:
            return None
        return cv2.cvtColor(cv2.imread(self.zed_left_files[idx], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    
    def read_realsense_img(self, idx: np.intp):
        if idx is None:
            return None
        return cv2.cvtColor(cv2.imread(self.realsense_files[idx], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        
    def load_poses(self, poses_file):
        with open(poses_file, 'r') as f:
            poses = [[float(x) for x in line.split(' ')] for line in f.readlines()]

        self.poses = []
        self.poses_ts = []
        for t, x, y, z, qx, qy, qz, qw in poses:
            self.poses_ts.append(t)
            P = build_matrix(x, y, z, (qw, qx, qy, qz))
            self.poses.append(P)
        
        self.poses_ts = np.asarray(self.poses_ts)
        # self.poses_ts = self.poses_ts - self.poses_ts[0]

        self.logger.warning(f"Poses ts: {self.poses_ts[:10]}")
        



    # def load_poses(self, poses_file):
    #     def _lidar_pose_gt(poses_gt):
    #         _tr = self.calibration["Tr"].reshape(3, 4)
    #         tr = np.eye(4, dtype=np.float64)
    #         tr[:3, :4] = _tr
    #         left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
    #         right = np.einsum("...ij,...jk->...ik", left, tr)
    #         return right

    #     poses = pd.read_csv(poses_file, sep=" ", header=None).values
    #     n = poses.shape[0]
    #     poses = np.concatenate(
    #         (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
    #     )
    #     poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
    #     return _lidar_pose_gt(poses)
    
    def load_ts(self, ts_file):
        with open(ts_file, 'r') as f:
            ts = np.asarray([float(filename[:-1]) for filename in f.readlines()])
        return ts
    
    
    
    