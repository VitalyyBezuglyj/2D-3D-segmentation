
## Adopted from https://github.com/PRBonn/vdbfusion 

import os
import yaml
import glob
import logging

import cv2
import numpy as np
from typing import Union, Tuple

from src.utils import read_calib_file, build_matrix
from src.logger import TqdmLoggingHandler

class MIPT_Campus_Dataset:
    def __init__(self, config):
        """Simple DataLoader to provide a ready-to-run example.

        Heavily inspired in PyLidar SLAM
        """
        # Config stuff
        self.cfg = config

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.cfg.logging_level)
        self.logger.addHandler(TqdmLoggingHandler())

        self.zed_left_dir = os.path.join(self.cfg.dataset_root_dir,
                                         'data',
                                         self.cfg.sequence, 
                                         'front_left')
        self.logger.debug(f"zed_left_dir: {self.zed_left_dir}")
        
        self.zed_right_dir = os.path.join(self.cfg.dataset_root_dir, 
                                          'data',
                                          self.cfg.sequence, 
                                          'front_right')
        self.logger.debug(f"zed_right_dir: {self.zed_right_dir}")
        
        self.realsense_dir = os.path.join(self.cfg.dataset_root_dir, 
                                          'data',
                                          self.cfg.sequence, 
                                          'back_left')
        self.logger.debug(f"realsense_dir: {self.realsense_dir}")
        
        self.velodyne_dir = os.path.join(self.cfg.dataset_root_dir, 
                                         'data',
                                         self.cfg.sequence, 
                                         'velodyne')
        self.logger.debug(f"velodyne_dir: {self.velodyne_dir}")
        
        self._getitem_set = {'images': False,
                             'scan': True,
                             'init_scan_labels': False, #TODO
                             'scan_labels': False, #TODO
                             'pose': True,
                             'segmentation_masks': True}

        # Read stuff
        #? self.zed_calibration = read_calib_file(os.path.join(self.cfg.dataset_root_dir, 
        #?                                                          "zed_calib.yml"))
        #? self.realsense_calibration = read_calib_file(os.path.join(self.cfg.dataset_root_dir, 
        #?                                                                "realsense_calib.yml"))

        self.load_poses(os.path.join(self.cfg.dataset_root_dir,
                                     'data',
                                     self.cfg.sequence, 
                                     'gt_poses.tum'))

        self.zed_left_ts = self.load_ts(os.path.join(self.zed_left_dir, "timestamps.txt"))
        self.logger.debug(f"Zed ts {self.zed_left_ts.shape}")

        self.zed_right_ts = self.load_ts(os.path.join(self.zed_right_dir, "timestamps.txt"))
        self.realsense_ts = self.load_ts(os.path.join(self.realsense_dir, "timestamps.txt"))
        self.scan_ts = self.load_ts(os.path.join(self.velodyne_dir, "timestamps.txt"))

        self.zed_left_files = sorted(glob.glob(self.zed_left_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.zed_left_files)} Zed left images")

        self.zed_right_files = sorted(glob.glob(self.zed_right_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.zed_right_files)} Zed right images")

        self.realsense_files = sorted(glob.glob(self.realsense_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.realsense_files)} Realsense images")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "/clouds/*.bin"))
        self.logger.debug(f"Found {len(self.scan_files)} Scan files")

        #TODO Add right segmentation masks (wider viewing angle)
        self.zed_seg_files = sorted(glob.glob(self.zed_left_dir + "/anno/*.png"))
        self.logger.debug(f"Found {len(self.zed_seg_files)} Zed left masks")

        self.realsense_seg_files = sorted(glob.glob(self.realsense_dir + "/anno/*.png"))
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
            # realsense_idx = self.align_by_ts(idx, self.realsense_ts)
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
            return_list.append(return_list.append(self.read_zed_mask(zed_left_idx))) # type: ignore
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
            self.logger.debug(f"zed_idx: {zed_left_idx}") # type: ignore
            self.logger.debug(f"rs_idx: {realsense_idx}") # type: ignore

        self.logger.debug(f"Pose_idx: {idx}")
        self.logger.debug(f"scan_idx: {scan_idx}")
        
        # input()
        return_list = []

        try:
            if self._getitem_set['pose']:
                return_list.append(self.poses[idx])
        except Exception as e:
            self.logger.error(f"Get pose exception: {e}")
        
        try:
            if self._getitem_set['scan']:
                scan = self.scans(scan_idx)
                return_list.append(scan)
        except Exception as e:
            self.logger.error(f"Get scan exception: {e}")

        try:
            if self._getitem_set['init_scan_labels']:
                return_list.append(self.init_labels(idx))
                return_list.append(self.init_uncert(idx))
        except Exception as e:
            self.logger.error(f"Get scan labels exception: {e}")

        try:
            if self._getitem_set['images']:
                return_list.append(self.read_zed_img(zed_left_idx)) # type: ignore
                #? return_list.append(self.read_zed_img(zed_right_idx)) # type: ignore
                return_list.append(self.read_realsense_img(realsense_idx)) # type: ignore
        except Exception as e:
            self.logger.error(f"Get images exception: {e}")
        
        try:
            if self._getitem_set['segmentation_masks']:
                return_list.append(self.read_zed_mask(zed_left_idx)) # type: ignore
                #? return_list.append(return_list.append(self.read_zed_mask(zed_left_idx))) # type: ignore
                return_list.append(self.read_realsense_mask(realsense_idx)) # type: ignore
        except Exception as e:
            self.logger.error(f"Get semantic masks exception: {e}")

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
    
    def _init_labels_path(self, idx):
        return os.path.join(self.velodyne_dir, f"init_semantics_conf/labels/labels_{idx}.label")
    
    def init_labels(self, idx):
        if idx is None:
            return None
        labels_path = self._init_labels_path(idx)
        self.logger.debug(f"Labels path: {labels_path}")
        if os.path.exists(labels_path):
            return np.fromfile(labels_path, dtype=np.int16)
        else:
            self.logger.debug(f"File {labels_path} not found!")
            return None
        
    def _init_uncert_path(self, idx):
        return os.path.join(self.velodyne_dir, f"init_semantics_conf/uncert/uncert_{idx}.bin")
    
    def init_uncert(self, idx):
        if idx is None:
            return None
        uncert_path = self._init_uncert_path(idx)
        if os.path.exists(uncert_path):
            return np.fromfile(uncert_path, dtype=np.float32)
        else:
            self.logger.debug(f"File {uncert_path} not found!")
            return None
    
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
        self.poses_t = [] # only translation
        self.poses_ts = []
        for t, x, y, z, qx, qy, qz, qw in poses:
            self.poses_ts.append(t)
            P = build_matrix(x, y, z, (qw, qx, qy, qz))
            self.poses.append(P)
            self.poses_t.append(np.asarray((x, y, z)))
        
        self.poses_ts = np.asarray(self.poses_ts)
        self.poses_t = np.asarray(self.poses_t)

        self.logger.warning(f"Poses ts: {self.poses_ts[:10]}")

    def n_nearest_by_pose(self, idx: Union[int, np.intp], max_dist: float, without_self = True) -> np.ndarray:
        poses_t = self.poses_t - self.poses_t[idx]
        if without_self:
            poses_t[idx] = np.asarray([max_dist * 3,max_dist * 3, max_dist * 3])
        
        in_range_idx = np.all(np.logical_and([-max_dist, -max_dist, -max_dist] <= poses_t, 
                                             poses_t <= [max_dist, max_dist, max_dist]), axis=1)

        return np.asarray(range(len(self.poses_t)))[in_range_idx]
    
    
    def load_ts(self, ts_file):
        with open(ts_file, 'r') as f:
            ts = np.asarray([float(filename[:-1]) for filename in f.readlines()])
        return ts
    
    
    
    