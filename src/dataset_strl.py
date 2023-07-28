
## Adopted from https://github.com/PRBonn/vdbfusion 

import os
import yaml
import glob
import logging

import cv2
import numpy as np
from typing import Union, Tuple

from src.utils import read_calib_file, build_matrix, transform_xyz
from src.logger import TqdmLoggingHandler

class MIPT_STRL_Dataset:
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
                                         self.cfg.sequence, 
                                         'zed_node_left_image_rect_color_compressed')
        self.logger.debug(f"zed_left_dir: {self.zed_left_dir}")
        

        self.zed_depth_dir = os.path.join(self.cfg.dataset_root_dir, 
                                          self.cfg.sequence, 
                                          'zed_node_depth_depth_registered')
        
        self.logger.debug(f"zed_depth: {self.zed_depth_dir}")
        
        self.velodyne_dir = os.path.join(self.cfg.dataset_root_dir, 
                                         self.cfg.sequence, 
                                         'velodyne_points')
        self.logger.debug(f"velodyne_dir: {self.velodyne_dir}")
        
        self._getitem_set = {'images': False,
                             'scan': True,
                             'lidar_pose': True,
                             'cam_poses': False,
                             'depth': False}

        # Read stuff
        self.calib_zed = read_calib_file(self.cfg.front_cam.config_path)

        self.load_poses_strl(os.path.join(self.cfg.dataset_root_dir,
                                     self.cfg.sequence, 
                                     'cartographer_tracked_global_odometry/depth_timestamps/poses.txt'))

        self.zed_left_ts = self.load_ts(os.path.join(self.zed_left_dir, "timestamps.txt"))
        self.logger.debug(f"Zed ts {self.zed_left_ts.shape}")

        self.zed_depth_ts = self.load_ts(os.path.join(self.zed_depth_dir, "timestamps.txt"))
        self.scan_ts = self.load_ts(os.path.join(self.velodyne_dir, "timestamps.txt"))

        self.zed_left_files = sorted(glob.glob(self.zed_left_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.zed_left_files)} Zed left images")

        self.zed_depth_files = sorted(glob.glob(self.zed_depth_dir + "/images/*.png"))
        self.logger.debug(f"Found {len(self.zed_depth_files)} Zed depth images")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "/clouds/*.bin"))
        self.logger.debug(f"Found {len(self.scan_files)} Scan files")

        assert(len(self.zed_left_ts) == len(self.zed_left_files))
        # self.logger.error(f"Len files: {len(self.zed_right_files)} -- Len ts: {len(self.zed_right_ts)}")
        # assert(len(self.zed_right_ts) == len(self.zed_right_files))
        assert(len(self.scan_ts) == len(self.scan_files))

        self.big_ts_diff = False

        self.t_base2lidar = build_matrix(*self.cfg.lidar.base_link2lidar_t, q=self.cfg.lidar.base_link2lidar_q)
        self.t_lidar2base = np.linalg.inv(self.t_base2lidar)

        self.t_lidar2cam = build_matrix(*self.cfg.front_cam.left.lidar2cam_t, q=self.cfg.front_cam.left.lidar2cam_q)
        self.t_cam2lidar = np.linalg.inv(self.t_lidar2cam)

        self.set_lidar_poses()
        self.set_cam_poses()



    
    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self.poses) + idx
        elif idx > len(self.poses):
            idx -= len(self.poses)

        scan_idx = self.align_by_ts(idx, self.scan_ts)

        if self._getitem_set['images']:
            zed_left_idx = self.align_by_ts(idx, self.zed_left_ts)
            zed_depth_idx = self.align_by_ts(idx, self.zed_depth_ts)
            self.logger.debug(f"zed_idx: {zed_left_idx}") # type: ignore
            self.logger.debug(f"rs_idx: {zed_depth_idx}") # type: ignore

        self.logger.debug(f"Pose_idx: {idx}")
        self.logger.debug(f"scan_idx: {scan_idx}")
        
        # input()
        return_list = []

        try:
            if self._getitem_set['lidar_pose']:
                return_list.append(self.lidar_poses[idx])
        except Exception as e:
            self.logger.error(f"Get pose exception: {e}")
        
        try:
            if self._getitem_set['scan']:
                scan = self.scans(scan_idx)
                return_list.append(scan)
        except Exception as e:
            self.logger.error(f"Get scan exception: {e}")

        try:
            if self._getitem_set['cam_poses']:
                return_list.append(self.cam_poses[idx])
        except Exception as e:
            self.logger.error(f"Get cam pose exception: {e}")

        try:
            if self._getitem_set['images']:
                return_list.append(self.read_zed_img(zed_left_idx)) # type: ignore
                #? return_list.append(self.read_zed_img(zed_right_idx)) # type: ignore
        except Exception as e:
            self.logger.error(f"Get images exception: {e}")

        try:
            if self._getitem_set['depth']:
                return_list.append(self.read_zed_img(zed_left_idx)) # type: ignore
                #? return_list.append(self.read_zed_img(zed_right_idx)) # type: ignore
        except Exception as e:
            self.logger.error(f"Get images exception: {e}")

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

        # points = transform_xyz(self.t_base2lidar, points)
        #TODO Add poses
        # points = self.transform_points(points, self.poses[idx]) if apply_pose else points
        return points

    def read_zed_deph(self, idx: np.intp):
        if idx is None:
            return None
        return cv2.imread(self.zed_depth_files[idx], cv2.IMREAD_UNCHANGED)
    
    def read_zed_img(self, idx: np.intp):
        if idx is None:
            return None
        return cv2.cvtColor(cv2.imread(self.zed_left_files[idx], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)


    def load_poses_strl(self, poses_file):
        with open(poses_file, 'r') as f:
            poses = [[float(x) for x in line.split(' ')] for line in f.readlines()]

        self.poses = []
        self.poses_t = [] # only translation
        self.poses_ts = self.load_ts(os.path.join(self.cfg.dataset_root_dir,
                                     self.cfg.sequence, 
                                     'cartographer_tracked_global_odometry/depth_timestamps/timestamps.txt'))
        for vec in poses:
            P_ = np.reshape(np.asarray(vec), (3, 4))
            P = np.eye(4)
            P[:3,:] = P_
            t = P_[:, 3]
            self.poses.append(P)
            self.poses_t.append(np.asarray(t))
        
        self.poses_ts = np.asarray(self.poses_ts)
        self.poses_t = np.asarray(self.poses_t)

        self.logger.warning(f"Poses ts: {self.poses_ts[:10]}")
        
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

    def set_cam_poses(self):
        self.cam_poses = np.asarray([self.t_cam2lidar @ pose @ self.t_lidar2cam for pose in self.lidar_poses])

    def set_lidar_poses(self):
        self.lidar_poses = np.asarray([self.t_base2lidar @ pose for pose in self.poses])

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
    
    
    
    