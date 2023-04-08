from typing import Tuple, Optional, Union
import logging
import os
import shutil
import multiprocessing


import numpy as np
from tqdm import tqdm
from sklearn.neighbors import RadiusNeighborsClassifier

from src.config import get_config, Config
from src.projection import project_scan_to_camera
from src.dataset import MIPT_Campus_Dataset
from src.utils import get_points_labels_by_mask, read_calib_file, transform_xyz
from src.visualize import cloudshow

cfg = get_config()

logger = logging.getLogger(__name__)
logger.setLevel(cfg.logging_level) # type: ignore

def segment_pointcloud(points: np.ndarray,
                       seg_mask: np.ndarray,
                       cam_config: Config,
                       point_labels: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Args:
            points (ndarray): LiDAR pointcloud with shape (n, 3)
            seg_mask (ndarray): segmentation mask with shape (W, H, 1) and uint8 type.
            cam_config (Config): Config object for one of cameras (should contatin path to calib file and resolution)
            point_labels (Optional[np.ndarray]): Array of labels for points (shape (n,), dtype=np.uint8). If passed - wil be updated, if not - created new one for all points in cloud and default label 0 (Unknown).
        Returns:
            point_labels (np.ndarray): Array of labels for points (shape (n,), dtype=np.uint16).
            uncertanities (np.ndarray): Array of uncertanities for labels (shape (n,), dtype=np.float32). Calculated as 1/depth, where depth is the distance from the optical center of the camera to the point.
            in_image_mask (np.ndarray): Binary mask for cloud points, with True for points that that were caught in the frame.
    """

    if point_labels is None:
        point_labels = np.zeros(len(points), dtype=np.uint8)
    
    P_z = read_calib_file(cam_config.config_path) # type: ignore
    projected_z, depths, in_image = project_scan_to_camera(points, P_z['P'], # type: ignore
                                                        P_z['resolution'],
                                                        tf_config=cam_config.left, # type: ignore
                                                        return_mask=True)
    point_labels[in_image] = get_points_labels_by_mask(projected_z, seg_mask)

    uncertanities = 1./np.asarray(depths)

    return point_labels, uncertanities, in_image

def segment_pointcloud_batch(batch: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    for pose, scan, zed_mask, realsense_mask in tqdm(batch, "batch processing"):
        if scan is None:
            logging.warning("Skip pose from batch!")
            continue
        scan_labels = np.zeros(len(scan), dtype=np.uint16)

        # Front cam
        if zed_mask is not None:
            scan_labels, unct, zed_in_frame = segment_pointcloud(scan, zed_mask, cfg.front_cam) # type: ignore

        # Back cam
        if realsense_mask is not None:
            scan_labels, unct, rs_in_frame = segment_pointcloud(scan, realsense_mask, cfg.back_cam, scan_labels) # type: ignore


        scan_t = transform_xyz(pose, scan)
        scan_batch = np.append(scan_batch, scan_t, axis=0)
        batch_labels = np.append(batch_labels, scan_labels, axis=0)
        batch_uncert = np.append(batch_uncert, unct, axis=0) # type: ignore

        # Normalize
        batch_uncert = (batch_uncert - min(batch_uncert)) / (max(batch_uncert) - min(batch_uncert))

    return scan_batch, batch_labels, batch_uncert

def segment_pointcloud_w_semantic_uncert(dataset: MIPT_Campus_Dataset,
                                target_scan_id: Union[int, np.intp],
                                estimation_range: float) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Segments pointcloud and estimates labels uncertanies comparing them with other scans in some range.

        Args:
            dataset (MIPT_Campus_Dataset): dataset instance for accessing other scans

            target_scan_id (int OR np.intp): keypose id for segmentation.

            estimation_range (float): the range within which the key poses (robot poses) will be found, to assess the confidence in the markup of the current scan. 

            Note: it's not range for points, it range for robot poses. So, points range will be: estimation_range + lidar_range.
        Returns:
            target_scan_labels (np.ndarray): Array of labels for points (shape (n,), dtype=np.uint16.
            labels uncertanities (np.ndarray): Array of uncertanities for labels (shape (n,), dtype=np.float32). Calculated as 1/depth, where depth is the distance from the optical center of the camera to the point.
    """
    
    target_pose, target_scan, _, _ = dataset[target_scan_id]
    
    batch_idx = dataset.n_nearest_by_pose(target_scan_id, max_dist=estimation_range)
    logger.debug(f"N nearest: {batch_idx}")
    # input()
    batch = [dataset[_i] for _i in batch_idx]

    batch.insert(0, dataset[target_scan_id])

    scan_weights = np.zeros(len(target_scan), dtype=np.float32)
    target_scan_t = transform_xyz(target_pose, target_scan)
        
    batch_scans, batch_labels, batch_uncert = segment_pointcloud_batch(batch=batch)
    
    #Check available cpu's
    cpus = multiprocessing.cpu_count() - 1
    logger.info(f"Use {cpus} cpus for KD-tree")    

    # Adopt to labels
    batch_uncert[batch_labels == 0] = 0 # Unknown
    batch_uncert[batch_labels == 28] = 0 # Sky
    
    
    #Build KD-tree on top of the source data
    nn = RadiusNeighborsClassifier(algorithm='kd_tree', radius=0.3, weights='distance', outlier_label=0, n_jobs=cpus) # type: ignore
    nn.fit(batch_scans, [int(l) for l in batch_labels])
    r_neigh_dist, r_neigh_ids = nn.radius_neighbors(target_scan_t, radius=0.3, return_distance=True, sort_results=True)

    for m_id, (n_dists, n_ids) in tqdm(enumerate(zip(r_neigh_dist, r_neigh_ids)), 'init semantic weights'):
            
        # remove target point itself to avoid zero divisions
        n_dists = n_dists[1:]
        n_ids = n_ids[1:]

        if len(n_ids) == 0:
            continue    
        
        # semantic consistency
        neighbours_dist_weights = 1./np.asarray(n_dists)
        sensor_dist_weights = batch_uncert[n_ids]
            
        label_weights = neighbours_dist_weights * sensor_dist_weights

        m_label = batch_labels[m_id]
        if m_label == 0:
            scan_weights[m_id] = 0
            continue
        n_labels = batch_labels[n_ids]

        if sum(n_labels) == 0:
            ratio_my_labels = batch_uncert[m_id]
        else:
            ratio_my_labels = np.sum(np.asarray(n_labels == m_label, dtype=np.int32) * label_weights) /float(len(n_labels))


        # moving objects
        temporal_consist = float(np.sum(n_ids >= len(target_scan)))/len(n_ids)
        logger.debug(f"temporal_consist: {temporal_consist}")


        scan_weights[m_id] = temporal_consist * ratio_my_labels
        logger.debug(f"scan_weights[m_id]: {scan_weights[m_id]}")


    # Normalize
    max_w = max(scan_weights)
    min_w = min(scan_weights)
    if max_w - min_w != 0:
        scan_weights -= min_w
        scan_weights /= (max_w - min_w)

    logger.debug(f"Weights > 0: {len(scan_weights)}")

    # cloudshow(m_scan, np.asarray(scan_weights > 0.2, dtype=np.uint16), colorscale=simple_colors)
    # cloudshow(target_scan, batch_labels[:len(target_scan)], labels=["Uncert: "+str(x) for x in scan_weights])
    # input()
    return batch_labels[:len(target_scan)], scan_weights
    
def dataset_segmentation(dataset: MIPT_Campus_Dataset, 
                         overwrite=cfg.semantics.overwrite_existing_data) -> None: # type: ignore
    
    pbar = tqdm(total=len(dataset), desc="Segmenting scans..") #tqdm(total=len(dataset))
    i = 0

    out_dir = os.path.join(cfg.dataset_root_dir, # type: ignore
                           'unpacked_'+ cfg.sequence, # type: ignore
                           'velodyne_points',
                           cfg.semantics.initial_out_dir) # type: ignore
    
    # env_setup
    if os.path.exists(out_dir):
        if overwrite:
            logger.warning("Remove existing initial segmentation data!")
            shutil.rmtree(out_dir)
        else:
            logger.warning("Dataset segmentation data already exists! Skipping this step. If you want to ovewrite existing data, please, call func with corresponding flag.")
            return
        
    os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True, mode=0o775)
    os.makedirs(os.path.join(out_dir, "uncert"), exist_ok=True, mode=0o775)

    for m_pose, m_scan, _, _ in dataset: #zed_img, rs_img, , scan, zed_mask, realsense_mask
        # if i < 149:
        #     i+=1
        #     pbar.update(1)
        #     continue
        if i == 150:
            break
        if m_scan is None:
            pbar.update(1)
            i+=1
            logging.warning("Skip pose")
            continue

        # m_scan_t = transform_xyz(m_pose, m_scan)
        scan_labels, scan_uncert = segment_pointcloud_w_semantic_uncert(dataset, 
                                                                        i, 
                                                                        estimation_range=cfg.semantic_uncert_range) # type: ignore
        
        scan_labels.tofile(os.path.join(out_dir, "labels", f"labels_{i}.label")) # np.int16 shape (n,)
        scan_uncert.tofile(os.path.join(out_dir, "uncert", f"uncert_{i}.bin")) # np.float32 shape (n,)

        pbar.update(1)
        i+=1

        logger.info("Succefully segment dataset!")

        
