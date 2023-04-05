import coloredlogs, logging
import numpy as np
from tqdm import tqdm

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

logger = logging.getLogger(__name__)

from src.config import get_config, Config
from src.dataset import MIPT_Campus_Dataset
from src.visualize import VisImage, cloudshow, save_colored_cloud, imshow, get_colored_mask
from src.projection import project_scan_to_camera
from src.utils import get_points_labels_by_mask, read_calib_file, transform_xyz


def main(cfg: Config):
    dataset = MIPT_Campus_Dataset(cfg)
    # dataset._getitem_set['images'] = True

    logger.info(f"Len ds: {len(dataset)}")
    logger.info(f"Len zed left: {len(dataset.zed_left_files)}")
    logger.info(f"Len realsense: {len(dataset.realsense_files)}")
    logger.info(f"Len lidar: {len(dataset.scan_files)}")
    input()

    # cloudshow(scan)
    map = np.empty((0,3), dtype='float32')
    map_labels = np.empty((0), dtype='int8')

    pbar = tqdm(total=len(dataset), desc="Segmenting scans..") #tqdm(total=len(dataset))
    i = 0
    for pose, scan, zed_mask, realsense_mask in dataset: #zed_img, rs_img, 
        # if i < 469:
        #     i+=1
        #     continue
        if i ==500:
            break
        if scan is None:
            logging.warning("Skip pose")
            continue

        scan_labels = np.zeros(len(scan), dtype=np.uint8)
        # map_labels = np.append(map_labels, scan_labels, axis=0)

        # Front cam
        if zed_mask is not None:
            P_z = read_calib_file(cfg.front_cam.config_path) # type: ignore
            projected_z, depths, in_z_image = project_scan_to_camera(scan, P_z['P'],              # type: ignore
                                                                P_z['resolution'],
                                                                tf_config=cfg.front_cam.left, # type: ignore
                                                                return_mask=True)
            scan_labels[in_z_image] = get_points_labels_by_mask(projected_z, zed_mask)

        # Back cam
        if realsense_mask is not None:
            P_r = read_calib_file(cfg.back_cam.config_path) # type: ignore
            projected_r, depths, in_r_image = project_scan_to_camera(scan, P_r['P'],              # type: ignore
                                                                P_r['resolution'],
                                                                tf_config=cfg.back_cam.left, # type: ignore
                                                                return_mask=True)
            scan_labels[in_r_image] = get_points_labels_by_mask(projected_r, realsense_mask)

        scan_t = transform_xyz(pose, scan)
        
        # ! #####
        # colored = get_colored_mask(zed_img, zed_mask)
        # colored.save('output/colored_469_zed.png')

        # colored_r = get_colored_mask(rs_img, realsense_mask)
        # colored_r.save('output/colored_470_rs.png')
        # ! ##
        # cloudshow(scan, scan_labels)
        # input()
        # exit(0)
        # ! #####
        logger.debug(f"Map shape: {map.shape} Scan shape: {scan.shape}")
        # map = np.append(map, scan, axis=0)
        map = np.append(map, scan_t, axis=0)
        logger.debug(f"Labels shape: {map_labels.shape} labels shape: {scan_labels.shape}")
        map_labels = np.append(map_labels, scan_labels, axis=0)

        pbar.update(1)
        i+=1
    
    # cloudshow(map, map_labels)
    save_colored_cloud(map, map_labels)

    logger.info('Done!')


if __name__ == "__main__":
    cfg = get_config()
    logger.setLevel(cfg.logging_level) #type: ignore
    coloredlogs.install(level=cfg.logging_level) #type: ignore

    main(cfg)