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
from src.utils import get_points_labels_by_mask, read_calib_file, transform_xyz
from src.segmentation import segment_pointcloud, segment_pointcloud_w_semantic_uncert


def main(cfg: Config):
    dataset = MIPT_Campus_Dataset(cfg)
    # dataset._getitem_set['scan'] = False
    # dataset._getitem_set['segmentation_masks'] = False

    logger.info(f"Len ds: {len(dataset)}")
    logger.info(f"Len zed left: {len(dataset.zed_left_files)}")
    logger.info(f"Len realsense: {len(dataset.realsense_files)}")
    logger.info(f"Len lidar: {len(dataset.scan_files)}")
    input()

    #! MAP
    map = np.empty((0,3), dtype='float32')
    map_labels = np.empty((0), dtype='int8')
    #!\\\\\

    batch_point_nums = []

    pbar = tqdm(total=len(dataset), desc="Segmenting scans..") #tqdm(total=len(dataset))
    i = 0
    for m_pose, m_scan, _, _ in dataset: #zed_img, rs_img, , scan, zed_mask, realsense_mask
        if i < 149:
            i+=1
            pbar.update(1)
            continue
        if i == 150:
            break
        if m_scan is None:
            pbar.update(1)
            i+=1
            logging.warning("Skip pose")
            continue

        m_scan_t = transform_xyz(m_pose, m_scan)
        scan_labels, scan_uncert = segment_pointcloud_w_semantic_uncert(dataset, i, estimation_range=5.)

        map = np.append(map, m_scan_t, axis=0)
        # logger.debug(f"Labels shape: {map_labels.shape} labels shape: {batch_labels.shape}")
        map_labels = np.append(map_labels, scan_labels, axis=0)
        #? map_labels = np.append(map_labels, np.asarray(scan_uncert > 0.7, dtype=np.uint16), axis=0)

        pbar.update(1)
        i+=1
    
    # cloudshow(map, map_labels)
    save_colored_cloud(map, map_labels, save_path='output/filter_moving_07.pcd')

    logger.info('Done!')


if __name__ == "__main__":
    cfg = get_config()
    logger.setLevel(cfg.logging_level) #type: ignore
    coloredlogs.install(level=cfg.logging_level) #type: ignore

    main(cfg)