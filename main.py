import coloredlogs, logging
import numpy as np
from tqdm import tqdm
from vdbfusion import VDBVolume
import trimesh

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

logger = logging.getLogger(__name__)

from src.config import get_config, Config
from src.dataset import MIPT_Campus_Dataset
from src.visualize import save_colored_cloud, cloudshow
from src.utils import transform_xyz
from src.segmentation import segment_pointcloud_w_semantic_uncert, dataset_segmentation, refine_segmentation


def main(cfg: Config):
    # dataset._getitem_set['scan'] = False
    # dataset._getitem_set['segmentation_masks'] = False

    # logger.info(f"Len ds: {len(dataset)}")
    # logger.info(f"Len zed left: {len(dataset.zed_left_files)}")
    # logger.info(f"Len realsense: {len(dataset.realsense_files)}")
    # logger.info(f"Len lidar: {len(dataset.scan_files)}")
    # input()

    dataset_segmentation(MIPT_Campus_Dataset(cfg))

    # logger.info('Datset segmented!\nPress enter to continue..')
    # input()

    dataset = MIPT_Campus_Dataset(cfg)
    dataset._getitem_set['init_scan_labels'] = True
    dataset._getitem_set['segmentation_masks'] = False

    #! MAP
    map = np.empty((0,3), dtype='float32')
    map_labels = np.empty((0), dtype='int16')
    #!\\\\\

    batch_point_nums = []


    vdb_volume = VDBVolume(voxel_size=0.15, sdf_trunc=0.3, space_carving=False)

    pbar = tqdm(total=len(dataset), desc="Segmenting scans..") #tqdm(total=len(dataset))
    i = 0
    for pose, scan, scan_labels, scan_uncert in dataset: #zed_img, rs_img, , scan, zed_mask, realsense_mask #
        # if i < 100:
        #     i+=1
        #     pbar.update(1)
        #     continue
        if i == 100: #! MAX 150
            break
        if scan is None:
            pbar.update(1)
            i+=1
            logging.warning("Skip pose")
            continue

        #? target_labels, target_weights = refine_segmentation(dataset, i, cfg.semantic_refine_range) #type: ignore
        target_labels = np.zeros(len(scan), dtype='int16')
        
        logger.debug(f"Scan idx: {i}")
        logger.debug(f"Scan_labels: {scan_labels.shape}")
        logger.debug(f"Scan_labels: {scan_uncert.shape}")

        scan_t = transform_xyz(pose, scan)
        vdb_volume.integrate(scan_t.astype(np.float64), pose)
        # scan_labels, scan_uncert = segment_pointcloud_w_semantic_uncert(dataset, i, estimation_range=5.)

        map = np.append(map, scan_t, axis=0)
        # logger.debug(f"Labels shape: {map_labels.shape} labels shape: {batch_labels.shape}")
        map_labels = np.append(map_labels, target_labels, axis=0)
        #? map_labels = np.append(map_labels, np.asarray(scan_uncert > 0.7, dtype=np.uint16), axis=0)
        # cloudshow(scan, np.asarray(scan_uncert > 0.05, dtype=np.int16) + np.asarray(scan_uncert > 0.1, dtype=np.int16))
        # cloudshow(scan, scan_labels, labels=["Uncert: "+str(x) for x in scan_uncert])
        # cloudshow(scan, scan_labels)
        # cloudshow(scan, target_labels, labels=["Uncert: "+str(x) for x in target_weights])
        # input()

        pbar.update(1)
        i+=1
    
    # cloudshow(map, map_labels)
    save_colored_cloud(map, map_labels, save_path='output/test_night.pcd')

    vertices, triangles = vdb_volume.extract_triangle_mesh(fill_holes=True, min_weight=3.0)
    logger.info(f"V: {len(vertices)} T: {len(triangles)}")
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    trimesh.exchange.export.export_mesh(mesh, 'output/test_mesh.ply', 'ply') #type: ignore

    logger.info('Done!')


if __name__ == "__main__":
    cfg = get_config()
    logger.setLevel(cfg.logging_level) #type: ignore
    coloredlogs.install(level=cfg.logging_level) #type: ignore

    main(cfg)