import coloredlogs, logging
import numpy as np
from tqdm import tqdm

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

logger = logging.getLogger(__name__)

from src.config import get_config, Config
from src.dataset_strl import MIPT_STRL_Dataset
from src.visualize import save_colored_cloud
from src.utils import transform_xyz, stack_size, read_calib_file


def main(cfg: Config):
    logger.error(f"Rec limit: {stack_size()}")
    dataset = MIPT_STRL_Dataset(cfg)
    dataset._getitem_set['images'] = True
    dataset._getitem_set['depth'] = True
    dataset._getitem_set['image_poses'] = True

    #! MAP
    map = np.empty((0,3), dtype='float32')
    map_labels = np.empty((0), dtype='int16')
    #!\\\\\

    pbar = tqdm(total=len(dataset), desc="Preprocessing data..", colour = 'GREEN') #tqdm(total=len(dataset))
    i = 0
    for pose, scan, image, depth in dataset:
        # if i < 3000:
        #     i+=1
        #     pbar.update(1)
        #     continue
        # if i == 20: #! MAX 150
        #     break
        if scan is None:
            pbar.update(1)
            i+=1
            logging.warning("Skip pose")
            continue

        target_labels = np.zeros(len(scan), dtype='int16')
        
        logger.debug(f"Scan idx: {i}")
        logger.debug(f"Scan: {scan.shape}")

        scan_t = transform_xyz(pose, scan)


        map = np.append(map, scan_t, axis=0)
        map_labels = np.append(map_labels, target_labels, axis=0)

        pbar.update(1)
        i+=1
    
    # cloudshow(map, map_labels)

    # map, map_labels = filter_group_of_objects(map, map_labels, group='moving_object')
    # map, map_labels = filter_group_of_objects(map, map_labels, group='unknown')
    save_colored_cloud(map, map_labels, save_path='output/test_strl_3.pcd')

    # vertices, triangles = vdb_volume.extract_triangle_mesh(fill_holes=True, min_weight=3.0)
    # logger.info(f"V: {len(vertices)} T: {len(triangles)}")
    # mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # trimesh.exchange.export.export_mesh(mesh, 'output/test_mesh.ply', 'ply') #type: ignore
    # mesh_builder.export_mesh('output/semantic_mesh_kimera_style_3000-.ply')

    logger.info('Done!')


if __name__ == "__main__":
    cfg = get_config()
    logger.setLevel(cfg.logging_level) #type: ignore
    coloredlogs.install(level=cfg.logging_level) #type: ignore

    main(cfg)