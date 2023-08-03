import coloredlogs, logging
import numpy as np
import imageio
from tqdm import tqdm

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)
import os
import torch
import cv2


logger = logging.getLogger(__name__)

from src.config import get_config, Config
from src.dataset_strl import MIPT_STRL_Dataset
from src.visualize import save_colored_cloud, save_colored_map
from src.utils import transform_xyz, stack_size, adjust_intrinsic, float2color, normalize


def main(cfg: Config):
    logger.error(f"Rec limit: {stack_size()}")
    dataset = MIPT_STRL_Dataset(cfg)
    dataset._getitem_set['images'] = True
    dataset._getitem_set['depth'] = True
    dataset._getitem_set['cam_poses'] = True

    #! MAP
    map = np.empty((0,3), dtype='float32')
    map_colors = np.empty((0,3), dtype='float32')
    cam_poses = []
    li_poses = []
    #!\\\\\

    save_data = True
    if not os.path.isdir(cfg.preprocess_root_dir):
        os.makedirs(cfg.preprocess_root_dir)
    else:
        save_data = False and cfg.overwrite_existing_data
    output_depth_path = f"{cfg.preprocess_root_dir}/strl_2d/{cfg.sequence}/depth"
    if not os.path.isdir(output_depth_path):
        os.makedirs(output_depth_path)
    output_pose_path = f"{cfg.preprocess_root_dir}/strl_2d/{cfg.sequence}/pose"
    if not os.path.isdir(output_pose_path):
        os.makedirs(output_pose_path)
    output_color_path = f"{cfg.preprocess_root_dir}/strl_2d/{cfg.sequence}/color"
    if not os.path.isdir(output_color_path):
        os.makedirs(output_color_path)
    output_3d_path = f"{cfg.preprocess_root_dir}/strl_3d"
    if not os.path.isdir(output_3d_path):
        os.makedirs(output_3d_path)

    img_dim = (640, 360)
    
    original_img_dim = dataset.calib_zed['resolution']
    # intrinsic parameters on the original image size
    intrinsics = dataset.calib_zed['P']

    # save the intrinsic parameters of resized images
    intrinsics = adjust_intrinsic(intrinsics, original_img_dim, img_dim)
    np.savetxt(os.path.join(cfg.preprocess_root_dir, f"strl_2d", 'intrinsics.txt'), intrinsics)

    pbar = tqdm(total=len(dataset), desc="Preprocessing data..", colour = 'GREEN') #tqdm(total=len(dataset))
    i = 0
    for li_pose, scan, cam_pose, image, depth in dataset:
        # if i < 3000:
        #     i+=1
        #     pbar.update(1)
        #     continue
        # if i == 20:
        #     break
        if scan is None:
            pbar.update(1)
            i+=1
            logging.warning("Skip pose")
            continue
        
        logger.debug(f"Scan idx: {i}")
        logger.debug(f"Scan: {scan.shape}")
        logger.debug(f"Image: {image.shape}")
        logger.debug(f"Depth: {depth[:,:,0].shape}")

        # Save data
        if i % cfg.preproc_sample_freq == 0 and save_data:
            image = cv2.resize(depth, img_dim, interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(os.path.join(output_color_path, str(i)+'.jpg'), image)
            depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(os.path.join(output_depth_path, str(i)+'.png'), depth[:,:,0])
            np.savetxt(os.path.join(output_pose_path, str(i)+'.txt'), cam_pose)
        
        scan_t = transform_xyz(li_pose, scan)


        map = np.append(map, scan_t, axis=0)
        colors = np.repeat(np.asarray([[127, 127, 127]]), scan_t.shape[0], axis=0)
        # colors = np.asarray(float2color(normalize(scan_t[:,2], t_min=0, t_max=5))).T
        map_colors = np.append(map_colors, colors, axis=0)

        cam_poses.append(cam_pose)
        li_poses.append(li_pose)

        pbar.update(1)
        i+=1
    
    cam_poses = np.asarray(cam_poses)
    li_poses = np.asarray(li_poses)
    save_colored_cloud(map, colors=map_colors, save_path='output/test_strl_3.pcd')
    save_colored_map(map, colors=map_colors, camera_poses=cam_poses, lidar_poses=li_poses)

    coords = np.ascontiguousarray(map)
    colors = np.ascontiguousarray(map_colors) / 127.5 - 1

    # no GT labels are provided, set all to 255
    labels = 255*np.ones((coords.shape[0], ), dtype=np.int32)
    torch.save((coords, colors, labels),
            os.path.join(output_3d_path, f"{cfg.sequence}.pth"))

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