import logging
import numpy as np
from tqdm import tqdm
from vdbfusion import VDBVolume
import trimesh
import json

from typing import Dict

logger = logging.getLogger(__name__)

from src.config import Config
from src.dataset import MIPT_Campus_Dataset
from src.visualize import save_colored_cloud, cloudshow
from src.utils import transform_xyz
from src.segmentation import segment_pointcloud_w_semantic_uncert, dataset_segmentation, refine_segmentation

class SemanticMeshBuilder():
    def __init__(self, cfg: Config, save_poses=False) -> None:
        self.mesh_layers = {}

        # with open(self.cfg.segmentation_config, 'r') as f: # type: ignore
        #     seg_config = json.load(f)
        # self.stuff_classes = seg_config['stuff_classes']
        # self.stuff_colors = seg_config['stuff_colors']

        self.sem_cfg = cfg.semantics_mapping.__dict__ # type: ignore
        self.sem_classes = list(self.sem_cfg.keys()) # type: ignore

        self.sem_classes.remove('unknown')
        for s_class in self.sem_classes:
            self.mesh_layers[s_class] = VDBVolume(
                                            voxel_size=self.sem_cfg[s_class].voxel_size,
                                            sdf_trunc=0.3,
                                            space_carving=False)

        pass

    def _split_by_classes(self, labels: np.ndarray) -> Dict:
        """
        Generates a list of binary masks, one for each semantic class based on the list of semantic labels. The partitioning by class is specified in the config.

        Args:
            labels (ndarray): list of semantic labels (of LiDAR scan) with shape (N, ) and type np.int16.

        Returns:
            binary masks ( dict('class_1': mask(nd.array), ..) ): dictionary with semantic classes as keys, and binary masks as values.
        """

        logger.debug(f"Splitting scan by labels:")
        masks = {}
        for s_class in self.sem_classes:
            new_mask = np.zeros(len(labels), dtype=bool)
            for label in self.sem_cfg[s_class].labels:
                new_mask = np.logical_or(new_mask, labels==label)
            masks[s_class] = new_mask
            logger.debug(f"\tFound {sum(new_mask.astype(np.int16))} labels of {s_class} class")
        
        return masks

    def integrate_scan(self, scan: np.ndarray, pose: np.ndarray, labels: np.ndarray) -> None:
        """
        Integrates semantically labelled LiDAR scan into mesh map in semantic-aware manner.

        Args:
            points (ndarray): LiDAR pointcloud with shape (n, 3) and type np.float32 | np.float64

            pose (ndarray): SO3 tranformation matrix representing the robot's (lidar's) pose 

            labels (ndarray): list of semantic labels (of LiDAR scan) with shape (N, ) and type np.int16.

        Returns:
            None
        """
        
        sem_masks = self._split_by_classes(labels)

        logger.debug(f"Semantic aware scan integration:")
        for s_class in self.sem_classes:
            if sum(sem_masks[s_class]) == 0:
                logger.debug(f"\tSkipping {s_class} mesh layer - mask is empty.")
                continue
            self.mesh_layers[s_class].integrate(scan[sem_masks[s_class]].astype(np.float64), pose)
            logger.debug(f"\t{sum(sem_masks[s_class])} points integrated to {s_class} mesh layer")
        

    def _get_face_colors(self, faces_num: int, class_name: str) -> np.ndarray:
        return np.tile(np.asarray([*self.sem_cfg[class_name].color, 255], dtype=np.uint8), faces_num).reshape(-1, 4)

    def export_mesh(self, path: str) -> None:
        final_scene = trimesh.Scene()
        for s_class in self.sem_classes:
            vertices, triangles = self.mesh_layers[s_class].extract_triangle_mesh(fill_holes=True,
                                                                                  min_weight=self.sem_cfg[s_class].ext_weight)
            
            logger.debug(f"Extracted {s_class} mesh layer with V: {len(vertices)} T: {len(triangles)}")
            mesh_colors = self._get_face_colors(len(triangles), s_class)
            logger.debug(f"Color: {self.sem_cfg[s_class].color}")
            logger.debug(f"Color w a: {[*self.sem_cfg[s_class].color, 255]}")
            logger.debug(f"Colors: {mesh_colors.shape}")
            final_scene.add_geometry(trimesh.Trimesh(vertices=vertices, faces=triangles, face_colors=mesh_colors))
        
        final_scene.export(path, file_type="ply")



        