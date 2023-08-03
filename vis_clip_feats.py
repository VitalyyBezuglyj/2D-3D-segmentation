import colorsys
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def normalize(arr, t_min, t_max) -> np.ndarray:
    arr = np.asarray(arr)
    # norm_arr = []
    diff = t_max - t_min
    arr_min = min(arr)
    arr_max = 1# max(arr)
    diff_arr = arr_max - min(arr)   
    return (((arr - arr_min)*diff)/diff_arr) + t_min

float2color = np.vectorize(lambda f: tuple([int(255*c) for c in colorsys.hsv_to_rgb(f, 1, 1)][::-1]))

labelset = ["chair",        # 0
            "plastic box",  # 1
            "soft toy",     # 2
            "cat toy",      # 3
            "table",        # 4
            "drawer",       # 5
            "TV set",       # 6
            "car",          # 7
            "kettle",       # 8
            "trash bin",    # 9
            "sofa",         # 10
            "vacuum cleaner"# 11
            ]

map_feat = np.load("/home/kowalski/data/strl/10_find_openscene_feat_fusion.npy")
map_pcd = o3d.io.read_point_cloud("/home/kowalski/3d_maps/2D-3D-segmentation/output/colored_map_pcl.ply")
print(np.asarray(map_pcd.points).shape)

colors = map_feat[:, 6]
print(f"Min: {min(colors)}, max:{max(colors)}")
colors_n = normalize(colors, 0, 0.75)
print(f"Min: {min(colors_n)}, max:{max(colors_n)}")
# colors_e = np.exp(colors_n)
# colors_e /= colors_e.sum() 
# print(f"Min: {min(colors_e)}, max:{max(colors_e)}")


map_pcd.colors = o3d.utility.Vector3dVector(np.asarray(float2color(colors_n)).T) #o3d.utility.Vector3dVector(plt.cm.jet(colors_n)[:, :3])

o3d.io.write_point_cloud("output/clip_cis.pcd", map_pcd)

# print(map_feat.shape)

