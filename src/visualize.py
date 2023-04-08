# Partially borrowed from https://github.com/SHI-Labs/OneFormer

import cv2
import json
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import plotly.graph_objects as go
import pycocotools.mask as mask_util
import matplotlib.figure as mplfigure

from src.config import get_config
config = get_config()
# from src.config import get_config_notebook
# config = get_config_notebook('/home/kowalski/3d_maps/2D-3D-segmentation/config/mipt_campus_day.yaml')

from src.utils import read_calib_file

from matplotlib.backends.backend_agg import FigureCanvasAgg

with open(config.segmentation_config, 'r') as f: # type: ignore
    seg_config = json.load(f)

STUFF_CLASSES = seg_config['stuff_classes']
STUFF_COLORS = seg_config['stuff_colors']

PLOTLY_COLORSCALE=[ f'rgb({r},{g},{b})' for r,g,b in STUFF_COLORS]

def imshow(img: np.ndarray, mask=False):
    plt.figure(figsize=[10, 10])
    if not mask:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)


def cloudshow(cloud, colors=None, save_path=None, colorscale=None, labels=None):
    colored = not colors is None

    if not colored:
        colors = cloud[:,2]
        colorscale = 'Viridis'
        label=None
    else:
        if colorscale is None:
            colorscale = PLOTLY_COLORSCALE

    if colored:
        label=[STUFF_CLASSES[idx] for idx in colors]
        if labels is not None:
            label = [l_1 + '\n' + str(l_2) for l_1, l_2 in zip(label, labels)] # type: ignore

    fig = go.Figure(data=[go.Scatter3d(
        x=cloud[:,0],
        y=cloud[:,1],
        z=cloud[:,2],
        mode='markers',
        text=label, # type: ignore
        marker=dict(
            size=2,
            color=colors,                # set color to an array/list of desired values
            colorscale=colorscale,   # choose a colorscale
            opacity=0.8
        )
    )],)

    # tight layout
    fig.update_layout(scene_aspectmode="data")

    if save_path is None:
        fig.show()
    else:
        fig.write_image(save_path)

class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.
        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")
    

def draw_binary_mask(binary_mask, color=None, *, res_img, alpha=0.5):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component smaller than this area will not be shown.
        Returns:
            output (VisImage): image object with mask drawn.
        """
        color = mplc.to_rgb(color) # type: ignore

        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        # mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = color
        rgba[:, :, 3] = (binary_mask == 1).astype("float32") * alpha
        res_img.ax.imshow(rgba, extent=(0, res_img.width, res_img.height, 0))
        
        return res_img

def get_colored_mask(img, mask, alpha=0.5, show=False):
    res_img = VisImage(img)

    labels, areas = np.unique(mask, return_counts=True)
    sorted_idxs = np.argsort(-areas).tolist()
    labels = labels[sorted_idxs]
    for label in filter(lambda l: l < len(STUFF_CLASSES), labels):
        try:
            mask_color = [x / 255 for x in STUFF_COLORS[label]]
        except (AttributeError, IndexError):
            mask_color = None

        binary_mask = (mask == label).astype(np.uint8)
        res_img = draw_binary_mask(
            binary_mask,
            color=mask_color,
            res_img=res_img,
            alpha=alpha,
        )
    
    if show:
        plt.figure(figsize=[10, 10])
        plt.imshow(res_img.get_image())

    return res_img


def draw_points_on_image(img: np.ndarray, points: np.ndarray, colors: np.ndarray):
    """
        Args:
            img (ndarray): standart opencv image.

            points (ndarray): array of 2D coordinates of projected points with shape (n, 2). Coordinates should match with cam_resolution.

            colors (ndarray): array of colors for each point in RGB uint8 format with shape (n, 3).
            
        Returns:
            img (ndarray): standart opencv image with points drawn.
    """
    proj_img = img.copy()
    
    for point, d in zip(points, colors): # points.T
        c = (int(d[0]), int(d[1]), int(d[2]))
        proj_img = cv2.circle(proj_img, point, radius=2, color=c, thickness=cv2.FILLED)
    
    return proj_img


def save_colored_cloud(scan: np.ndarray, labels: np.ndarray, save_path='output/colored_scan.pcd'):
    scan_colors = np.array([np.asarray(STUFF_COLORS[label], dtype=float)/255. for label in labels])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan)
    pcd.colors = o3d.utility.Vector3dVector(scan_colors)

    o3d.io.write_point_cloud(save_path, pcd)