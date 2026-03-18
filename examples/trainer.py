import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import assert_never
from itertools import tee
from enum import Enum

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
import viser
import plotly.graph_objects as go
import plotly.express as px
from datasets.colmap import Dataset, Parser, BlenderDataset
from datasets.traj import generate_interpolated_path
from utils import rgb_to_sh
from util_viewer import UtilViewer
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)

from a2tg.rendering import rasterization_2dgs, rasterization_textured_gaussians, rasterization_packed_textured_gaussians
from a2tg.strategy import DefaultStrategy, MCMCStrategy, TextureStrategy


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Dataset mode
    dataset: str = "colmap"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [30_000])
    # Steps to pause when training for debugging
    pause_steps: List[int] = field(default_factory=lambda: [])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    min_opacity: float = 0.005

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = True
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    background_mode: str = None

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = False
    # Weight for normal loss
    normal_lambda: float = 5e-2
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 7_000

    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    # Alpha loss
    alpha_loss: bool = False
    alpha_lambda: float = 1e-1

    # scale_loss
    scale_loss: bool = False 
    scale_lambda: float = 1e-1
    
    # Model for splatting.
    model_type: Literal["2dgs", "textured_gaussians", "a2tg"] = "2dgs"

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=MCMCStrategy
    )

    # Pretrained checkpoints
    pretrained_path: str = None

    # textured gaussians
    texture_resolution: int = 50
    textured_rgb: bool = False
    textured_alpha: bool = False

    min_tex_res: int = 1
    max_tex_res: int = 16
    min_aspect_ratio: float = 6.0
    max_scale_for_thin: float = 0.05
    upscale_grad2d: float = 0.0008
    upscale_start_iter: int = 0
    upscale_stop_iter: int = 1002
    upscale_every: int = 500
    upscale_abs_grad: bool = True

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            strategy.min_opacity = float(self.min_opacity)
        else:
            assert_never(strategy)

        


def create_splats_with_optimizers(
    parser: Parser,
    cfg: Config,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        if init_num_pts < points.shape[0]:
            sampled_pts_idx = np.random.choice(points.shape[0], init_num_pts, replace=False)
        else:
            sampled_pts_idx = np.arange(points.shape[0])
        # randomly sample points from the SfM points
        points = points[sampled_pts_idx]
        rgbs = rgbs[sampled_pts_idx]
    elif init_type == "pretrained":
        assert cfg.pretrained_path is not None
        ckpt = torch.load(cfg.pretrained_path)["splats"]
        if init_num_pts < ckpt["means"].shape[0]:
            sampled_pts_idx = np.random.choice(ckpt["means"].shape[0], init_num_pts, replace=False)
        else:
            sampled_pts_idx = np.arange(ckpt["means"].shape[0])
        points = ckpt["means"][sampled_pts_idx]
        rgbs = torch.rand((points.shape[0], 3))
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    
    if init_type == "pretrained":
        scales = ckpt["scales"][sampled_pts_idx]
        quats = ckpt["quats"][sampled_pts_idx]
        opacities = ckpt["opacities"][sampled_pts_idx]
    else:
        N = points.shape[0]
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]
    # print(f"points: {points.min()}, {points.max()}")
    # print(f"scales: {scales.min()}, {scales.max()}")
    # print(f"quats: {quats.min()}, {quats.max()}")
    # print(f"opacities: {opacities.min()}, {opacities.max()}")

    constants = {}

    # SH coefficients
    if feature_dim is None:
        # color is SH coefficients.
        if init_type == "pretrained":
            params.append(("sh0", torch.nn.Parameter(ckpt["sh0"][sampled_pts_idx]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(ckpt["shN"][sampled_pts_idx]), 2.5e-3 / 20))
        else:
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))  

    if cfg.model_type == "textured_gaussians":
        textures = torch.ones(points.shape[0], cfg.texture_resolution, cfg.texture_resolution, 4)
        textures[:, :, :, :3] = 0.1 # init color to low value
        textures[:, :, :, 3:] = 1.0 # init alpha to 1.0
        params.append(("textures", torch.nn.Parameter(textures), 2.5e-3))
    elif cfg.model_type == "a2tg":
        textures = torch.ones(points.shape[0], cfg.texture_resolution, cfg.texture_resolution, 4)
        textures[:, :, :, :3] = 0.1 # init color to low value
        textures[:, :, :, 3:] = 1.0 # init alpha to 1.0

        N, H, W, C = textures.shape
        assert C == 4, "Expected 4 channels (RGBA)"
        textures_rgb = textures[:, :, :, :4]  # (N, H, W, 4)

        # Rearrange to (N, 3, H, W) then flatten to pack all textures
        textures_rgb = textures_rgb.permute(0, 3, 1, 2).contiguous()  # (N, 4, H, W)
        textures_packed = textures_rgb.reshape(4, -1)  # (4, N*H*W)
        textures_packed[:3, :] = 0.1
        textures_packed[3, :] = 1.0

        # Texture dimensions per texture (same for all)
        # H,W
        texture_dims = torch.tensor([[H, W]] * N, dtype=torch.int32, requires_grad=False, device=device)

        # Compute offsets for each texture in packed array
        pixels_per_texture = H * W
        texture_offsets = torch.arange(N, dtype=torch.int32, requires_grad=False, device=device).unsqueeze(1) * pixels_per_texture  # (N, 1)

        params.append(("textures_packed", torch.nn.Parameter(textures_packed), 2.5e-3))
        constants["texture_dims"] = texture_dims
        constants["texture_offsets"] = texture_offsets


    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers, constants

HIST_MIN_VAL = 1.0
HIST_MAX_VAL = 5000
HIST_BINS = 1000

class RenderMode(Enum):
    RGB='RGB'
    GRAD='grad'
    TEX_SIZE='texture size'
    SH='sh'

class TrainViewer(UtilViewer):
    def __init__(self, *args, **kwargs):
        self.gui_handles = {
            'render_mode': None,
            'texture_size_plot': None,
            'grad_plot': None
        }
        self.render_mode = 'RGB'
        super().__init__(*args, **kwargs)


    def _init_rendering_tab(self):
        super()._init_rendering_tab()
        self._visualization_folder = self.server.gui.add_folder("Visualization")

    def _populate_rendering_tab(self):
        super()._populate_rendering_tab()
        with self._visualization_folder:
            render_mode_dropdown = self.server.gui.add_dropdown(
                label='render mode',
                options=[
                    RenderMode.RGB,
                    RenderMode.GRAD,
                    RenderMode.TEX_SIZE,
                    RenderMode.SH
                ],
                initial_value=RenderMode.RGB
            )
            self.gui_handles['render_mode'] = render_mode_dropdown
            grad2d_slider = self.server.gui.add_slider(
                label='grad2d threshold',
                min=0.0,
                max=0.01,
                step=0.0001,
                initial_value=0.001,
                visible=False
            )
            self.gui_handles['grad2d_slider'] = grad2d_slider
            grad2d_count = self.server.gui.add_number(
                label='grad2d count',
                initial_value=0,
                min=0,
                max=1000000,
                step=1,
                disabled=True,
                visible=False
            )
            self.gui_handles['grad2d_count'] = grad2d_count

            @grad2d_slider.on_update
            def _(_) -> None:
                self.rerender(_)

            @render_mode_dropdown.on_update
            def _(_):
                grad2d_slider.visible = False
                grad2d_count.visible = False
                match render_mode_dropdown.value:
                    case RenderMode.RGB:
                        pass
                    case RenderMode.GRAD:
                        grad2d_slider.visible = True
                        grad2d_count.visible = True
                    case RenderMode.TEX_SIZE:
                        pass
                    case RenderMode.SH:
                        pass
                self.rerender(_)

            self.gui_handles['texture_size_plot'] = self.server.gui.add_plotly(
                figure=go.Figure()
            )
            self.server.gui.add_markdown(
                content='grad plot'
            )
            self.gui_handles['grad_plot'] = self.server.gui.add_plotly(
                figure=go.Figure()
            )

    def set_plot(
            self, 
            name: str, 
            histc_input: torch.Tensor, 
            title: str,
            hist_bins: int,
            hist_min_val: int,
            hist_max_val: int
        ) -> None:
        """Generates a histogram plot for given tensor data, using predefined constants."""
        with torch.no_grad():
            hist = torch.histc(histc_input, bins=hist_bins, min=hist_min_val, max=hist_max_val).cpu().detach().numpy()

        bin_edges = torch.linspace(hist_min_val, hist_max_val, steps=hist_bins + 1).cpu().numpy()
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        fig = px.histogram(
            x=bin_centers,
            y=hist,
            nbins=hist_bins,
            labels={'x': 'Value', 'y': 'Count'},
            title=title
        ).update_layout(margin=dict(l=10, r=10, t=30, b=10))
        self.gui_handles[name].figure = fig

    def plot_3d_histogram_bars(
        self,
        data: np.ndarray,
        bins=(20, 20),
        bar_scale=0.9,
        colorscale="Viridis",
        renderer=None,          # e.g. "browser", "vscode", "notebook_connected"
        save_html=None          # e.g. "3d_hist.html"
    ):
        """
        Plot a 3D bar-chart histogram for 2D points using Plotly Mesh3d.

        data: (n,2) array of (x,y) points
        bins: (bx, by) or [edges_x, edges_y]
        bar_scale: 0..1, shrink bars inside each bin so gaps are visible
        """
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must have shape (n, 2)")

        # 2D histogram
        H, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=bins)

        # bin centers and widths (per-bin to be safe)
        x_cent = 0.5*(xedges[:-1] + xedges[1:])
        y_cent = 0.5*(yedges[:-1] + yedges[1:])
        x_w = np.diff(xedges)
        y_w = np.diff(yedges)

        # Collect vertices and faces for all cuboids
        X, Y, Z = [], [], []
        I, J, K = [], [], []
        intensity = []

        def add_bar(x0, x1, y0, y1, z):
            """Add one cuboid [x0,x1]x[y0,y1]x[0,z] as 8 verts + 12 triangles."""
            base = len(X)
            # order: 0..3 bottom, 4..7 top (see diagram in code comments)
            xs = [x0, x1, x1, x0, x0, x1, x1, x0]
            ys = [y0, y0, y1, y1, y0, y0, y1, y1]
            zs = [0, 0, 0, 0, z, z, z, z]
            X.extend(xs); Y.extend(ys); Z.extend(zs)
            intensity.extend([z]*8)

            # 12 triangles (two per face)
            faces = [
                (0,1,2),(0,2,3),       # bottom
                (4,6,5),(4,7,6),       # top
                (0,5,1),(0,4,5),       # side x+
                (1,6,2),(1,5,6),       # side y+
                (2,7,3),(2,6,7),       # side x-
                (3,4,0),(3,7,4)        # side y-
            ]
            for a,b,c in faces:
                I.append(base+a); J.append(base+b); K.append(base+c)

        # build bars
        for ix, xc in enumerate(x_cent):
            dx = x_w[ix]*bar_scale
            x0, x1 = xc - dx/2, xc + dx/2
            for iy, yc in enumerate(y_cent):
                count = int(H[ix, iy])  # H is (len(xedges)-1, len(yedges)-1)
                if count <= 0:
                    continue
                dy = y_w[iy]*bar_scale
                y0, y1 = yc - dy/2, yc + dy/2
                add_bar(x0, x1, y0, y1, count)

        fig = go.Figure(go.Mesh3d(
            x=X, y=Y, z=Z,
            i=I, j=J, k=K,
            intensity=intensity, colorscale=colorscale, showscale=True,
            flatshading=True, opacity=1.0,
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1),
            lightposition=dict(x=100, y=200, z=0)
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title="Texture Width",
                yaxis_title="Y",
                zaxis_title="Count",
                aspectmode="data"
            ),
            title="3D Histogram (bars) from 2D data"
        )

        self.gui_handles['texture_size_plot'].figure = fig

class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"
        self.step = 0 # current optimization step

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.dataset == "colmap":
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=True,
                test_every=cfg.test_every,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
            self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        elif cfg.dataset == "blender":
            self.parser = None
            if cfg.background_mode == "white":
                bg_color = (255, 255, 255)
            else:
                bg_color = (0, 0, 0)
            self.trainset = BlenderDataset(data_dir=cfg.data_dir, split="train", bg_color=bg_color)
            self.valset = BlenderDataset(data_dir=cfg.data_dir, split="val", bg_color=bg_color)
            self.scene_scale = 1.0 # no scaling required
        else:
            raise ValueError(f"Dataset mode {cfg.dataset} not supported!")

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.init_type = cfg.init_type
        self.splats, self.optimizers, self.constants = create_splats_with_optimizers(
            self.parser,
            self.cfg,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        if self.model_type in ["2dgs", "textured_gaussians", "a2tg"]:
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient=key_for_gradient,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.cfg.strategy.initialize_state()

        self.texture_strategy = TextureStrategy(
            min_tex_res=self.cfg.min_tex_res,
            max_tex_res=self.cfg.max_tex_res,
            min_aspect_ratio=self.cfg.min_aspect_ratio,
            max_scale_for_thin=self.cfg.max_scale_for_thin,
            upscale_grad2d=self.cfg.upscale_grad2d,
            upscale_start_iter=self.cfg.upscale_start_iter,
            upscale_stop_iter=self.cfg.upscale_stop_iter,
            upscale_every=self.cfg.upscale_every,
            reset_every=self.cfg.reset_every,
            absgrad=self.cfg.absgrad,
            verbose=True
        )
        self.texture_strategy_state = self.texture_strategy.initialize_state()

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = TrainViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Rasterization modes data
        self.render_mode_data = {
            RenderMode.RGB: {},
            RenderMode.GRAD: {},
            RenderMode.TEX_SIZE: {},
            RenderMode.SH: {}
        }

    def get_splat_mem_size_bytes(self) -> Dict:
        print(self.splats.keys())
        print(self.cfg.init_type)
        mem_size_stats = {
            "means": self.splats["means"].numel() * 4,
            "scales": self.splats["scales"][:,:2].numel() * 4,
            "quats": self.splats["quats"].numel() * 4,
            "opacities": self.splats["opacities"].numel() * 4,
        }
        if "sh0" in self.splats.keys():
            mem_size_stats["sh0"] = self.splats["sh0"].numel() * 4
            mem_size_stats["shN"] = self.splats["shN"].numel() * 4
        else:
            mem_size_stats["features"] = self.splats["features"].numel() * 4
        if self.model_type == 'a2tg':
            texture_channels = 0
            if self.cfg.textured_rgb:
                texture_channels += 3
            if self.cfg.textured_alpha:
                texture_channels += 1
            mem_size_stats["textures_packed"]= self.splats["textures_packed"].shape[1] * texture_channels * 4
            # mem_size_stats["texture_dims"] = self.constants["texture_dims"].numel() * 4
        elif self.model_type == "textured_gaussians":
            mem_size_stats["textures"] = self.splats["textures"].numel()*4
        
        return mem_size_stats

    def get_textures(self):
        # textures: [N, L, L, 4]
        textures = self.splats["textures"]
        if not self.cfg.textured_rgb:
            rgb_textures = torch.zeros_like(textures[..., :3]) # [N, L, L, 3]
        else:
            rgb_textures = textures[..., :3] # [N, L, L, 3]
        if not self.cfg.textured_alpha:
            alpha_textures = torch.ones_like(textures[..., 3:4]) # [N, L, L, 1]
        else:
            alpha_textures = textures[..., 3:4] # [N, L, L, 1]
            alpha_textures = alpha_textures / (alpha_textures.amax(dim=[1, 2], keepdim=True) + 1e-6) # normalize so that the max is 1
        textures = torch.cat([rgb_textures, alpha_textures], dim=-1) # [N, L, L, 4]
        textures = textures.clamp(0.0, 1.0)
        return textures

    def get_textures_packed(self):
        # textures_packed: [3, \sum(N_i * H_i * W_i)]
        textures_packed = self.splats["textures_packed"]
        if not self.cfg.textured_rgb:
            rgb_textures = torch.zeros_like(textures_packed[:3, :]) # [3, \sum(N_i * H_i * W_i)]
        else:
            rgb_textures = textures_packed[:3, :]
        if not self.cfg.textured_alpha:
            alpha_textures = torch.ones_like(textures_packed[3:4, :])
        else:
            alpha_textures = textures_packed[3:4, :]
            alpha_textures = alpha_textures / (alpha_textures.amax(dim=1, keepdim=True) + 1e-6) # normalize so that the max is 1
        textures_packed = torch.cat([rgb_textures, alpha_textures], dim=0) # [4, \sum(N_i * H_i * W_i)]
        textures_packed = textures_packed.clamp(0.0, 1.0)
        return textures_packed

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]

        opacities = torch.sigmoid(self.splats["opacities"]) # [N,]
        

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        extra = {}

        if self.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                info,
            ) = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "a2tg":
            # textures = self.get_textures()
            textures_packed = self.get_textures_packed()
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                gs_contrib_count,
                gs_contrib_sum,
                gs_weight_sum,
                gs_dx_sum,
                gs_dy_sum,
                info,
            ) = rasterization_packed_textured_gaussians(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                textures_packed=textures_packed,  # [3, \sum(N_i * H_i * W_i)]
                texture_dims=self.constants["texture_dims"],  # [N, 2]
                texture_offsets= self.constants["texture_offsets"],  # [N, 1]
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
            extra['gs_contrib_count'] = gs_contrib_count
            extra['gs_contrib_sum'] = gs_contrib_sum
        elif self.model_type == "textured_gaussians":
            textures = self.get_textures()
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                gs_contrib_count,
                gs_contrib_sum,
                info,
            ) = rasterization_textured_gaussians(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                textures=textures,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
            extra['gs_contrib_count'] = gs_contrib_count
            extra['gs_contrib_sum'] = gs_contrib_sum
        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            extra,
            info,
        )
    
    def rasterize_splats_with_mode(
        self,
        mode: str,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]

        opacities = torch.sigmoid(self.splats["opacities"]) # [N,]
        
        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        match mode:
            case RenderMode.RGB:
                pass
            case RenderMode.GRAD:
                grad2d = self.texture_strategy_state['grad2d']
                threshold = self.viewer.gui_handles['grad2d_slider'].value
                mask = (grad2d / (self.step % self.cfg.upscale_every) * 100.0) > threshold
                self.viewer.gui_handles['grad2d_count'].value = mask.sum().item()
                # set opacity to full
                # opacities[mask] = 1.0
                # set color to red
                colors[mask,0,:] = rgb_to_sh(torch.Tensor([1.0, 0.0, 0.0]).cuda())

                # TODO: set texture to black
            case RenderMode.TEX_SIZE:
                # TODO: set color to texture size
                # TODO: set texture to black
                pass
            case RenderMode.SH:
                # TODO: set texture to black
                pass

        extra = {}

        if self.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                info,
            ) = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "a2tg":
            textures_packed = self.get_textures_packed()
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                _,
                _,
                _,
                info,
            ) = rasterization_packed_textured_gaussians(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                textures_packed=textures_packed,  # [3, \sum(N_i * H_i * W_i)]
                texture_dims=self.constants["texture_dims"],  # [N, 2]
                texture_offsets= self.constants["texture_offsets"],  # [N, 1]
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "textured_gaussians":
            textures = self.get_textures()
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                _,
                _,
                _,
                info,
            ) = rasterization_textured_gaussians(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                textures=textures,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            extra,
            info,
        )

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        # with open(f"{cfg.result_dir}/cfg.json", "w") as f:
        #     json.dump(vars(cfg), f)

        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)


        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter  = iter(trainloader)

        if not self.cfg.disable_viewer:
            self.viewer.custom_update(self.trainset, self.valset)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:

            self.step = step

            if not cfg.disable_viewer:
                if step in cfg.pause_steps:
                    self.viewer._training_tab_handles['pause_train_button'].visible = \
                        not self.viewer._training_tab_handles['pause_train_button'].visible
                    self.viewer._training_tab_handles['resume_train_button'].visible = \
                        not self.viewer._training_tab_handles['resume_train_button'].visible
                    if self.viewer.state != "completed":
                        self.viewer.state = "paused" if self.viewer.state == "training" else "training"
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]
            if cfg.alpha_loss:
                alphas_gt = data["alpha"].to(device) # [1, H, W]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            if self.cfg.pretrained_path is not None:
                sh_degree_to_use = cfg.sh_degree
            else:
                sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                extra,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
                distloss=self.cfg.dist_loss,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            
            if cfg.background_mode is not None:
                if cfg.background_mode == "random":
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)
                elif cfg.background_mode == "white":
                    colors = colors + 1.0 * (1.0 - alphas)
                elif cfg.background_mode == "black":
                    colors = colors + 0.0 * (1.0 - alphas)
                else:
                    raise ValueError(f"Background mode {cfg.background_mode} not supported!")


            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            if self.cfg.model_type == 'a2tg':
                self.texture_strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.texture_strategy_state,
                    step=step,
                    info=info
                )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda
            
            if cfg.alpha_loss:
                alphas = alphas.squeeze(-1) # [1, H, W]
                alpha_error = (alphas - alphas_gt).abs().mean()
                alpha_loss = cfg.alpha_lambda * alpha_error
                loss += alpha_loss

            if cfg.scale_loss:
                max_scale = torch.exp(self.splats["scales"]).amax(dim=-1)
                scale_loss = cfg.scale_lambda * max_scale.mean()
                loss += scale_loss

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.normal_loss:
                desc += f"normal loss={normalloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            if cfg.alpha_loss:
                desc += f"alpha loss={alpha_loss.item():.6f}| "
            if cfg.scale_loss:
                desc += f"scale loss={scale_loss.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                if step > self.texture_strategy.upscale_start_iter and step < self.texture_strategy.upscale_stop_iter:
                    grad2d: torch.Tensor = self.texture_strategy_state['grad2d'].clone() / (self.step % self.cfg.upscale_every) * 100.
                    with torch.no_grad():
                        hist = torch.histc(grad2d, bins=100, min=0.0, max=0.01).cpu().detach().numpy()
                    self.writer.add_histogram(
                        "grad2d/histogram", 
                        hist,
                        step
                    )
                    self.writer.add_scalar("grad2d/mean", grad2d.mean(), step)
                    self.writer.add_scalar("grad2d/median", grad2d.median(), step)
                    self.writer.add_scalar("grad2d/max", grad2d.max(), step)
                self.writer.flush()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            if self.model_type == "a2tg":
                self.texture_strategy.step_post_backward(
                        constants=self.constants,
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.texture_strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                if self.cfg.model_type == "a2tg":
                    torch.save(
                        {
                            "step": step,
                            "splats": self.splats.state_dict(),
                            "texture_dims": self.constants['texture_dims'],
                            "texture_offsets": self.constants['texture_offsets']
                        },
                        f"{self.ckpt_dir}/ckpt_{step}.pt",
                    )
                else:
                    torch.save(
                        {
                            "step": step,
                            "splats": self.splats.state_dict(),
                        },
                        f"{self.ckpt_dir}/ckpt_{step}.pt",
                    )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

                # if step > self.texture_strategy.upscale_start_iter and step % 100 == 0:
                #     grad2d = self.texture_strategy_state["grad2d"]
                #     self.viewer.set_plot(
                #         name='grad_plot', 
                #         histc_input=grad2d/(step % self.cfg.upscale_every), 
                #         title='gradient 2d abs',
                #         hist_bins=1000,
                #         hist_min_val=0,
                #         hist_max_val=0.1
                #     )

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=2
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "val_frame_time": []}
        per_view_metrics = {}
        for i, data in tqdm.tqdm(enumerate(valloader)):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            colors = colors[..., :3]  # Take RGB channels

            if cfg.background_mode is not None:
                if cfg.background_mode == "random":
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)
                elif cfg.background_mode == "white":
                    colors = colors + 1.0 * (1.0 - alphas)
                elif cfg.background_mode == "black":
                    colors = colors + 0.0 * (1.0 - alphas)
                else:
                    raise ValueError(f"Background mode {cfg.background_mode} not supported!")
            colors = torch.clamp(colors, 0.0, 1.0)

                
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            # write median depths
            # render_median = (render_median - render_median.min()) / (
            #     render_median.max() - render_median.min()
            # )
            # # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            # render_median = (
            #     render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()
            # )

            # imageio.imwrite(
            #     f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
            #     (render_median * 255).astype(np.uint8),
            # )

            # # write normals
            # normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            # normals_output = (normals * 255).astype(np.uint8)
            # imageio.imwrite(
            #     f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
            # )

            # # write normals from depth
            # normals_from_depth *= alphas.squeeze(0).detach()
            # normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            # normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
            #     np.max(normals_from_depth) - np.min(normals_from_depth)
            # )
            # normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            # if len(normals_from_depth_output.shape) == 4:
            #     normals_from_depth_output = normals_from_depth_output.squeeze(0)
            # imageio.imwrite(
            #     f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
            #     normals_from_depth_output,
            # )

            # # write distortions
            # render_dist = render_distort
            # dist_max = torch.max(render_dist)
            # dist_min = torch.min(render_dist)
            # render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            # render_dist = (
            #     colormap(render_dist.cpu().numpy()[0])
            #     .permute((1, 2, 0))
            #     .numpy()
            #     .astype(np.uint8)
            # )
            # imageio.imwrite(
            #     f"{self.render_dir}/val_{i:04d}_distortions_{step}.png", render_dist
            # )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            psnr = self.psnr(colors, pixels)
            ssim = self.ssim(colors, pixels)
            lpips = self.lpips(colors, pixels)  
            per_view_metrics[data["image_name"][0]] = {
                "id": data['image_id'].item(),
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item()
            }
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["lpips"].append(lpips)

        ellipse_time /= len(valloader)

        for i, data in tqdm.tqdm(enumerate(valloader)):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            frame_time = time.time() - tic
            metrics["val_frame_time"].append(frame_time)


        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        fps = 1.0 / np.array(metrics["val_frame_time"]).mean()
        mem_stats = self.get_splat_mem_size_bytes()
        total_size = sum(mem_stats.values())
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])} "
            f"Total mem in MB: {total_size / (1000**2)}"
            f"val FPS: {fps}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
            **mem_stats,
            "mem_mb": total_size / (1000**2),
            "val_fps": fps
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        with open(f"{self.stats_dir}/val_step{step:04d}_per_view.json", "w") as f:
            json.dump(per_view_metrics, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        if cfg.dataset == "colmap":
            camtoworlds = self.parser.camtoworlds[5:-5]
            camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
                ],
                axis=1,
            )  # [N, 4, 4]

            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
            K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
            width, height = list(self.parser.imsize_dict.values())[0]
        elif cfg.dataset == "blender":
            camtoworlds = np.stack(self.trainset.camtoworlds) # [N, 4, 4]
            camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
                ],
                axis=1,
            )  # [N, 4, 4]
            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
            K = torch.from_numpy(self.trainset.K).float().to(device)
            width, height = self.trainset.image_size, self.trainset.image_size

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, surf_normals, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=1 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to videowith
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _, _, _, _, _, _ = self.rasterize_splats_with_mode(
            mode=self.viewer.gui_handles['render_mode'].value,
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        for k in runner.constants.keys():
            runner.constants[k] = ckpt[k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    # if not cfg.disable_viewer:
    #     print("Viewer running... Ctrl+C to exit.")
    #     time.sleep(1000000)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                strategy=MCMCStrategy(verbose=False),
            ),
        ),
    }
    # cfg = tyro.cli(Config)
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
