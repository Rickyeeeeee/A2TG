import argparse
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import viser
from torch import Tensor

from a2tg.distributed import cli
from a2tg.rendering import (
    rasterization_2dgs,
    rasterization_packed_textured_gaussians,
    rasterization_textured_gaussians,
)
from datasets.colmap import BlenderDataset, Dataset, Parser
from util_viewer import UtilViewer
from utils import rgb_to_sh


class ModelType(str, Enum):
    AUTO = "auto"
    D2GS = "2dgs"
    TEXTURED_GAUSSIANS = "textured_gaussians"
    A2TG = "a2tg"


class RenderMode(Enum):
    RGB = "RGB"
    SH = "SH"
    TEX_SIZE = "texture size"
    NO_TEX = "no texture"
    NO_BASE = "no base"
    SCALE = "scale"
    TEX_ONLY = "texture only"


@dataclass
class GaussianModelBase:
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    sh_degree: int


@dataclass
class DenseTexturedGaussianModel(GaussianModelBase):
    textures: Tensor
    texture_dims: Tensor


@dataclass
class PackedTexturedGaussianModel(GaussianModelBase):
    textures_packed: Tensor
    texture_dims: Tensor
    texture_offsets: Tensor


class UnifiedViewer(UtilViewer):
    def __init__(self, *args, **kwargs):
        self.render_mode = RenderMode.RGB
        self.tex_value = 0
        self.scale_ratio = 3.0
        self.min_scale_value = 0.1
        self.scale_modifier_value = 1.0
        self.opacity_modifier_value = 1.0
        self.sh_level_index = 2  # 0->1, 1->1+2, 2->1+2+3(full)

        super().__init__(*args, **kwargs)

    def _init_rendering_tab(self):
        super()._init_rendering_tab()
        self._visualization_folder = self.server.gui.add_folder("Visualization")

    def _populate_rendering_tab(self):
        super()._populate_rendering_tab()

        with self._visualization_folder:
            scale_modifier_slider = self.server.gui.add_slider(
                label="scale modifier",
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=1.0,
                marks=(1.0,),
            )
            opacity_modifier_slider = self.server.gui.add_slider(
                label="opacity modifier",
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=1.0,
                marks=(1.0,),
            )
            render_mode_dropdown = self.server.gui.add_dropdown(
                label="render mode",
                options=[
                    RenderMode.RGB,
                    RenderMode.SH,
                    RenderMode.NO_TEX,
                    RenderMode.NO_BASE,
                    RenderMode.TEX_SIZE,
                    RenderMode.TEX_ONLY,
                    RenderMode.SCALE,
                ],
                initial_value=RenderMode.RGB,
            )
            tex_slider = self.server.gui.add_slider(
                label="texture size",
                min=0,
                max=6,
                step=1,
                initial_value=0,
                visible=False,
                hint="Threshold is 2^value",
            )
            scale_ratio_slider = self.server.gui.add_slider(
                label="scale proportion",
                min=1.0,
                max=100.0,
                step=0.01,
                initial_value=3.0,
                visible=False,
            )
            min_scale_slider = self.server.gui.add_slider(
                label="min scale",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.1,
                visible=False,
            )
            sh_level_slider = self.server.gui.add_slider(
                label="SH level (0:1, 1:1+2, 2:1+2+3)",
                min=0,
                max=2,
                step=1,
                initial_value=2,
                visible=False,
            )
            sh_level_label = self.server.gui.add_markdown(content="SH level: 1+2+3")

            def _update_mode_control_visibility() -> None:
                tex_visible = self.render_mode in (RenderMode.TEX_SIZE, RenderMode.TEX_ONLY)
                scale_visible = self.render_mode is RenderMode.SCALE
                sh_visible = self.render_mode is RenderMode.SH
                tex_slider.visible = tex_visible
                scale_ratio_slider.visible = scale_visible
                min_scale_slider.visible = scale_visible
                sh_level_slider.visible = sh_visible
                if hasattr(sh_level_label, "visible"):
                    sh_level_label.visible = sh_visible

            @scale_modifier_slider.on_update
            def _(_) -> None:
                self.scale_modifier_value = float(scale_modifier_slider.value)
                self.rerender(_)

            @opacity_modifier_slider.on_update
            def _(_) -> None:
                self.opacity_modifier_value = float(opacity_modifier_slider.value)
                self.rerender(_)

            @tex_slider.on_update
            def _(_) -> None:
                self.tex_value = int(tex_slider.value)
                self.rerender(_)

            @scale_ratio_slider.on_update
            def _(_) -> None:
                self.scale_ratio = float(scale_ratio_slider.value)
                self.rerender(_)

            @min_scale_slider.on_update
            def _(_) -> None:
                self.min_scale_value = float(min_scale_slider.value)
                self.rerender(_)

            @sh_level_slider.on_update
            def _(_) -> None:
                self.sh_level_index = int(sh_level_slider.value)
                labels = ["1", "1+2", "1+2+3"]
                sh_level_label.content = f"SH level: {labels[self.sh_level_index]}"
                self.rerender(_)

            @render_mode_dropdown.on_update
            def _(_) -> None:
                self.render_mode = render_mode_dropdown.value
                _update_mode_control_visibility()
                self.rerender(_)

            _update_mode_control_visibility()


class UnifiedGaussianViewerApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda", args.local_rank)
        self.server = viser.ViserServer(port=args.port, verbose=False)
        self.viewer = UnifiedViewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="rendering",
        )
        self.server.gui.set_panel_label("viewer")

        self.model_type: ModelType | None = None
        self.model: GaussianModelBase | DenseTexturedGaussianModel | PackedTexturedGaussianModel | None = None
        self.trainset = None
        self.valset = None
        self._warned_noop_modes: set[str] = set()
        self._rasterization_2dgs_fn = rasterization_2dgs

    def _load_checkpoint(self, path: str) -> dict[str, Any]:
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _infer_model_type_from_checkpoint(self, ckpt_full: dict[str, Any]) -> ModelType:
        if "splats" not in ckpt_full:
            raise ValueError("Checkpoint is missing top-level 'splats' key.")
        splats = ckpt_full["splats"]
        if "textures_packed" in splats:
            if "texture_dims" not in ckpt_full or "texture_offsets" not in ckpt_full:
                raise ValueError(
                    "Checkpoint contains packed textures but is missing top-level "
                    "'texture_dims' and/or 'texture_offsets' required for a2tg."
                )
            return ModelType.A2TG
        if "textures" in splats:
            return ModelType.TEXTURED_GAUSSIANS
        return ModelType.D2GS

    def _resolve_model_type(self, inferred: ModelType) -> ModelType:
        if self.args.model_type == ModelType.AUTO.value:
            return inferred
        requested = ModelType(self.args.model_type)
        if requested != inferred:
            raise ValueError(
                f"Checkpoint type mismatch: requested '{requested.value}', "
                f"but checkpoint looks like '{inferred.value}'."
            )
        return requested

    def _normalize_dense_textures(self, textures: Tensor) -> Tensor:
        rgb_textures = textures[..., :3]
        alpha_textures = textures[..., 3:4]
        alpha_textures = alpha_textures / (alpha_textures.amax(dim=[1, 2], keepdim=True) + 1e-6)
        return torch.cat([rgb_textures, alpha_textures], dim=-1).clamp(0.0, 1.0)

    def _normalize_packed_textures(self, textures_packed: Tensor) -> Tensor:
        rgb_textures = textures_packed[:3, :]
        alpha_textures = textures_packed[3:4, :]
        alpha_textures = alpha_textures / (alpha_textures.amax(dim=1, keepdim=True) + 1e-6)
        return torch.cat([rgb_textures, alpha_textures], dim=0).clamp(0.0, 1.0)

    def _load_models(self) -> None:
        means_list: list[Tensor] = []
        quats_list: list[Tensor] = []
        scales_list: list[Tensor] = []
        opacities_list: list[Tensor] = []
        colors_list: list[Tensor] = []

        dense_textures_list: list[Tensor] = []
        dense_texture_dims_list: list[Tensor] = []
        dense_hw: tuple[int, int] | None = None

        packed_textures_list: list[Tensor] = []
        packed_texture_dims_list: list[Tensor] = []
        packed_texture_offsets_list: list[Tensor] = []
        packed_offset_shift = 0

        resolved_model_type: ModelType | None = None

        for ckpt_path in self.args.ckpt:
            ckpt_full = self._load_checkpoint(ckpt_path)
            inferred_type = self._infer_model_type_from_checkpoint(ckpt_full)
            ckpt_model_type = self._resolve_model_type(inferred_type)

            if resolved_model_type is None:
                resolved_model_type = ckpt_model_type
            elif ckpt_model_type != resolved_model_type:
                raise ValueError(
                    f"All checkpoints must be the same model type. Got "
                    f"'{resolved_model_type.value}' and '{ckpt_model_type.value}'."
                )

            splats = ckpt_full["splats"]
            means_list.append(splats["means"])
            quats_list.append(F.normalize(splats["quats"], p=2, dim=-1))
            scales_list.append(torch.exp(splats["scales"]))
            opacities_list.append(torch.sigmoid(splats["opacities"]))
            colors_list.append(torch.cat([splats["sh0"], splats["shN"]], dim=-2))

            if ckpt_model_type is ModelType.TEXTURED_GAUSSIANS:
                if "textures" not in splats:
                    raise ValueError(f"{ckpt_path} is missing splats['textures'].")
                textures = self._normalize_dense_textures(splats["textures"])
                if textures.ndim != 4 or textures.shape[-1] != 4:
                    raise ValueError(f"Expected dense textures [N,H,W,4], got {tuple(textures.shape)}")
                hw = (int(textures.shape[1]), int(textures.shape[2]))
                if dense_hw is None:
                    dense_hw = hw
                elif hw != dense_hw:
                    raise ValueError(
                        "All dense textured_gaussians checkpoints must share the same "
                        f"texture resolution to be concatenated. Got {dense_hw} and {hw}."
                    )
                dense_textures_list.append(textures)
                dims = torch.tensor(
                    [[hw[0], hw[1]]] * int(textures.shape[0]),
                    dtype=torch.int32,
                    device=textures.device,
                )
                dense_texture_dims_list.append(dims)

            if ckpt_model_type is ModelType.A2TG:
                if "textures_packed" not in splats:
                    raise ValueError(f"{ckpt_path} is missing splats['textures_packed'].")
                if "texture_dims" not in ckpt_full or "texture_offsets" not in ckpt_full:
                    raise ValueError(f"{ckpt_path} is missing top-level texture_dims/texture_offsets.")

                textures_packed = self._normalize_packed_textures(splats["textures_packed"])
                texture_dims = ckpt_full["texture_dims"]
                texture_offsets = ckpt_full["texture_offsets"]
                if texture_offsets.ndim != 2 or texture_offsets.shape[1] != 1:
                    texture_offsets = texture_offsets.view(-1, 1)
                if texture_dims.dtype != torch.int32:
                    texture_dims = texture_dims.to(torch.int32)
                if texture_offsets.dtype != torch.int32:
                    texture_offsets = texture_offsets.to(torch.int32)

                rebased_offsets = texture_offsets + packed_offset_shift
                packed_textures_list.append(textures_packed)
                packed_texture_dims_list.append(texture_dims)
                packed_texture_offsets_list.append(rebased_offsets)
                packed_offset_shift += int(textures_packed.shape[1])

        if resolved_model_type is None:
            raise ValueError("No checkpoints provided.")

        means = torch.cat(means_list, dim=0)
        quats = torch.cat(quats_list, dim=0)
        scales = torch.cat(scales_list, dim=0)
        opacities = torch.cat(opacities_list, dim=0)
        colors = torch.cat(colors_list, dim=0)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        self.model_type = resolved_model_type
        if resolved_model_type is ModelType.D2GS:
            self.model = GaussianModelBase(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh_degree=sh_degree,
            )
        elif resolved_model_type is ModelType.TEXTURED_GAUSSIANS:
            self.model = DenseTexturedGaussianModel(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh_degree=sh_degree,
                textures=torch.cat(dense_textures_list, dim=0),
                texture_dims=torch.cat(dense_texture_dims_list, dim=0),
            )
        elif resolved_model_type is ModelType.A2TG:
            self.model = PackedTexturedGaussianModel(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh_degree=sh_degree,
                textures_packed=torch.cat(packed_textures_list, dim=1),
                texture_dims=torch.cat(packed_texture_dims_list, dim=0),
                texture_offsets=torch.cat(packed_texture_offsets_list, dim=0),
            )
        else:
            raise ValueError(f"Unsupported model type: {resolved_model_type}")

        print(f"Loaded model type: {self.model_type.value}")
        print(f"Number of Gaussians: {len(self.model.means)}")

    def _load_datasets(self) -> None:
        if self.args.dataset == "none":
            return
        if not self.args.data_dir:
            raise ValueError("--data-dir must be set when --dataset is not 'none'.")

        if self.args.dataset == "colmap":
            parser = Parser(
                data_dir=self.args.data_dir,
                factor=self.args.data_factor,
                normalize=True,
                test_every=self.args.test_every,
            )
            self.trainset = Dataset(parser, split="train")
            self.valset = Dataset(parser, split="val")
            return

        if self.args.dataset == "blender":
            self.trainset = BlenderDataset(data_dir=self.args.data_dir, split="train")
            self.valset = BlenderDataset(data_dir=self.args.data_dir, split="val")
            return

        raise ValueError(f"Unsupported dataset mode: {self.args.dataset}")

    def _warn_unsupported_mode_once(self, mode: RenderMode) -> None:
        key = f"{self.model_type.value}:{mode.name}"
        if key not in self._warned_noop_modes:
            self._warned_noop_modes.add(key)
            print(
                f"[viewer] Render mode '{mode.name}' is not supported for "
                f"'{self.model_type.value}'. Falling back to RGB."
            )

    def _effective_sh_degree(self, full_sh_degree: int) -> int:
        if self.viewer.render_mode is not RenderMode.SH:
            return full_sh_degree
        if self.viewer.sh_level_index <= 0:
            return 0
        if self.viewer.sh_level_index == 1:
            return min(1, full_sh_degree)
        return full_sh_degree

    def _black_sh(self, colors: Tensor) -> Tensor:
        black = torch.tensor([0.0, 0.0, 0.0], device=colors.device, dtype=colors.dtype)
        return rgb_to_sh(black)

    def _red_sh(self, colors: Tensor) -> Tensor:
        red = torch.tensor([1.0, 0.0, 0.0], device=colors.device, dtype=colors.dtype)
        return rgb_to_sh(red)

    def _blue_sh(self, colors: Tensor) -> Tensor:
        blue = torch.tensor([0.0, 0.0, 1.0], device=colors.device, dtype=colors.dtype)
        return rgb_to_sh(blue)

    def _apply_scale_mode(self, colors: Tensor, scales: Tensor) -> None:
        if scales.shape[1] < 2:
            return
        sx = scales[:, 0].clamp_min(1e-12)
        sy = scales[:, 1].clamp_min(1e-12)
        mask = (sx / sy) > self.viewer.scale_ratio
        mask |= (sy / sx) > self.viewer.scale_ratio
        mask &= sx < self.viewer.min_scale_value
        mask &= sy < self.viewer.min_scale_value
        if mask.any():
            colors[mask, 0, :] = self._red_sh(colors)

    def _tex_valid_mask(self, texture_dims: Tensor) -> Tensor:
        if texture_dims is None or texture_dims.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        u = texture_dims[:, 0]
        v = texture_dims[:, 1]
        threshold = 2 ** int(self.viewer.tex_value)
        if threshold <= 1:
            return torch.zeros_like(u, dtype=torch.bool)
        valid = (u <= threshold) & (v <= threshold) & ~((u == 1) & (v == 1))
        return valid

    def _apply_tex_size_mode(self, colors: Tensor, texture_dims: Tensor) -> None:
        if texture_dims is None or texture_dims.numel() == 0:
            return
        valid = self._tex_valid_mask(texture_dims)
        if valid.numel() == 0 or not valid.any():
            return
        u = texture_dims[:, 0]
        v = texture_dims[:, 1]
        square_mask = valid & (u == v)
        nonsquare_mask = valid & (u != v)
        if square_mask.any():
            colors[square_mask, 0, :] = self._blue_sh(colors)
        if nonsquare_mask.any():
            colors[nonsquare_mask, 0, :] = self._red_sh(colors)

    def _apply_tex_only_mode(self, opacities: Tensor, texture_dims: Tensor) -> None:
        if texture_dims is None or texture_dims.numel() == 0:
            return
        valid = self._tex_valid_mask(texture_dims)
        if valid.numel() == 0:
            return
        # texture threshold <= 1 -> keep all visible (matches existing viewer behaviour)
        if valid.any() or (2 ** int(self.viewer.tex_value)) > 1:
            opacities[~valid] = 0.0

    def _disable_texture_dense(self, textures: Tensor) -> None:
        textures[..., :3] = 0.0
        textures[..., 3:4] = 1.0

    def _disable_texture_packed(self, textures_packed: Tensor) -> None:
        textures_packed[:3, :] = 0.0
        textures_packed[3:4, :] = 1.0

    def _apply_common_mode_mutations(
        self,
        colors: Tensor,
        scales: Tensor,
        opacities: Tensor,
        texture_dims: Tensor | None = None,
        disable_texture_fn=None,
        supports_texture_modes: bool = False,
    ) -> None:
        mode = self.viewer.render_mode
        if mode is RenderMode.SCALE:
            self._apply_scale_mode(colors, scales)
            return

        if mode is RenderMode.NO_BASE:
            if supports_texture_modes:
                colors[:, 0, :] = self._black_sh(colors)
            else:
                self._warn_unsupported_mode_once(mode)
            return

        if mode is RenderMode.NO_TEX:
            if supports_texture_modes and disable_texture_fn is not None:
                disable_texture_fn()
            else:
                self._warn_unsupported_mode_once(mode)
            return

        if mode is RenderMode.SH:
            if supports_texture_modes and disable_texture_fn is not None:
                disable_texture_fn()
            return

        if mode is RenderMode.TEX_SIZE:
            if supports_texture_modes:
                self._apply_tex_size_mode(colors, texture_dims)
                if disable_texture_fn is not None and (2 ** int(self.viewer.tex_value)) > 1:
                    disable_texture_fn()
            else:
                self._warn_unsupported_mode_once(mode)
            return

        if mode is RenderMode.TEX_ONLY:
            if supports_texture_modes:
                self._apply_tex_only_mode(opacities, texture_dims)
            else:
                self._warn_unsupported_mode_once(mode)
            return

        if mode is not RenderMode.RGB:
            # Safety fallback for unexpected enum values.
            self._warn_unsupported_mode_once(mode)

    def _viewer_render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        if self.model is None or self.model_type is None:
            width = getattr(render_tab_state, "viewer_width", 640)
            height = getattr(render_tab_state, "viewer_height", 480)
            return np.zeros((height, width, 3), dtype=np.float32)

        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        with torch.no_grad():
            c2w = torch.as_tensor(camera_state.c2w, device=self.device, dtype=torch.float32)
            K = torch.as_tensor(camera_state.get_K([width, height]), device=self.device, dtype=torch.float32)
            viewmats = torch.linalg.inv(c2w[None])
            Ks = K[None]

            if self.model_type is ModelType.D2GS:
                return self._render_2dgs_frame(viewmats=viewmats, Ks=Ks, width=width, height=height)

            means = self.model.means
            quats = self.model.quats
            mode = self.viewer.render_mode
            scale_modifier = float(self.viewer.scale_modifier_value)
            opacity_modifier = float(self.viewer.opacity_modifier_value)
            tex_threshold_enabled = (2 ** int(self.viewer.tex_value)) > 1
            scales = self.model.scales if scale_modifier == 1.0 else (self.model.scales * scale_modifier)
            sh_degree = self._effective_sh_degree(self.model.sh_degree)

            if self.model_type is ModelType.TEXTURED_GAUSSIANS:
                assert isinstance(self.model, DenseTexturedGaussianModel)
                needs_color_mut = mode in (RenderMode.SCALE, RenderMode.NO_BASE) or (
                    mode is RenderMode.TEX_SIZE and tex_threshold_enabled
                )
                needs_texture_mut = mode in (RenderMode.NO_TEX, RenderMode.SH) or (
                    mode is RenderMode.TEX_SIZE and tex_threshold_enabled
                )
                needs_opacity_mut = (mode is RenderMode.TEX_ONLY) or (opacity_modifier != 1.0)

                colors = self.model.colors.clone() if needs_color_mut else self.model.colors
                if mode is RenderMode.TEX_ONLY and opacity_modifier == 1.0:
                    opacities = self.model.opacities.clone()
                elif opacity_modifier == 1.0:
                    opacities = self.model.opacities
                else:
                    opacities = self.model.opacities * opacity_modifier
                if mode is RenderMode.TEX_ONLY and opacity_modifier != 1.0:
                    opacities = opacities.clone()

                textures = self.model.textures.clone() if needs_texture_mut else self.model.textures
                self._apply_common_mode_mutations(
                    colors=colors,
                    scales=scales,
                    opacities=opacities,
                    texture_dims=self.model.texture_dims,
                    disable_texture_fn=lambda: self._disable_texture_dense(textures),
                    supports_texture_modes=True,
                )
                render_colors, *_ = rasterization_textured_gaussians(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    textures=textures,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                )
                return render_colors[0].detach().cpu().numpy()

            if self.model_type is ModelType.A2TG:
                assert isinstance(self.model, PackedTexturedGaussianModel)
                needs_color_mut = mode in (RenderMode.SCALE, RenderMode.NO_BASE) or (
                    mode is RenderMode.TEX_SIZE and tex_threshold_enabled
                )
                needs_texture_mut = mode in (RenderMode.NO_TEX, RenderMode.SH) or (
                    mode is RenderMode.TEX_SIZE and tex_threshold_enabled
                )

                colors = self.model.colors.clone() if needs_color_mut else self.model.colors
                if mode is RenderMode.TEX_ONLY and opacity_modifier == 1.0:
                    opacities = self.model.opacities.clone()
                elif opacity_modifier == 1.0:
                    opacities = self.model.opacities
                else:
                    opacities = self.model.opacities * opacity_modifier
                if mode is RenderMode.TEX_ONLY and opacity_modifier != 1.0:
                    opacities = opacities.clone()

                textures_packed = (
                    self.model.textures_packed.clone() if needs_texture_mut else self.model.textures_packed
                )
                self._apply_common_mode_mutations(
                    colors=colors,
                    scales=scales,
                    opacities=opacities,
                    texture_dims=self.model.texture_dims,
                    disable_texture_fn=lambda: self._disable_texture_packed(textures_packed),
                    supports_texture_modes=True,
                )
                render_colors, *_ = rasterization_packed_textured_gaussians(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    textures_packed=textures_packed,
                    texture_dims=self.model.texture_dims,
                    texture_offsets=self.model.texture_offsets,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                )
                return render_colors[0].detach().cpu().numpy()

        return np.zeros((height, width, 3), dtype=np.float32)

    def _render_2dgs_frame(self, viewmats: Tensor, Ks: Tensor, width: int, height: int) -> np.ndarray:
        assert self.model_type is ModelType.D2GS
        model = self.model
        assert isinstance(model, GaussianModelBase)

        mode = self.viewer.render_mode
        scale_modifier = float(self.viewer.scale_modifier_value)
        opacity_modifier = float(self.viewer.opacity_modifier_value)
        sh_degree = self._effective_sh_degree(model.sh_degree)

        unsupported_texture_mode = mode in (
            RenderMode.TEX_SIZE,
            RenderMode.NO_TEX,
            RenderMode.NO_BASE,
            RenderMode.TEX_ONLY,
        )
        if unsupported_texture_mode:
            self._warn_unsupported_mode_once(mode)
            mode = RenderMode.RGB

        # Fast path: identical spirit to the original 2dgs viewer (no tensor clones/mutations).
        if mode in (RenderMode.RGB, RenderMode.SH) and scale_modifier == 1.0 and opacity_modifier == 1.0:
            render_colors, *_ = self._rasterization_2dgs_fn(
                means=model.means,
                quats=model.quats,
                scales=model.scales,
                opacities=model.opacities,
                colors=model.colors,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree,
            )
            return render_colors[0].detach().cpu().numpy()

        scales = model.scales if scale_modifier == 1.0 else (model.scales * scale_modifier)
        opacities = model.opacities if opacity_modifier == 1.0 else (model.opacities * opacity_modifier)
        colors = model.colors

        if mode is RenderMode.SCALE:
            colors = model.colors.clone()
            self._apply_scale_mode(colors, scales)

        render_colors, *_ = self._rasterization_2dgs_fn(
            means=model.means,
            quats=model.quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
        )
        return render_colors[0].detach().cpu().numpy()

    def run(self) -> None:
        torch.manual_seed(42)
        self._load_datasets()
        if self.trainset is not None or self.valset is not None:
            self.viewer.custom_update(train_dataset=self.trainset, val_dataset=self.valset)

        print("Viewer running... Ctrl+C to exit.")
        time.sleep(100000)


def main(local_rank: int, world_rank: int, world_size: int, args: argparse.Namespace) -> None:
    del world_rank, world_size
    args.local_rank = local_rank
    app = UnifiedGaussianViewerApp(args)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, nargs="+", required=True, help="checkpoint path(s)")
    parser.add_argument(
        "--model-type",
        type=str,
        default=ModelType.AUTO.value,
        choices=[m.value for m in ModelType],
        help="model type (default: auto)",
    )
    parser.add_argument("--port", type=int, default=8080, help="viewer port")
    parser.add_argument(
        "--dataset",
        type=str,
        default="none",
        choices=["colmap", "blender", "none"],
        help="optional dataset to show train/val frustums",
    )
    parser.add_argument("--data-dir", type=str, default="", help="dataset path")
    parser.add_argument("--data-factor", type=int, default=1, help="dataset downsample factor")
    parser.add_argument("--test-every", type=int, default=8, help="dataset val split stride")
    args = parser.parse_args()

    cli(main, args, verbose=True)
