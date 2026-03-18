import argparse
import math
import os
import time
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

import imageio
import plotly.express as px
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm
import viser
import nerfview
from pathlib import Path
from pycolmap.rotation import Quaternion
import numpy as np

class UtilViewer(nerfview.Viewer):
    def __init__(self, *args, **kwargs):
        self.train_dataset = None
        self.val_dataset = None
        self.selected_train_idx = 0
        self.selected_val_idx = 0
        self.selected_train_idx_gui = None
        self.selected_val_idx_gui = None
        self.train_frustums = []
        self.val_frustums = []

        super().__init__(*args, **kwargs)


    def _init_rendering_tab(self):
        super()._init_rendering_tab()
        self._dataset_render_handles = {}
        self._dataset_rendering_folder = self.server.gui.add_folder("Dataset")

    def _populate_rendering_tab(self):
        super()._populate_rendering_tab()

        with self._dataset_rendering_folder:
            self.selected_train_idx_gui = self.server.gui.add_markdown(
                content=f"Selected Train Index: 0"
            )
            self.selected_val_idx_gui = self.server.gui.add_markdown(
                content=f"Selected Validation Index: 0"
            )


    def custom_update(self, train_dataset=None, val_dataset=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        def make_on_click_frustum_train(frustum: viser.CameraFrustumHandle, idx):
            @frustum.on_click
            def _(_,) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frustum.wxyz
                    client.camera.position = frustum.position
                    self.selected_train_idx = idx
                    self.selected_train_idx_gui.content = f"Selected Train Index: {self.selected_train_idx}"

        def make_on_click_frustum_val(frustum: viser.CameraFrustumHandle, idx):
            @frustum.on_click
            def _(_,) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frustum.wxyz
                    client.camera.position = frustum.position
                    self.selected_val_idx = idx
                    self.selected_val_idx_gui.content = f"Selected Validation Index: {self.selected_val_idx}"
                    
        if train_dataset:
            for i, data in enumerate(train_dataset):
                image_name = data["image_name"]
                image = data["image"].detach().cpu().numpy() / 255.0
                image_id = data["image_id"]
                camtoworld = data["camtoworld"].detach().cpu().numpy()
                position = camtoworld[:3, 3]
                wxyz = Quaternion.FromR(camtoworld[:3, :3]).q
                K = data["K"]
                fx = K[0, 0].detach().cpu().numpy()
                fy = K[1, 1].detach().cpu().numpy()
                fov_x = 2 * np.arctan(image.shape[0] / (2 * fx))
                fov_y = 2 * np.arctan(image.shape[1] / (2 * fy))
                frustum = self.server.scene.add_camera_frustum(
                    f"/train_dataset/{image_name}",
                    fov=fov_x,
                    aspect=image.shape[1] / image.shape[0],
                    scale=0.05,
                    position=position,
                    wxyz=wxyz,
                    color=(20, 200, 20),
                    image=image
                )
                
                self.train_frustums.append(frustum)
                make_on_click_frustum_train(frustum=frustum, idx=i)

                self.server.scene.add_label(
                    name=f"/train_label/{image_name}",
                    text=f"/train/{image_name}",
                    position=position,
                )

        if val_dataset:
            for i, data in enumerate(val_dataset):
                image_name = data["image_name"]
                image = data["image"].detach().cpu().numpy() / 255.0
                image_id = data["image_id"]
                camtoworld = data["camtoworld"].detach().cpu().numpy()
                position = camtoworld[:3, 3]
                wxyz = Quaternion.FromR(camtoworld[:3, :3]).q
                K = data["K"]
                fx = K[0, 0].detach().cpu().numpy()
                fy = K[1, 1].detach().cpu().numpy()
                fov_x = 2 * np.arctan(image.shape[0] / (2 * fx))
                fov_y = 2 * np.arctan(image.shape[1] / (2 * fy))
                frustum = self.server.scene.add_camera_frustum(
                    f"/val_dataset/{image_name}",
                    fov=fov_x,
                    aspect=image.shape[1] / image.shape[0],
                    scale=0.05,
                    position=position,
                    wxyz=wxyz,
                    color=(200, 20, 20),
                    image=image
                )
                self.val_frustums.append(frustum)
                make_on_click_frustum_val(frustum=frustum, idx=i)

                self.server.scene.add_label(
                    name=f"/test_label/{image_name}",
                    text=f"/test/{image_name}",
                    position=position,
                )
