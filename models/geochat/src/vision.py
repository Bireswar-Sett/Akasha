from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
)


class _CLIPVisionWrapper(nn.Module):
    """
    Wrapper preserving the original GeoChat checkpoint hierarchy:

        vision_tower.vision_tower.vision_model.*
    """

    def __init__(self, model_name: str):
        super().__init__()

        config = CLIPVisionConfig.from_pretrained(
            model_name
        )

        self.vision_model = CLIPVisionModel(config)

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)


class GeoChatVisionTower(nn.Module):

    def __init__(self, config,):
        super().__init__()

        self.vision_tower_name = config.mm_vision_tower
        self.select_layer = config.mm_vision_select_layer
        self.select_feature = config.mm_vision_select_feature

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        self.vision_tower = _CLIPVisionWrapper(
            self.vision_tower_name
        )

        self.vision_tower.requires_grad_(False)

    def feature_select(self, vision_outputs):
        image_features = vision_outputs.hidden_states[
            self.select_layer
        ]

        if self.select_feature == "patch":
            image_features = image_features[:, 1:]

        elif self.select_feature == "cls_patch":
            pass

        else:
            raise ValueError(
                f"Unsupported vision feature selection: "
                f"{self.select_feature!r}"
            )

        return image_features

    @torch.no_grad()
    def forward(
        self,
        pixel_values: Union[
            torch.Tensor,
            list[torch.Tensor],
        ],
    ):
        if isinstance(pixel_values, list):
            features = []

            for image in pixel_values:
                if image.ndim == 3:
                    image = image.unsqueeze(0)

                image = image.to(
                    device=self.device,
                    dtype=self.dtype,
                )

                outputs = self.vision_tower(
                    pixel_values=image,
                    output_hidden_states=True,
                )

                selected = self.feature_select(outputs)
                features.append(selected)

            return features

        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        pixel_values = pixel_values.to(
            device=self.device,
            dtype=self.dtype,
        )

        outputs = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        return self.feature_select(outputs)

    @property
    def dtype(self):
        return next(
            self.vision_tower.parameters()
        ).dtype

    @property
    def device(self):
        return next(
            self.vision_tower.parameters()
        ).device

    @property
    def config(self):
        return self.vision_tower.vision_model.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size

        return (image_size // patch_size) ** 2