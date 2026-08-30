from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from transformers import LlamaForCausalLM, LlamaModel

from .configuration_geochat import GeoChatConfig
from .projector import build_projector
from .vision import GeoChatVisionTower


# LLaVA / GeoChat convention.
# This is an internal sentinel and is NOT directly passed to
# the normal token embedding lookup.
IMAGE_TOKEN_INDEX = -200


class GeoChatLlamaModel(LlamaModel):
    """
    LLaMA backbone extended with GeoChat's:

        - CLIP vision tower
        - multimodal projector
        - image-token embedding insertion

    The module hierarchy intentionally follows the original
    GeoChat checkpoint.
    """

    config_class = GeoChatConfig

    def __init__(self, config: GeoChatConfig):
        super().__init__(config)

        self.vision_tower = GeoChatVisionTower(config)

        self.mm_projector = build_projector(
            projector_type=config.mm_projector_type,
            input_dim=config.mm_hidden_size,
            output_dim=config.hidden_size,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_vision_tower(self) -> GeoChatVisionTower:
        return self.vision_tower

    def get_mm_projector(self):
        return self.mm_projector

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def encode_images(self, pixel_values: Tensor) -> Tensor:
        print("\n[DEBUG] encode_images")
        print(f"pixel_values shape: {pixel_values.shape}")
        print(f"pixel_values dtype: {pixel_values.dtype}")
        print(f"pixel_values device: {pixel_values.device}")

        vision_tower = self.vision_tower
        print(
            f"vision device: {vision_tower.device}"
        )
        print(
            f"vision dtype: {vision_tower.dtype}"
        )

        image_features = vision_tower(pixel_values)

        print(
            f"CLIP features shape: {image_features.shape}"
        )
        print(
            f"CLIP features dtype: {image_features.dtype}"
        )
        print(
            f"CLIP features device: {image_features.device}"
        )

        projected = self.mm_projector(image_features)

        print(
            f"Projected shape: {projected.shape}"
        )
        print(
            f"Projected dtype: {projected.dtype}"
        )
        print(
            f"Projected device: {projected.device}"
        )

        return projected

    # ------------------------------------------------------------------
    # Image token handling
    # ------------------------------------------------------------------

    @staticmethod
    def _find_image_positions(
        input_ids: Tensor,
    ) -> list[Tensor]:
        """
        Find IMAGE_TOKEN_INDEX positions for every sample.
        """

        return [
            (row == IMAGE_TOKEN_INDEX).nonzero(
                as_tuple=False
            ).flatten()
            for row in input_ids
        ]

    def _merge_image_features(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        image_features: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ):
        """
        Replace each image sentinel with the corresponding
        sequence of projected visual tokens.

        Example:

            text:
                [A, B, <IMAGE>, C, D]

            visual features:
                [V1 ... V576]

            result:
                [A, B, V1 ... V576, C, D]

        Returns:
            merged_inputs_embeds
            merged_attention_mask
            merged_labels
        """

        batch_size = input_ids.shape[0]

        if image_features.shape[0] != batch_size:
            raise ValueError(
                "Number of image feature batches must match "
                "input batch size."
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device,
            )

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    "labels must have the same shape as input_ids."
                )

        image_positions = self._find_image_positions(input_ids)

        merged_embeddings = []
        merged_attention_masks = []
        merged_labels = []

        for batch_idx in range(batch_size):
            positions = image_positions[batch_idx]

            if positions.numel() == 0:
                merged_embeddings.append(
                    inputs_embeds[batch_idx]
                )

                merged_attention_masks.append(
                    attention_mask[batch_idx]
                )

                if labels is not None:
                    merged_labels.append(
                        labels[batch_idx]
                    )

                continue

            if positions.numel() > 1:
                raise NotImplementedError(
                    "Multiple image tokens per sample are not "
                    "supported yet."
                )

            image_position = positions[0].item()

            before_embeddings = inputs_embeds[
                batch_idx,
                :image_position,
            ]

            after_embeddings = inputs_embeds[
                batch_idx,
                image_position + 1:,
            ]

            visual_tokens = image_features[batch_idx]

            merged = torch.cat(
                [
                    before_embeddings,
                    visual_tokens,
                    after_embeddings,
                ],
                dim=0,
            )

            merged_embeddings.append(merged)

            # ----------------------------------------------------------
            # Attention mask
            # ----------------------------------------------------------

            before_mask = attention_mask[
                batch_idx,
                :image_position,
            ]

            after_mask = attention_mask[
                batch_idx,
                image_position + 1:,
            ]

            visual_mask = torch.ones(
                visual_tokens.shape[0],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            merged_mask = torch.cat(
                [
                    before_mask,
                    visual_mask,
                    after_mask,
                ],
                dim=0,
            )

            merged_attention_masks.append(
                merged_mask
            )

            # ----------------------------------------------------------
            # Labels
            #
            # Visual tokens don't correspond to target text, so their
            # loss labels must be -100.
            # ----------------------------------------------------------

            if labels is not None:
                before_labels = labels[
                    batch_idx,
                    :image_position,
                ]

                after_labels = labels[
                    batch_idx,
                    image_position + 1:,
                ]

                visual_labels = torch.full(
                    (visual_tokens.shape[0],),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )

                merged_label = torch.cat(
                    [
                        before_labels,
                        visual_labels,
                        after_labels,
                    ],
                    dim=0,
                )

                merged_labels.append(
                    merged_label
                )

        # ------------------------------------------------------------------
        # Dense batching
        # ------------------------------------------------------------------

        lengths = {
            embedding.shape[0]
            for embedding in merged_embeddings
        }

        if len(lengths) != 1:
            raise ValueError(
                "All samples must have the same multimodal sequence "
                "length. Batch padding for multimodal inputs is not "
                "implemented yet."
            )

        merged_embeddings = torch.stack(
            merged_embeddings,
            dim=0,
        )

        merged_attention_masks = torch.stack(
            merged_attention_masks,
            dim=0,
        )

        if labels is not None:
            merged_labels = torch.stack(
                merged_labels,
                dim=0,
            )
        else:
            merged_labels = None

        return (
            merged_embeddings,
            merged_attention_masks,
            merged_labels,
        )

    # ------------------------------------------------------------------
    # Multimodal input preparation
    # ------------------------------------------------------------------

    def prepare_multimodal_inputs(
    self,
    input_ids: Tensor,
    pixel_values: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
):
        """
        Convert text IDs + image tensors into the final LLaMA
        embedding sequence.

        IMAGE_TOKEN_INDEX is a sentinel and must never be passed
        directly to the embedding layer.
        """

        if input_ids is None:
            raise ValueError(
                "input_ids must be provided."
            )

        # --------------------------------------------------------------
        # No image: normal text embedding.
        # --------------------------------------------------------------

        if pixel_values is None:
            inputs_embeds = self.embed_tokens(
                input_ids
            )

            return (
                inputs_embeds,
                attention_mask,
                labels,
            )

        # --------------------------------------------------------------
        # Image case
        #
        # Replace the internal -200 sentinel with a valid token ID
        # temporarily so nn.Embedding does not receive -200.
        #
        # The corresponding embedding is removed later and replaced
        # by the visual embeddings.
        # --------------------------------------------------------------

        safe_input_ids = input_ids.clone()

        image_token_mask = (
            safe_input_ids == IMAGE_TOKEN_INDEX
        )

        safe_input_ids[image_token_mask] = (
            self.config.pad_token_id
        )

        inputs_embeds = self.embed_tokens(
            safe_input_ids
        )

        # --------------------------------------------------------------
        # Encode images and replace the sentinel embedding.
        # --------------------------------------------------------------

        image_features = self.encode_images(
            pixel_values
        )

        return self._merge_image_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Standard LLaMA forward pass.

        Multimodal preparation is handled by the causal-LM wrapper.
        """

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )


class GeoChatLlamaForCausalLM(LlamaForCausalLM):
    """
    GeoChat multimodal causal language model.

    Architecture:

        CLIP
          ↓
        projector
          ↓
        visual tokens
          ↓
        LLaMA
          ↓
        language output
    """

    config_class = GeoChatConfig

    def __init__(self, config: GeoChatConfig):
        super().__init__(config)

        # Replace the standard LLaMA backbone with our multimodal one.
        self.model = GeoChatLlamaModel(config)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_model(self) -> GeoChatLlamaModel:
        return self.model

    def get_vision_tower(self) -> GeoChatVisionTower:
        return self.model.get_vision_tower()

    def get_mm_projector(self):
        return self.model.get_mm_projector()

    def encode_images(
        self,
        pixel_values: Tensor,
    ) -> Tensor:
        return self.model.encode_images(
            pixel_values
        )

    # ------------------------------------------------------------------
    # Multimodal forward
    # ------------------------------------------------------------------

    def forward(
    self,
    input_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    pixel_values: Optional[Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[Tensor] = None,
    **kwargs,
):
        """
        GeoChat multimodal causal-LM forward.

        First pass:
            text + image -> multimodal embeddings

        Cached pass:
            one new token + existing KV cache
        """

        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "Either input_ids or inputs_embeds must be provided."
            )

        # ==============================================================
        # FIRST MULTIMODAL PASS
        # ==============================================================

        is_first_multimodal_pass = (
            pixel_values is not None
            and past_key_values is None
            and inputs_embeds is None
            and input_ids is not None
        )

        if is_first_multimodal_pass:

            (
                inputs_embeds,
                attention_mask,
                labels,
            ) = self.model.prepare_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            # ----------------------------------------------------------
            # The image sentinel has now been replaced by 576 visual
            # embeddings.
            # ----------------------------------------------------------

            input_ids = None
            pixel_values = None

            multimodal_length = inputs_embeds.shape[1]

            # The multimodal attention mask is already returned by
            # prepare_multimodal_inputs(). Do not replace it with the
            # original text-only mask.
            if attention_mask is None:
                attention_mask = torch.ones(
                    (
                        inputs_embeds.shape[0],
                        multimodal_length,
                    ),
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # For the initial pass, explicitly define positions for the
            # complete multimodal sequence.
            # ----------------------------------------------------------

            position_ids = torch.arange(
                multimodal_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)

            cache_position = torch.arange(
                multimodal_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # ==============================================================
        # CACHED DECODING
        # ==============================================================

        elif past_key_values is not None:

            # The caller is responsible for giving us only the new
            # token. Keep this defensive slice because it also makes
            # this method compatible with generation callers that pass
            # the accumulated sequence.
            if input_ids is not None:
                input_ids = input_ids[:, -1:]

            # The image has already been consumed by the cache.
            pixel_values = None

            # ----------------------------------------------------------
            # CRITICAL:
            #
            # Let LlamaModel derive the new position from the existing
            # cache unless the caller explicitly supplied one.
            #
            # DynamicCache tracks the number of tokens already seen.
            # ----------------------------------------------------------

            if position_ids is not None:
                position_ids = position_ids[:, -1:]

            if cache_position is not None:
                cache_position = cache_position[-1:]

        # ==============================================================
        # STANDARD LLAMA FORWARD
        # ==============================================================

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Generation input preparation
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(
    self,
    input_ids: Optional[Tensor] = None,
    past_key_values=None,
    attention_mask: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    pixel_values: Optional[Tensor] = None,
    cache_position: Optional[Tensor] = None,
    **kwargs,
):
        """
        Prepare inputs for modern Transformers generation.
        """

        # ==============================================================
        # CACHED STEP
        # ==============================================================

        if past_key_values is not None:

            if input_ids is not None:
                input_ids = input_ids[:, -1:]

            return {
                "input_ids": input_ids,
                "inputs_embeds": None,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "pixel_values": None,
                "cache_position": cache_position,
                **kwargs,
            }

        # ==============================================================
        # FIRST STEP
        # ==============================================================

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "past_key_values": None,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "cache_position": cache_position,
            **kwargs,
        }