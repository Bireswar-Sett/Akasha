from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    LlamaTokenizer,
)


# ----------------------------------------------------------------------
# GeoChat / LLaVA conventions
# ----------------------------------------------------------------------

IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN = "<image>"

SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence "
    "assistant. The assistant gives helpful, detailed, and polite "
    "answers to the human's questions."
)

USER_ROLE = "USER"
ASSISTANT_ROLE = "ASSISTANT"


class GeoChatProcessor:
    """
    Processor for the modern GeoChat implementation.

    Handles:

        - LLaMA SentencePiece tokenization
        - LLaVA-v1 conversation formatting
        - GeoChat image sentinel insertion
        - CLIP image preprocessing
        - square image padding
        - decoding generated tokens
    """

    def __init__(
        self,
        model_name_or_path: str,
        vision_model_name: str = (
            "openai/clip-vit-large-patch14-336"
        ),
        image_aspect_ratio: str = "pad",
    ):
        self.model_name_or_path = model_name_or_path
        self.vision_model_name = vision_model_name
        self.image_aspect_ratio = image_aspect_ratio

        # GeoChat checkpoint contains a LLaMA SentencePiece tokenizer.
        #
        # Explicitly use LlamaTokenizer so modern Transformers does
        # not try to reinterpret tokenizer.model as TikToken data.
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
        )

        self.image_processor = CLIPImageProcessor.from_pretrained(
            vision_model_name,
        )

    # ==================================================================
    # IMAGE PROCESSING
    # ==================================================================

    @staticmethod
    def pad_to_square(
        image: Image.Image,
        background_color=(122, 116, 104),
    ) -> Image.Image:
        """
        Pad an image to a square canvas.

        GeoChat's checkpoint configuration specifies:
            image_aspect_ratio = "pad"
        """

        width, height = image.size

        if width == height:
            return image

        size = max(width, height)

        canvas = Image.new(
            image.mode,
            (size, size),
            background_color,
        )

        left = (size - width) // 2
        top = (size - height) // 2

        canvas.paste(
            image,
            (left, top),
        )

        return canvas

    def preprocess_image(
        self,
        image: Union[
            Image.Image,
            str,
            Path,
        ],
    ) -> torch.Tensor:
        """
        Convert an image into CLIP pixel values.

        Returns:
            Tensor with shape [3, H, W].
        """

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image or a path"
            )

        image = image.convert("RGB")

        if self.image_aspect_ratio == "pad":
            image = self.pad_to_square(image)

        elif self.image_aspect_ratio != "anyres":
            raise ValueError(
                f"Unsupported image aspect ratio mode: "
                f"{self.image_aspect_ratio!r}"
            )

        processed = self.image_processor(
            images=image,
            return_tensors="pt",
        )

        return processed["pixel_values"][0]

    # ==================================================================
    # PROMPT CONSTRUCTION
    # ==================================================================

    # ==================================================================
    # PROMPT CONSTRUCTION
    # ==================================================================

    def build_prompt(
        self,
        query: str,
    ) -> str:
        """
        Reproduce the original GeoChat llava_v1 prompt construction.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query must be a non-empty string"
            )

        # Original Chat.ask():
        #
        # '<image>\n' + ' ' + query
        #
        # Original llava_v1 then produces:
        #
        # SYSTEM USER: <image>\n query ASSISTANT:
        #
        return (
            f"{SYSTEM_PROMPT} "
            f"{USER_ROLE}: "
            f"{IMAGE_TOKEN}\n "
            f"{query} "
            f"{ASSISTANT_ROLE}:"
        )


    # ==================================================================
    # TOKENIZATION
    # ==================================================================

    def tokenize_with_image_token(
        self,
        text: str,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """
        Reproduce the original GeoChat tokenizer_image_token() logic.

        Crucially:
            - the first chunk retains the LLaMA BOS token
            - <image> becomes -200
            - BOS is not duplicated
        """

        if return_tensors != "pt":
            raise ValueError(
                "Only return_tensors='pt' is supported."
            )

        # Exactly like the original implementation:
        #
        # tokenizer(chunk).input_ids
        #
        # rather than forcing add_special_tokens=False.
        prompt_chunks = [
            self.tokenizer(chunk).input_ids
            for chunk in text.split(IMAGE_TOKEN)
        ]

        input_ids: list[int] = []

        offset = 0

        # Preserve the BOS token from the first chunk.
        if (
            prompt_chunks
            and prompt_chunks[0]
            and prompt_chunks[0][0]
            == self.tokenizer.bos_token_id
        ):
            offset = 1

            input_ids.append(
                prompt_chunks[0][0]
            )

        # Insert -200 between the chunks.
        for chunk_index, chunk in enumerate(
            prompt_chunks
        ):
            if chunk_index > 0:
                input_ids.append(
                    IMAGE_TOKEN_INDEX
                )

            # Remove BOS from subsequent chunks.
            input_ids.extend(
                chunk[offset:]
            )

        return torch.tensor(
            input_ids,
            dtype=torch.long,
        ).unsqueeze(0)

    def tokenize(
    self,
    text: str,
    add_image_token: bool = False,
    return_tensors: str = "pt",
) -> dict[str, torch.Tensor]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if add_image_token and IMAGE_TOKEN not in text:
            text = (
                f"{IMAGE_TOKEN}\n "
                f"{text}"
            )

        input_ids = self.tokenize_with_image_token(
            text,
            return_tensors=return_tensors,
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # ==================================================================
    # IMAGE + QUERY
    # ==================================================================

    def __call__(
        self,
        image: Union[
            Image.Image,
            str,
            Path,
        ],
        query: str,
    ) -> dict[str, torch.Tensor]:
        """
        Process one image and one user query.

        Returns:

            {
                "input_ids":      [1, sequence_length],
                "attention_mask": [1, sequence_length],
                "pixel_values":   [1, 3, 336, 336],
            }
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string"
            )

        # --------------------------------------------------------------
        # Image
        # --------------------------------------------------------------

        pixel_values = self.preprocess_image(
            image
        )

        # --------------------------------------------------------------
        # Original GeoChat/LLaVA-v1 style prompt
        # --------------------------------------------------------------

        prompt = self.build_prompt(
            query
        )

        # --------------------------------------------------------------
        # Tokenize while inserting -200 sentinel
        # --------------------------------------------------------------

        text_inputs = self.tokenize(
            prompt,
            add_image_token=False,
        )

        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values.unsqueeze(0),
        }

    # ==================================================================
    # DECODING
    # ==================================================================

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode generated token IDs.

        Any negative internal multimodal sentinel is removed first.
        """

        token_ids = token_ids[
            token_ids >= 0
        ]

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """
        Decode a batch of generated sequences.

        Negative internal multimodal sentinels are removed.
        """

        cleaned_sequences: list[torch.Tensor] = []

        for sequence in token_ids:
            sequence = sequence[
                sequence >= 0
            ]

            cleaned_sequences.append(
                sequence
            )

        return self.tokenizer.batch_decode(
            cleaned_sequences,
            skip_special_tokens=skip_special_tokens,
        )