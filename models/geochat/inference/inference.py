from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ..src.configuration_geochat import GeoChatConfig
from ..src.modeling_geochat import GeoChatLlamaForCausalLM
from ..src.processing_geochat import GeoChatProcessor


class GeoChatInference:
    """
    High-level inference wrapper for GeoChat.

    Designed for:
        - local development
        - 8 GB VRAM systems
        - later HF deployment
        - later Qwen-agent integration
    """

    def __init__(
        self,
        model_id: str = "MBZUAI/geochat-7B",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA device is available."
            )

        self.processor = GeoChatProcessor(
            model_name_or_path=model_id,
            vision_model_name=(
                "openai/clip-vit-large-patch14-336"
            ),
        )

        quantization_config = None

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = GeoChatLlamaForCausalLM.from_pretrained(
            model_id,
            config=GeoChatConfig.from_pretrained(
                model_id,
            ),
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        self.model.eval()

    @torch.inference_mode()
    def analyze(
        self,
        image: Union[Image.Image, str, Path],
        query: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> dict:
        """
        Run GeoChat on an image and query.

        Returns a structured dictionary suitable for
        consumption by the Qwen agent.
        """

        inputs = self.processor(
            image=image,
            query=query,
        )

        # Move only tensors to the model's execution device.
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        generation_kwargs = {
            "max_new_tokens": (
                max_new_tokens
                if max_new_tokens is not None
                else self.max_new_tokens
            ),
            "do_sample": temperature > 0,
            "temperature": temperature,
            "use_cache": True,
        }

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        # Remove the prompt portion when decoding.
        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[
            :,
            input_length:,
        ]

        answer = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

        return {
            "answer": answer,
            "observations": [],
            "grounding": [],
            "evidence": [],
            "confidence": None,
        }