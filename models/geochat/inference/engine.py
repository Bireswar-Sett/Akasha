from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig

from models.geochat.src.configuration_geochat import GeoChatConfig
from models.geochat.src.modeling_geochat import (
    GeoChatLlamaForCausalLM,
)
from models.geochat.src.processing_geochat import (
    GeoChatProcessor,
)


MODEL_ID = "MBZUAI/geochat-7B"


class GeoChatEngine:
    """
    Persistent GeoChat inference engine.

    The model is loaded once and remains resident on the GPU.

    Public interface:

        generate(image, prompt)
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
    ) -> None:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GeoChat requires a CUDA-capable GPU."
            )

        self.model_id = model_id

        print(
            f"[GeoChat] Loading {model_id}..."
        )

        config = GeoChatConfig.from_pretrained(
            model_id,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = (
            GeoChatLlamaForCausalLM.from_pretrained(
                model_id,
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        )

        self.model.eval()

        self.processor = GeoChatProcessor(
            model_name_or_path=model_id,
            vision_model_name=config.mm_vision_tower,
        )

        self.device = next(
            self.model.parameters()
        ).device

        self.eos_token_id = (
            self.processor.tokenizer.eos_token_id
        )

        print(
            "[GeoChat] Ready."
        )

    @staticmethod
    def _get_cache_length(
        past_key_values,
    ) -> int:
        """
        Support both modern and legacy cache objects.
        """

        if hasattr(
            past_key_values,
            "get_seq_length",
        ):
            return past_key_values.get_seq_length()

        return past_key_values[0][0].shape[-2]

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "prompt must be a string"
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "prompt must not be empty"
            )

        if not 1 <= max_new_tokens <= 512:
            raise ValueError(
                "max_new_tokens must be between 1 and 512"
            )

        image = image.convert("RGB")

        inputs = self.processor(
            image=image,
            query=prompt,
        )

        input_ids = inputs[
            "input_ids"
        ].to(self.device)

        attention_mask = inputs[
            "attention_mask"
        ].to(self.device)

        pixel_values = inputs[
            "pixel_values"
        ].to(
            device=self.device,
            dtype=torch.float16,
        ).contiguous()

        # ----------------------------------------------------------
        # PREFILL
        # ----------------------------------------------------------

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=True,
        )

        past_key_values = (
            outputs.past_key_values
        )

        next_logits = (
            outputs.logits[:, -1, :]
        )

        # ----------------------------------------------------------
        # Actual multimodal prefix length.
        # ----------------------------------------------------------

        multimodal_length = (
            outputs.logits.shape[1]
        )

        # ----------------------------------------------------------
        # GREEDY AUTOREGRESSIVE DECODING
        # ----------------------------------------------------------

        generated_tokens: list[
            torch.Tensor
        ] = []

        for step in range(
            max_new_tokens
        ):

            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )

            token_id = next_token.item()

            if (
                self.eos_token_id is not None
                and token_id == self.eos_token_id
            ):
                break

            generated_tokens.append(
                next_token
            )

            cache_length = (
                self._get_cache_length(
                    past_key_values
                )
            )

            cached_attention_mask = torch.ones(
                (
                    next_token.shape[0],
                    cache_length + 1,
                ),
                dtype=attention_mask.dtype,
                device=self.device,
            )

            outputs = self.model(
                input_ids=next_token,
                attention_mask=cached_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = (
                outputs.past_key_values
            )

            next_logits = (
                outputs.logits[:, -1, :]
            )

        if not generated_tokens:
            return ""

        generated = torch.cat(
            generated_tokens,
            dim=1,
        )

        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()