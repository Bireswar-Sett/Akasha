from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig

from models.geochat.src.configuration_geochat import GeoChatConfig
from models.geochat.src.modeling_geochat import GeoChatLlamaForCausalLM
from models.geochat.src.processing_geochat import GeoChatProcessor


MODEL_ID = "MBZUAI/geochat-7B"

IMAGE_PATH = Path(
    "/run/media/mirage/787657C376578134/Projects/GeoChatOLD/"
    "demo_images/MicrosoftTeams-image.png"
)

MAX_NEW_TOKENS = 20


def get_eos_token_id(
    processor: GeoChatProcessor,
    config: GeoChatConfig,
) -> int | None:
    if processor.tokenizer.eos_token_id is not None:
        return processor.tokenizer.eos_token_id

    return config.eos_token_id


def get_cache_length(
    past_key_values,
) -> int:
    """
    Get the number of cached sequence positions.

    Supports the modern Cache API and falls back to the
    legacy tuple-of-layers representation.
    """

    if hasattr(
        past_key_values,
        "get_seq_length",
    ):
        return past_key_values.get_seq_length()

    return past_key_values[0][0].shape[-2]


def main() -> None:
    print("=" * 80)
    print("GEOCHAT KV-CACHE GENERATION TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    config = GeoChatConfig.from_pretrained(
        MODEL_ID
    )

    # ------------------------------------------------------------------
    # 4-bit quantization
    # ------------------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print("\nLoading GeoChat...")

    model = GeoChatLlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("✓ Model loaded.")

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print("\nImage:")
    print(
        f"  path = {IMAGE_PATH}"
    )
    print(
        f"  size = {image.size}"
    )
    print(
        f"  mode = {image.mode}"
    )

    # ------------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------------

    processor = GeoChatProcessor(
        model_name_or_path=MODEL_ID,
        vision_model_name=config.mm_vision_tower,
    )

    query = (
        "Describe this satellite image in detail. "
        "Identify the main land cover, structures, roads, "
        "water bodies, vegetation, and any other notable "
        "features you can observe."
    )

    inputs = processor(
        image=image,
        query=query,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    # ------------------------------------------------------------------
    # Validate image token
    # ------------------------------------------------------------------

    image_token_mask = (
        input_ids == -200
    )

    if image_token_mask.sum().item() != 1:
        raise RuntimeError(
            "Expected exactly one image sentinel."
        )

    print(
        "\n✓ Image sentinel detected."
    )

    print(
        f"\nInitial text/sentinel sequence: "
        f"{input_ids.shape[1]}"
    )

    # ------------------------------------------------------------------
    # Device placement
    # ------------------------------------------------------------------

    device = next(
        model.parameters()
    ).device

    input_ids = input_ids.to(
        device
    )

    attention_mask = attention_mask.to(
        device
    )

    pixel_values = pixel_values.to(
        device=device,
        dtype=torch.float16,
    ).contiguous()

    # ------------------------------------------------------------------
    # PREFILL
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PREFILL")
    print("=" * 80)

    with torch.inference_mode():

        outputs = model(
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

    multimodal_length = (
        outputs.logits.shape[1]
    )

    cache_length = get_cache_length(
        past_key_values
    )

    print(
        f"\n✓ Prefill complete."
    )

    print(
        f"  text/sentinel length = "
        f"{input_ids.shape[1]}"
    )

    print(
        f"  multimodal length    = "
        f"{multimodal_length}"
    )

    print(
        f"  cache length         = "
        f"{cache_length}"
    )

    print(
        f"  logits shape         = "
        f"{tuple(outputs.logits.shape)}"
    )

    # ------------------------------------------------------------------
    # Cache must contain the complete multimodal prefix.
    # ------------------------------------------------------------------

    if cache_length != multimodal_length:
        raise RuntimeError(
            "KV cache does not contain the complete "
            "multimodal prefix: "
            f"cache={cache_length}, "
            f"multimodal={multimodal_length}"
        )

    print(
        "✓ KV cache contains the complete "
        "multimodal prefix."
    )

    # ------------------------------------------------------------------
    # GREEDY CACHED DECODING
    # ------------------------------------------------------------------

    eos_token_id = get_eos_token_id(
        processor,
        config,
    )

    generated_tokens: list[torch.Tensor] = []

    print("\n" + "=" * 80)
    print("CACHED DECODING")
    print("=" * 80)

    for step in range(
        MAX_NEW_TOKENS
    ):

        # --------------------------------------------------------------
        # Greedy next token.
        # --------------------------------------------------------------

        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        token_id = next_token.item()

        token_text = processor.decode(
            next_token[0],
            skip_special_tokens=False,
        )

        generated_tokens.append(
            next_token
        )

        print(
            f"step={step + 1:02d} "
            f"id={token_id:<6d} "
            f"text={token_text!r}"
        )

        # --------------------------------------------------------------
        # EOS
        # --------------------------------------------------------------

        if (
            eos_token_id is not None
            and token_id == eos_token_id
        ):
            print(
                "\n✓ EOS token reached."
            )
            break

        # --------------------------------------------------------------
        # Current cache length.
        #
        # IMPORTANT:
        # The cache itself tells us how many positions are already
        # stored. We do not manually invent a position from the
        # original text length.
        # --------------------------------------------------------------

        cache_length = get_cache_length(
            past_key_values
        )

        # --------------------------------------------------------------
        # One new token is about to be appended.
        #
        # Therefore attention must cover:
        #
        #     cached positions + new token
        # --------------------------------------------------------------

        cached_attention_mask = torch.ones(
            (
                next_token.shape[0],
                cache_length + 1,
            ),
            dtype=attention_mask.dtype,
            device=device,
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT manually pass position_ids.
        # Do NOT manually pass cache_position.
        #
        # Let the modern LLaMA implementation derive the new
        # position from the existing cache.
        # --------------------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                input_ids=next_token,
                attention_mask=cached_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # --------------------------------------------------------------
        # Update cache + next-token logits.
        # --------------------------------------------------------------

        past_key_values = (
            outputs.past_key_values
        )

        next_logits = (
            outputs.logits[:, -1, :]
        )

        # --------------------------------------------------------------
        # Verify that exactly one position was appended.
        # --------------------------------------------------------------

        new_cache_length = get_cache_length(
            past_key_values
        )

        expected_cache_length = (
            cache_length + 1
        )

        if new_cache_length != expected_cache_length:
            raise RuntimeError(
                "KV cache grew incorrectly: "
                f"expected {expected_cache_length}, "
                f"got {new_cache_length}"
            )

    # ------------------------------------------------------------------
    # Decode final answer
    # ------------------------------------------------------------------

    if not generated_tokens:
        raise RuntimeError(
            "No tokens were generated."
        )

    generated_tensor = torch.cat(
        generated_tokens,
        dim=1,
    )

    answer = processor.batch_decode(
        generated_tensor,
        skip_special_tokens=True,
    )[0].strip()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("GEOCHAT RESPONSE")
    print("=" * 80)

    print(
        f"\n{answer}"
    )

    print("\n" + "=" * 80)
    print("✓ KV-CACHE GENERATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()