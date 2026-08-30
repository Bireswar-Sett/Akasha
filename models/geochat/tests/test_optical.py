from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig

from models.geochat.src.configuration_geochat import GeoChatConfig
from models.geochat.src.modeling_geochat import GeoChatLlamaForCausalLM
from models.geochat.src.processing_geochat import GeoChatProcessor


MODEL_ID = "MBZUAI/geochat-7B"

# ------------------------------------------------------------
# PUT YOUR LOCAL SENTINEL-2 RGB IMAGE HERE
# ------------------------------------------------------------

IMAGE_PATH = Path(
    "/run/media/mirage/787657C376578134/Projects/GeoChatOLD/demo_images/MicrosoftTeams-image.png"
)


def main() -> None:
    print("=" * 80)
    print("GEOCHAT OPTICAL RGB IMAGE TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------

    config = GeoChatConfig.from_pretrained(
        MODEL_ID
    )

    # ------------------------------------------------------------
    # 4-bit quantization
    # ------------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ------------------------------------------------------------
    # Load GeoChat
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------

    processor = GeoChatProcessor(
        model_name_or_path=MODEL_ID,
        vision_model_name=config.mm_vision_tower,
    )

    query = (
        "Describe this optical satellite image. "
        "Identify the main land cover, buildings, roads, "
        "water bodies, vegetation, and other notable features."
    )

    inputs = processor(
        image=image,
        query=query,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    # ------------------------------------------------------------
    # Verify image sentinel
    # ------------------------------------------------------------

    print("\nToken IDs:")
    print(input_ids.tolist())

    if not (input_ids == -200).any():
        raise AssertionError(
            "IMAGE_TOKEN_INDEX (-200) was not found."
        )

    print(
        "\n✓ Image sentinel detected."
    )

    # ------------------------------------------------------------
    # Device placement
    # ------------------------------------------------------------

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

    print("\nModel inputs:")
    print(
        f"  input_ids      = "
        f"{tuple(input_ids.shape)}"
    )
    print(
        f"  attention_mask = "
        f"{tuple(attention_mask.shape)}"
    )
    print(
        f"  pixel_values   = "
        f"{tuple(pixel_values.shape)}"
    )

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    print(
        "\nRunning optical RGB multimodal forward..."
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    print("\nForward output:")
    print(
        f"  logits shape = "
        f"{tuple(outputs.logits.shape)}"
    )

    print(
        f"  logits dtype = "
        f"{outputs.logits.dtype}"
    )

    # ------------------------------------------------------------
    # Validate multimodal sequence length
    # ------------------------------------------------------------

    num_visual_tokens = 576

    image_token_count = (
        (input_ids == -200)
        .sum()
        .item()
    )

    if image_token_count != 1:
        raise AssertionError(
            f"Expected exactly one image token, "
            f"found {image_token_count}."
        )

    expected_length = (
        input_ids.shape[1]
        - 1
        + num_visual_tokens
    )

    actual_length = (
        outputs.logits.shape[1]
    )

    print("\nSequence lengths:")
    print(
        f"  text tokens      = "
        f"{input_ids.shape[1]}"
    )
    print(
        f"  visual tokens    = "
        f"{num_visual_tokens}"
    )
    print(
        f"  expected         = "
        f"{expected_length}"
    )
    print(
        f"  actual           = "
        f"{actual_length}"
    )

    assert (
        actual_length
        == expected_length
    )

    print("\n" + "=" * 80)
    print(
        "✓ OPTICAL RGB MULTIMODAL TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()