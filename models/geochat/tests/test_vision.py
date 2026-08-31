from __future__ import annotations

import torch
from PIL import Image

from models.geochat.src.configuration_geochat import GeoChatConfig
from models.geochat.src.modeling_geochat import GeoChatLlamaForCausalLM


MODEL_ID = "MBZUAI/geochat-7B"


def main():
    print("=" * 80)
    print("GEOCHAT VISION PIPELINE TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    config = GeoChatConfig.from_pretrained(MODEL_ID)

    print("\nLoading GeoChat...")
    model = GeoChatLlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.float16,
        },
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    vision_tower = model.get_vision_tower()

    # ------------------------------------------------------------
    # Create a dummy RGB image.
    #
    # We use a deterministic image here so we can repeat the test.
    # ------------------------------------------------------------

    image = Image.new(
        "RGB",
        (336, 336),
        color=(120, 120, 120),
    )

    # ------------------------------------------------------------
    # Use the GeoChat image processor.
    # ------------------------------------------------------------

    pixel_values = vision_tower.image_processor(
        images=image,
        return_tensors="pt",
    )["pixel_values"]

    # ------------------------------------------------------------
    # Put image on the vision tower's device.
    # ------------------------------------------------------------

    pixel_values = pixel_values.to(
        device=vision_tower.device,
        dtype=vision_tower.dtype,
    )

    print("\nPixel values:")
    print(f"  shape = {tuple(pixel_values.shape)}")
    print(f"  dtype = {pixel_values.dtype}")
    print(f"  device = {pixel_values.device}")

    # ------------------------------------------------------------
    # CLIP feature extraction
    # ------------------------------------------------------------

    with torch.inference_mode():

        vision_features = vision_tower(
            pixel_values
        )

        print("\nCLIP output:")
        print(
            f"  shape = {tuple(vision_features.shape)}"
        )

        # --------------------------------------------------------
        # Project into LLaMA embedding space.
        # --------------------------------------------------------

        projected_features = model.get_mm_projector()(
            vision_features
        )

    print("\nProjected visual tokens:")
    print(
        f"  shape = {tuple(projected_features.shape)}"
    )
    print(
        f"  dtype = {projected_features.dtype}"
    )
    print(
        f"  device = {projected_features.device}"
    )

    # ------------------------------------------------------------
    # Validate expected dimensions.
    # ------------------------------------------------------------

    batch_size = projected_features.shape[0]
    num_patches = projected_features.shape[1]
    hidden_size = projected_features.shape[2]

    assert batch_size == 1
    assert num_patches == 576
    assert hidden_size == config.hidden_size

    print("\n✓ Vision pipeline passed.")

    print(
        f"  {num_patches} visual tokens"
        f" × {hidden_size} dimensions"
    )


if __name__ == "__main__":
    main()