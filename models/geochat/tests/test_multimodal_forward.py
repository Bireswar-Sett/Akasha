from __future__ import annotations

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


def main() -> None:
    print("=" * 80)
    print("GEOCHAT MULTIMODAL FORWARD TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    # ------------------------------------------------------------------
    # Make CUDA report errors at the operation that actually caused them.
    # Useful while debugging the multimodal path.
    # ------------------------------------------------------------------

    torch.backends.cudnn.enabled = False

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

    config = GeoChatConfig.from_pretrained(
        MODEL_ID,
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
    # Load GeoChat
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

    # ------------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------------

    processor = GeoChatProcessor(
        model_name_or_path=MODEL_ID,
        vision_model_name=config.mm_vision_tower,
    )

    # ------------------------------------------------------------------
    # Dummy image
    # ------------------------------------------------------------------

    image = Image.new(
        "RGB",
        (336, 336),
        color=(120, 120, 120),
    )

    query = "Describe the image."

    # ------------------------------------------------------------------
    # Process image + text
    # ------------------------------------------------------------------

    inputs = processor(
        image=image,
        query=query,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    # ------------------------------------------------------------------
    # Verify the image sentinel exists.
    # ------------------------------------------------------------------

    print("\nToken IDs:")
    print(input_ids.tolist())

    if not (input_ids == -200).any():
        raise AssertionError(
            "IMAGE_TOKEN_INDEX (-200) was not found in input_ids."
        )

    print("\n✓ Image sentinel detected.")

    # ------------------------------------------------------------------
    # Move tensors to the model execution device.
    # ------------------------------------------------------------------

    model_device = next(
        model.parameters()
    ).device

    input_ids = input_ids.to(
        model_device
    )

    attention_mask = attention_mask.to(
        model_device
    )

    pixel_values = pixel_values.to(
        device=model_device,
        dtype=torch.float16,
    ).contiguous()

    print("\nInputs:")
    print(
        f"  input_ids      = {tuple(input_ids.shape)}"
    )
    print(
        f"  attention_mask = {tuple(attention_mask.shape)}"
    )
    print(
        f"  pixel_values   = {tuple(pixel_values.shape)}"
    )

    # ------------------------------------------------------------------
    # Run the multimodal forward pass.
    # ------------------------------------------------------------------

    print("\nRunning multimodal forward pass...")

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    # ------------------------------------------------------------------
    # Output information
    # ------------------------------------------------------------------

    print("\nForward output:")

    print(
        f"  logits shape = "
        f"{tuple(outputs.logits.shape)}"
    )

    print(
        f"  logits dtype = "
        f"{outputs.logits.dtype}"
    )

    # ------------------------------------------------------------------
    # Validate output dimensions.
    # ------------------------------------------------------------------

    assert outputs.logits.ndim == 3

    assert (
        outputs.logits.shape[0]
        == input_ids.shape[0]
    )

    assert (
        outputs.logits.shape[2]
        == config.vocab_size
    )

    # ------------------------------------------------------------------
    # Calculate expected multimodal sequence length.
    #
    # Original sequence:
    #
    #     N tokens
    #
    # One image sentinel is removed and replaced by 576 patch tokens:
    #
    #     N - 1 + 576
    # ------------------------------------------------------------------

    num_visual_tokens = 576

    num_image_tokens = (
        input_ids == -200
    ).sum(dim=1)

    if not torch.all(
        num_image_tokens == 1
    ):
        raise AssertionError(
            "Expected exactly one image sentinel per sample."
        )

    expected_sequence_length = (
        input_ids.shape[1]
        - 1
        + num_visual_tokens
    )

    actual_sequence_length = (
        outputs.logits.shape[1]
    )

    print(
        "\nSequence lengths:"
    )

    print(
        f"  Original text sequence : "
        f"{input_ids.shape[1]}"
    )

    print(
        f"  Visual tokens           : "
        f"{num_visual_tokens}"
    )

    print(
        f"  Expected multimodal     : "
        f"{expected_sequence_length}"
    )

    print(
        f"  Actual model output     : "
        f"{actual_sequence_length}"
    )

    assert (
        actual_sequence_length
        == expected_sequence_length
    ), (
        "Multimodal sequence length mismatch: "
        f"expected {expected_sequence_length}, "
        f"got {actual_sequence_length}"
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("✓ GEOCHAT MULTIMODAL FORWARD PASS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()