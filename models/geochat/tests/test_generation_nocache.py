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

MAX_NEW_TOKENS = 10


def main() -> None:
    print("=" * 80)
    print("GEOCHAT NO-CACHE GENERATION TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

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
    # Load image
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
    # Verify image sentinel
    # ------------------------------------------------------------------

    image_token_mask = (
        input_ids == -200
    )

    if not image_token_mask.any():
        raise RuntimeError(
            "Image sentinel (-200) was not found."
        )

    if image_token_mask.sum().item() != 1:
        raise RuntimeError(
            "Expected exactly one image sentinel."
        )

    print("\n✓ Image sentinel detected.")

    print(
        f"\nInitial token count: "
        f"{input_ids.shape[1]}"
    )

    # ------------------------------------------------------------------
    # Device
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
    # ENCODE IMAGE + TEXT ONCE
    #
    # This gives us the actual multimodal embedding sequence.
    #
    # We deliberately keep these embeddings and reuse them so the
    # image is NOT re-encoded every generation step.
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BUILDING MULTIMODAL PREFIX")
    print("=" * 80)

    with torch.inference_mode():

        multimodal_embeds = (
            model.model.prepare_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=None,
            )[0]
        )

    multimodal_length = (
        multimodal_embeds.shape[1]
    )

    print(
        f"\n✓ Multimodal prefix built."
    )

    print(
        f"  shape = "
        f"{tuple(multimodal_embeds.shape)}"
    )

    print(
        f"  sequence length = "
        f"{multimodal_length}"
    )

    print(
        "  image encoded exactly once."
    )

    # ------------------------------------------------------------------
    # Get input embedding layer
    # ------------------------------------------------------------------

    embedding_layer = (
        model.get_input_embeddings()
    )

    # ------------------------------------------------------------------
    # Greedy generation WITHOUT KV CACHE
    # ------------------------------------------------------------------

    generated_tokens: list[torch.Tensor] = []

    current_embeds = multimodal_embeds

    eos_token_id = (
        processor.tokenizer.eos_token_id
    )

    print("\n" + "=" * 80)
    print("GENERATING WITHOUT KV CACHE")
    print("=" * 80)

    for step in range(MAX_NEW_TOKENS):

        current_length = (
            current_embeds.shape[1]
        )

        current_attention_mask = torch.ones(
            (
                current_embeds.shape[0],
                current_length,
            ),
            dtype=attention_mask.dtype,
            device=device,
        )

        position_ids = torch.arange(
            current_length,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

        # --------------------------------------------------------------
        # Full-sequence forward.
        #
        # NO cache.
        # NO image tensor.
        # NO input_ids.
        #
        # We directly supply the multimodal embeddings.
        # --------------------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        # --------------------------------------------------------------
        # Greedy next token.
        # --------------------------------------------------------------

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        token_id = next_token.item()

        generated_tokens.append(
            next_token
        )

        # Decode exactly this token for diagnostics.
        token_text = processor.decode(
            next_token[0],
            skip_special_tokens=False,
        )

        print(
            f"step={step + 1:02d}  "
            f"id={token_id:<6d}  "
            f"text={token_text!r}"
        )

        # --------------------------------------------------------------
        # EOS
        # --------------------------------------------------------------

        if eos_token_id is not None:
            if token_id == eos_token_id:
                print(
                    "\n✓ EOS token reached."
                )
                break

        # --------------------------------------------------------------
        # Convert generated token into an embedding and append it.
        #
        # This is the critical part of the no-cache experiment.
        # On the next iteration the model receives:
        #
        #   visual + original text + token_1
        #
        # Then:
        #
        #   visual + original text + token_1 + token_2
        #
        # etc.
        # --------------------------------------------------------------

        with torch.inference_mode():

            next_token_embed = (
                embedding_layer(
                    next_token
                )
            )

        current_embeds = torch.cat(
            [
                current_embeds,
                next_token_embed,
            ],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Decode complete answer
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
    print("✓ NO-CACHE GENERATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()