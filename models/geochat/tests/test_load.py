from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig

from models.geochat.src.configuration_geochat import GeoChatConfig
from models.geochat.src.modeling_geochat import GeoChatLlamaForCausalLM


MODEL_ID = "MBZUAI/geochat-7B"


def print_gpu_memory(stage: str) -> None:
    """
    Print CUDA memory usage for the current GPU.
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    print(
        f"[{stage}] "
        f"allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB"
    )


def print_device_summary(model) -> None:
    """
    Print where model parameters are currently located.

    This avoids relying on hf_device_map, which is not guaranteed
    to exist for every custom model/loading path.
    """

    print("\nModel device placement:")

    devices: dict[str, int] = {}

    for name, parameter in model.named_parameters():
        device = str(parameter.device)

        if device not in devices:
            devices[device] = 0

        devices[device] += parameter.numel()

    for device, parameter_count in devices.items():
        print(
            f"  {device:<12} "
            f"{parameter_count:,} parameters"
        )


def main() -> None:
    print("=" * 80)
    print("GEOCHAT 4-BIT WEIGHT LOADING TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # CUDA check
    # ------------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this test."
        )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print_gpu_memory("before loading")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    config = GeoChatConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
    )

    print("\nConfiguration loaded.")

    print(f"  hidden_size        = {config.hidden_size}")
    print(f"  num_hidden_layers  = {config.num_hidden_layers}")
    print(f"  num_attention_heads = {config.num_attention_heads}")
    print(f"  mm_hidden_size     = {config.mm_hidden_size}")
    print(f"  projector          = {config.mm_projector_type}")
    print(f"  vision tower       = {config.mm_vision_tower}")

    # ------------------------------------------------------------------
    # 4-bit quantization
    # ------------------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("\nQuantization:")
    print("  load_in_4bit      = True")
    print("  quant type        = nf4")
    print("  double quant      = True")
    print("  compute dtype     = float16")

    # ------------------------------------------------------------------
    # Load actual checkpoint
    # ------------------------------------------------------------------

    print("\nLoading GeoChat checkpoint...")
    print("This is the first REAL weight-loading test.")

    model = GeoChatLlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("\n✓ Model loaded.")

    print_gpu_memory("after loading")

    # ------------------------------------------------------------------
    # Device placement
    # ------------------------------------------------------------------

    print_device_summary(model)

    # ------------------------------------------------------------------
    # Model dtype
    # ------------------------------------------------------------------

    print("\nModel dtype:")

    try:
        print(f"  {model.dtype}")
    except AttributeError:
        print("  unavailable")

    # ------------------------------------------------------------------
    # Vision tower
    # ------------------------------------------------------------------

    print("\nVision tower:")

    vision_tower = model.get_vision_tower()

    print(
        f"  type: {type(vision_tower).__name__}"
    )

    print(
        f"  hidden size: "
        f"{vision_tower.hidden_size}"
    )

    print(
        f"  num patches: "
        f"{vision_tower.num_patches}"
    )

    # ------------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------------

    print("\nProjector:")

    print(
        model.get_mm_projector()
    )

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    # ------------------------------------------------------------------
    # Final memory usage
    # ------------------------------------------------------------------

    print_gpu_memory("final")

    print("\n" + "=" * 80)
    print("✓ GEOCHAT WEIGHT LOADING TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()