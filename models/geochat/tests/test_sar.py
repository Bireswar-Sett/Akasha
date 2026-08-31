from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from transformers import BitsAndBytesConfig

from models.geochat.src.configuration_geochat import GeoChatConfig
from models.geochat.src.modeling_geochat import GeoChatLlamaForCausalLM
from models.geochat.src.processing_geochat import GeoChatProcessor


MODEL_ID = "MBZUAI/geochat-7B"

# ----------------------------------------------------------------------
# CHANGE THESE TWO PATHS
# ----------------------------------------------------------------------

VH_PATH = Path(
    "/run/media/mirage/787657C376578134/Projects/SIH/BigEarthNet.txt/S1A_IW_GRDH_1SDV_20170706T064235_29SND_18_12_VH.tif"
)

VV_PATH = Path(
    "/run/media/mirage/787657C376578134/Projects/SIH/BigEarthNet.txt/S1A_IW_GRDH_1SDV_20170706T064235_29SND_18_12_VV.tif"
)


def read_single_band_tif(path: Path) -> np.ndarray:
    """
    Read the first band of a GeoTIFF.

    Returns:
        float32 array with shape [H, W]
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with rasterio.open(path) as src:

        if src.count < 1:
            raise ValueError(
                f"No raster bands found in {path}"
            )

        data = src.read(1).astype(
            np.float32
        )

    return data


def normalize_band(
    band: np.ndarray,
) -> np.ndarray:
    """
    Min-max normalize a SAR band to [0, 1].

    NaN/Inf values are replaced before normalization.
    """

    band = np.nan_to_num(
        band,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    min_value = float(
        band.min()
    )

    max_value = float(
        band.max()
    )

    if max_value <= min_value:
        return np.zeros_like(
            band,
            dtype=np.float32,
        )

    return (
        (band - min_value)
        / (max_value - min_value)
    ).astype(np.float32)


def build_sar_rgb(
    vv: np.ndarray,
    vh: np.ndarray,
) -> Image.Image:
    """
    Construct the 3-channel SAR representation:

        R = VV
        G = VH
        B = (VV + VH) / 2

    Each channel is normalized independently to [0, 255].
    """

    if vv.shape != vh.shape:
        raise ValueError(
            "VV and VH images must have identical "
            f"dimensions. Got VV={vv.shape}, "
            f"VH={vh.shape}"
        )

    # --------------------------------------------------------------
    # Normalize VV and VH.
    # --------------------------------------------------------------

    vv_norm = normalize_band(vv)
    vh_norm = normalize_band(vh)

    # --------------------------------------------------------------
    # Third channel.
    #
    # This follows your previous SAR1 experiment.
    # --------------------------------------------------------------

    third = (
        vv_norm + vh_norm
    ) / 2.0

    # --------------------------------------------------------------
    # Stack into RGB.
    # --------------------------------------------------------------

    rgb = np.stack(
        [
            vv_norm,
            vh_norm,
            third,
        ],
        axis=-1,
    )

    rgb = (
        np.clip(
            rgb,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


def print_band_stats(
    name: str,
    band: np.ndarray,
) -> None:
    print(f"\n{name}:")
    print(f"  shape = {band.shape}")
    print(f"  dtype = {band.dtype}")
    print(f"  min   = {np.nanmin(band):.6f}")
    print(f"  max   = {np.nanmax(band):.6f}")
    print(f"  mean  = {np.nanmean(band):.6f}")


def main() -> None:

    print("=" * 80)
    print("GEOCHAT SAR1 VV/VH TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    # ------------------------------------------------------------------
    # Load SAR bands
    # ------------------------------------------------------------------

    print("\nReading SAR bands...")

    vv = read_single_band_tif(
        VV_PATH
    )

    vh = read_single_band_tif(
        VH_PATH
    )

    print_band_stats(
        "VV",
        vv,
    )

    print_band_stats(
        "VH",
        vh,
    )

    # ------------------------------------------------------------------
    # Build pseudo-RGB
    # ------------------------------------------------------------------

    print(
        "\nConstructing SAR pseudo-RGB:"
    )

    print(
        "  R = VV"
    )

    print(
        "  G = VH"
    )

    print(
        "  B = (VV + VH) / 2"
    )

    image = build_sar_rgb(
        vv,
        vh,
    )

    print(
        f"\nComposite image:"
    )

    print(
        f"  size = {image.size}"
    )

    print(
        f"  mode = {image.mode}"
    )

    composite = np.asarray(
        image
    )

    print(
        f"  shape = {composite.shape}"
    )

    print(
        f"  dtype = {composite.dtype}"
    )

    print(
        f"  min = {composite.min()}"
    )

    print(
        f"  max = {composite.max()}"
    )

    # ------------------------------------------------------------------
    # Load GeoChat
    # ------------------------------------------------------------------

    print("\nLoading GeoChat...")

    config = GeoChatConfig.from_pretrained(
        MODEL_ID
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

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

    query = (
        "Describe this SAR satellite image. "
        "What land cover, structures, water bodies, "
        "or other notable features can you identify?"
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

    print("\nToken IDs:")
    print(
        input_ids.tolist()
    )

    if not (
        input_ids == -200
    ).any():
        raise AssertionError(
            "IMAGE_TOKEN_INDEX (-200) "
            "was not found."
        )

    print(
        "\n✓ Image sentinel detected."
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    print(
        "\nRunning SAR multimodal forward..."
    )

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    print(
        "\nForward output:"
    )

    print(
        f"  logits shape = "
        f"{tuple(outputs.logits.shape)}"
    )

    print(
        f"  logits dtype = "
        f"{outputs.logits.dtype}"
    )

    # ------------------------------------------------------------------
    # Validate multimodal sequence length
    # ------------------------------------------------------------------

    num_visual_tokens = 576

    expected_length = (
        input_ids.shape[1]
        - 1
        + num_visual_tokens
    )

    actual_length = (
        outputs.logits.shape[1]
    )

    print(
        "\nSequence lengths:"
    )

    print(
        f"  text tokens       = "
        f"{input_ids.shape[1]}"
    )

    print(
        f"  visual tokens     = "
        f"{num_visual_tokens}"
    )

    print(
        f"  expected          = "
        f"{expected_length}"
    )

    print(
        f"  actual            = "
        f"{actual_length}"
    )

    assert (
        actual_length
        == expected_length
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✓ SAR1 VV/VH MULTIMODAL TEST PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()