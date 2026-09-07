from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def robust_normalize(
    array: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)

    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)

    values = arr[finite]
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))

    if high <= low:
        return np.zeros(arr.shape, dtype=np.float32)

    normalized = (arr - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def pseudo_rgb_from_vv_vh(
    vv: np.ndarray,
    vh: np.ndarray,
) -> np.ndarray:
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)

    if vv.shape != vh.shape:
        raise ValueError("VV and VH must have identical shapes.")

    combined = (vv + vh) / 2.0

    rgb = np.stack(
        [
            robust_normalize(vv),
            robust_normalize(vh),
            robust_normalize(combined),
        ],
        axis=-1,
    )

    return np.round(rgb * 255.0).astype(np.uint8)


def _safe_crop_bounds(
    bbox: dict[str, int] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if not bbox:
        return 0, 0, width, height

    x_min = max(0, min(width, int(bbox["x_min"])))
    y_min = max(0, min(height, int(bbox["y_min"])))
    x_max = max(0, min(width, int(bbox["x_max"])))
    y_max = max(0, min(height, int(bbox["y_max"])))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid crop bounding box.")

    return x_min, y_min, x_max, y_max


def sar_to_pseudo_rgb_file(
    image_path: str,
    crop: dict[str, int] | None = None,
) -> str:
    import rasterio
    from PIL import Image

    with rasterio.open(image_path) as src:
        if src.count < 2:
            raise ValueError(
                "SAR pseudo-RGB requires at least two bands; "
                "expected VV and VH."
            )

        x_min, y_min, x_max, y_max = _safe_crop_bounds(
            crop,
            src.width,
            src.height,
        )

        window = rasterio.windows.Window(
            x_min,
            y_min,
            x_max - x_min,
            y_max - y_min,
        )

        vv = src.read(1, window=window)
        vh = src.read(2, window=window)

    rgb = pseudo_rgb_from_vv_vh(vv, vh)

    directory = Path(
        tempfile.mkdtemp(prefix="akasha_sar_rgb_")
    )
    output_path = directory / "pseudo_rgb.png"

    Image.fromarray(rgb, mode="RGB").save(output_path)
    return str(output_path)


def combine_sar_channels_file(
    vv_path: str,
    vh_path: str,
) -> str:
    """Create a private two-band raster from separate VV and VH files."""
    import rasterio

    with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
        if (vv_src.width, vv_src.height) != (vh_src.width, vh_src.height):
            raise ValueError("VV and VH rasters must have identical dimensions.")
        vv = vv_src.read(1)
        vh = vh_src.read(1)
        profile = vv_src.profile.copy()
        profile.update(count=2, dtype=str(vv.dtype), driver="GTiff")

    directory = Path(tempfile.mkdtemp(prefix="akasha_sar_pair_"))
    output_path = directory / "vv_vh.tif"
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(vv, 1)
        destination.write(vh, 2)
    return str(output_path)


def extract_change_regions(
    probability_mask: np.ndarray,
    threshold: float,
    max_regions: int,
    min_area_pixels: int = 32,
) -> list[dict[str, Any]]:
    from scipy import ndimage

    mask = np.asarray(probability_mask, dtype=np.float32).squeeze()

    if mask.ndim != 2:
        raise ValueError(
            f"M2CD mask must be 2D, received {mask.shape}."
        )

    binary = np.isfinite(mask) & (mask >= threshold)
    labels, count = ndimage.label(binary)
    slices = ndimage.find_objects(labels)

    regions: list[dict[str, Any]] = []

    for label_id in range(1, count + 1):
        slc = slices[label_id - 1]
        if slc is None:
            continue

        ys, xs = slc
        component = labels[slc] == label_id
        area = int(component.sum())

        if area < min_area_pixels:
            continue

        values = mask[slc][component]
        confidence = float(values.mean()) if values.size else None

        regions.append(
            {
                "region_id": label_id,
                "bbox": {
                    "x_min": int(xs.start),
                    "y_min": int(ys.start),
                    "x_max": int(xs.stop),
                    "y_max": int(ys.stop),
                },
                "area_pixels": area,
                "confidence": confidence,
            }
        )

    regions.sort(
        key=lambda x: (
            x["confidence"] if x["confidence"] is not None else 0.0,
            x["area_pixels"],
        ),
        reverse=True,
    )

    return regions[:max_regions]


def load_numeric_mask(value: Any) -> np.ndarray:
    """
    Convert common M2CD outputs to a 2D float mask.

    Supported:
      - numpy arrays
      - torch tensors
      - Python lists
      - dicts containing mask/probability_mask
      - image paths (grayscale image)
    """
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32).squeeze()

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().astype(np.float32).squeeze()

    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32).squeeze()

    if isinstance(value, dict):
        for key in ("probability_mask", "mask", "change_map"):
            if key in value:
                return load_numeric_mask(value[key])
        raise ValueError("M2CD dict contains no recognized mask field.")

    if isinstance(value, str) and Path(value).is_file():
        from PIL import Image
        image = Image.open(value).convert("L")
        return np.asarray(image, dtype=np.float32) / 255.0

    raise TypeError(
        f"Unsupported M2CD mask type: {type(value).__name__}"
    )
