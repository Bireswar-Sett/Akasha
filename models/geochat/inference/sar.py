from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from PIL import Image


RASTER_SUFFIXES = {".tif", ".tiff", ".geotiff"}
IMAGE_SUFFIXES = RASTER_SUFFIXES | {".png", ".jpg", ".jpeg", ".webp"}


def _read_band(
    path: str | Path,
) -> np.ndarray:

    path = Path(path)

    if not path.exists():
        raise ValueError("Raster file was not found.")

    with rasterio.open(path) as src:

        if src.count < 1:
            raise ValueError("Raster contains no bands.")

        return src.read(
            1
        ).astype(
            np.float32
        )


def _normalize(
    band: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:

    band = np.nan_to_num(
        band,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros_like(band, dtype=np.float32)

    values = band[finite]
    minimum = float(np.percentile(values, low_percentile))
    maximum = float(np.percentile(values, high_percentile))

    if maximum <= minimum:
        return np.zeros_like(
            band,
            dtype=np.float32,
        )

    return (
        np.clip((band - minimum) / (maximum - minimum), 0.0, 1.0)
    ).astype(
        np.float32
    )


def _rgb_band_indexes(src: rasterio.io.DatasetReader) -> tuple[int, int, int]:
    """Return explicit RGB band indexes or reject ambiguous multispectral data."""
    if src.count == 1:
        return 1, 1, 1

    colorinterp = list(src.colorinterp)
    named = {
        "red": next((index + 1 for index, value in enumerate(colorinterp) if getattr(value, "name", "").lower() == "red"), None),
        "green": next((index + 1 for index, value in enumerate(colorinterp) if getattr(value, "name", "").lower() == "green"), None),
        "blue": next((index + 1 for index, value in enumerate(colorinterp) if getattr(value, "name", "").lower() == "blue"), None),
    }
    if all(value is not None for value in named.values()):
        return named["red"], named["green"], named["blue"]

    if src.count == 3:
        return 1, 2, 3

    raise ValueError(
        "The GeoTIFF has multiple bands but no explicit RGB band mapping."
    )


def raster_to_rgb(path: str | Path) -> Image.Image:
    """Convert a supported scientific raster into a PIL RGB visualization."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise ValueError("Raster file was not found.")

    try:
        with rasterio.open(path) as src:
            red_index, green_index, blue_index = _rgb_band_indexes(src)
            bands = [src.read(index).astype(np.float32) for index in (red_index, green_index, blue_index)]
    except rasterio.errors.RasterioError as exc:
        raise ValueError("Unable to decode the supplied GeoTIFF.") from exc

    if not (bands[0].shape == bands[1].shape == bands[2].shape):
        raise ValueError("Raster RGB bands do not have matching dimensions.")

    rgb = np.stack([_normalize(band) for band in bands], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def load_image_input(image: Image.Image | str | Path) -> Image.Image:
    """Normalize a PIL image or uploaded path without sending TIFF through Pillow."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    source = str(image)
    if source.startswith("https://"):
        parsed = urlparse(source)
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError("Unsupported image file type.")
        with TemporaryDirectory(prefix="geochat_download_") as directory:
            path = Path(directory) / f"input{suffix}"
            request = Request(source, headers={"User-Agent": "GeoChat"})
            with urlopen(request, timeout=180) as response, path.open("wb") as destination:
                destination.write(response.read())
            return load_image_input(path)

    path = Path(source)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Unsupported image file type.")
    if path.suffix.lower() in RASTER_SUFFIXES:
        return raster_to_rgb(path)

    try:
        with Image.open(path) as loaded:
            return loaded.convert("RGB")
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("Unable to decode the supplied image.") from exc


def sar1_to_rgb(
    vv_path: str | Path,
    vh_path: str | Path,
) -> Image.Image:
    """
    Convert a Sentinel-1 VV/VH pair into the
    3-channel representation used by the current
    GeoChat pipeline.

        R = VV
        G = VH
        B = (VV + VH) / 2
    """

    vv = _read_band(
        vv_path
    )

    vh = _read_band(
        vh_path
    )

    if vv.shape != vh.shape:
        raise ValueError(
            "VV and VH dimensions do not match: "
            f"{vv.shape} vs {vh.shape}"
        )

    vv_norm = _normalize(vv)
    vh_norm = _normalize(vh)

    third = (
        vv_norm + vh_norm
    ) / 2.0

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