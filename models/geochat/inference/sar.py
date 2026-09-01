from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


def _read_band(
    path: str | Path,
) -> np.ndarray:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"SAR file not found: {path}"
        )

    with rasterio.open(path) as src:

        if src.count < 1:
            raise ValueError(
                f"No raster bands found in {path}"
            )

        return src.read(
            1
        ).astype(
            np.float32
        )


def _normalize(
    band: np.ndarray,
) -> np.ndarray:

    band = np.nan_to_num(
        band,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(
        band.min()
    )

    maximum = float(
        band.max()
    )

    if maximum <= minimum:
        return np.zeros_like(
            band,
            dtype=np.float32,
        )

    return (
        (band - minimum)
        / (maximum - minimum)
    ).astype(
        np.float32
    )


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

    vv = _normalize(vv)

    vh = _normalize(vh)

    third = (
        vv + vh
    ) / 2.0

    rgb = np.stack(
        [
            vv,
            vh,
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