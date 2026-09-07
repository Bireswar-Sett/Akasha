from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from models.geochat.inference.sar import load_image_input, raster_to_rgb, sar1_to_rgb


def write_raster(path: Path, bands: np.ndarray, colorinterp=None) -> Path:
    bands = np.asarray(bands)
    if bands.ndim == 2:
        bands = bands[None, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[2],
        height=bands.shape[1],
        count=bands.shape[0],
        dtype="float32",
        transform=from_origin(0, 1, 1, 1),
    ) as destination:
        destination.write(bands.astype(np.float32))
        if colorinterp is not None:
            destination.colorinterp = colorinterp
    return path


def test_png_and_jpeg_inputs_become_rgb(tmp_path):
    image = Image.new("RGBA", (4, 3), (10, 20, 30, 255))
    png_path = tmp_path / "image.png"
    jpeg_path = tmp_path / "image.jpg"
    image.save(png_path)
    image.convert("RGB").save(jpeg_path)

    assert load_image_input(png_path).mode == "RGB"
    assert load_image_input(jpeg_path).mode == "RGB"


def test_single_rgb_tiff_is_loaded_with_rasterio(tmp_path):
    path = write_raster(tmp_path / "rgb.tif", np.arange(48).reshape(3, 4, 4))

    image = raster_to_rgb(path)

    assert image.mode == "RGB"
    assert image.size == (4, 4)


def test_geotiff_with_explicit_rgb_bands_is_supported(tmp_path):
    path = write_raster(tmp_path / "rgb_geotiff.tif", np.ones((3, 2, 5)))

    image = load_image_input(path)

    assert image.mode == "RGB"
    assert image.size == (5, 2)


def test_multiband_geotiff_uses_explicit_rgb_interpretation(tmp_path):
    path = write_raster(
        tmp_path / "multiband_rgb.tif",
        np.ones((4, 2, 2)),
        (ColorInterp.undefined, ColorInterp.red, ColorInterp.green, ColorInterp.blue),
    )

    image = raster_to_rgb(path)

    assert image.mode == "RGB"


def test_invalid_tiff_has_clean_processing_error(tmp_path):
    path = tmp_path / "invalid.tif"
    path.write_bytes(b"not-a-raster")

    with pytest.raises(ValueError, match="Unable to decode"):
        raster_to_rgb(path)


def test_unsupported_file_type_is_rejected(tmp_path):
    path = tmp_path / "image.txt"
    path.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image file type"):
        load_image_input(path)


def test_unsupported_multispectral_configuration_is_rejected(tmp_path):
    path = write_raster(tmp_path / "multispectral.tif", np.ones((4, 2, 2)))

    with pytest.raises(ValueError, match="no explicit RGB band mapping"):
        raster_to_rgb(path)


def test_sar_vv_vh_pseudo_rgb(tmp_path):
    vv = write_raster(tmp_path / "VV.tiff", np.arange(16).reshape(4, 4))
    vh = write_raster(tmp_path / "VH.tiff", np.arange(16, 32).reshape(4, 4))

    image = sar1_to_rgb(vv, vh)

    assert image.mode == "RGB"
    assert image.size == (4, 4)
    assert np.asarray(image).shape == (4, 4, 3)


def test_sar_dimensions_must_match(tmp_path):
    vv = write_raster(tmp_path / "VV.tif", np.ones((4, 4)))
    vh = write_raster(tmp_path / "VH.tif", np.ones((3, 4)))

    with pytest.raises(ValueError, match="dimensions do not match"):
        sar1_to_rgb(vv, vh)


def test_missing_vh_is_rejected(tmp_path):
    vv = write_raster(tmp_path / "VV.tif", np.ones((4, 4)))

    with pytest.raises((FileNotFoundError, ValueError)):
        sar1_to_rgb(vv, tmp_path / "missing-vh.tif")
