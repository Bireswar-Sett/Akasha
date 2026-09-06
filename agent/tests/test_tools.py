import numpy as np

from qwen.controller.image_processing import (
    extract_change_regions,
    pseudo_rgb_from_vv_vh,
)


def test_pseudo_rgb_shape():
    vv = np.arange(16, dtype=np.float32).reshape(4, 4)
    vh = vv[::-1].copy()

    rgb = pseudo_rgb_from_vv_vh(vv, vh)

    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8


def test_region_extraction():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 2:5] = 0.95

    regions = extract_change_regions(
        mask,
        threshold=0.5,
        max_regions=4,
        min_area_pixels=1,
    )

    assert len(regions) == 1
    assert regions[0]["bbox"]["x_min"] == 2
    assert regions[0]["bbox"]["y_min"] == 2
