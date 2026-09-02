from __future__ import annotations

import os


SERVICE_URLS = {
    "geochat": os.getenv(
        "GEOCHAT_URL",
        "http://localhost:7860/geochat",
    ),

    "teochat": os.getenv(
        "TEOCHAT_URL",
        "http://localhost:7861/teochat",
    ),

    "m2cd": os.getenv(
        "M2CD_URL",
        "http://localhost:7862/m2cd",
    ),

    "make_sar_rgb": os.getenv(
        "SAR_RGB_URL",
        "http://localhost:7863/make_sar_rgb",
    ),

    "crop_changed_region": os.getenv(
        "CROP_URL",
        "http://localhost:7864/crop_changed_region",
    ),
}