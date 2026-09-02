from __future__ import annotations

import os


# ----------------------------------------------------------------------
# Hugging Face / specialist configuration
# ----------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

GEOCHAT_SPACE = os.getenv(
    "GEOCHAT_SPACE",
    "Bireswar26/GeoChat",
)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_config() -> None:
    """
    Validate configuration required for specialist execution.
    """

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set."
        )