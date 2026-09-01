from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from models.geochat.inference.engine import (
    GeoChatEngine,
)
from models.geochat.inference.sar import (
    sar1_to_rgb,
)


app = FastAPI(
    title="GeoChat Inference Service",
    version="1.0.0",
)


# ----------------------------------------------------------------------
# Load model ONCE.
# ----------------------------------------------------------------------

engine = GeoChatEngine()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": engine.model_id,
    }


@app.post("/geochat")
async def geochat(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(128),
):
    """
    Analyze one already-prepared RGB image.
    """

    if not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt must not be empty.",
        )

    suffix = (
        Path(
            image.filename or ""
        ).suffix
        or ".png"
    )

    temp_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:

            temp_path = Path(
                tmp.name
            )

            contents = await image.read()

            tmp.write(
                contents
            )

        from PIL import Image

        pil_image = Image.open(
            temp_path
        ).convert("RGB")

        response = engine.generate(
            image=pil_image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        return {
            "response": response,
            "model": engine.model_id,
            "modality": "rgb",
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:

        if temp_path is not None:
            temp_path.unlink(
                missing_ok=True
            )


@app.post("/geochat/sar")
async def geochat_sar(
    vv: UploadFile = File(...),
    vh: UploadFile = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(128),
):
    """
    Analyze one Sentinel-1 VV/VH pair.

    The service constructs the pseudo-RGB image internally.
    """

    if not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt must not be empty.",
        )

    vv_path: Path | None = None
    vh_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".tif",
            delete=False,
        ) as vv_file:

            vv_path = Path(
                vv_file.name
            )

            vv_file.write(
                await vv.read()
            )

        with tempfile.NamedTemporaryFile(
            suffix=".tif",
            delete=False,
        ) as vh_file:

            vh_path = Path(
                vh_file.name
            )

            vh_file.write(
                await vh.read()
            )

        rgb_image = sar1_to_rgb(
            vv_path,
            vh_path,
        )

        response = engine.generate(
            image=rgb_image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        return {
            "response": response,
            "model": engine.model_id,
            "modality": "sar",
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:

        if vv_path is not None:
            vv_path.unlink(
                missing_ok=True
            )

        if vh_path is not None:
            vh_path.unlink(
                missing_ok=True
            )