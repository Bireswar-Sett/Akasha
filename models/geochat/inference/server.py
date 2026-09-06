from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
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


def _run_image(image, prompt: str, max_new_tokens: int = 128) -> dict:
    if image is None:
        raise ValueError("Image is required.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    response = engine.generate(
        image=image,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
    )
    return {
        "status": "completed",
        "tool": "GeoChat",
        "response": response,
        "answer": response,
        "evidence": {
            "observations": [response] if response else [],
            "interpretations": [],
            "spatial_outputs": [],
            "confidence": None,
        },
        "model": engine.model_id,
        "modality": "rgb",
    }


def geochat_specialist(image, prompt: str, max_new_tokens: int = 128) -> dict:
    """Gradio-callable one-image specialist operation for Qwen."""
    return _run_image(image, prompt, max_new_tokens)


geochat_demo = gr.Interface(
    fn=geochat_specialist,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Specialist Prompt"),
        gr.Number(value=128, minimum=1, maximum=512, precision=0, visible=False),
    ],
    outputs=gr.JSON(label="GeoChat Evidence"),
    api_name="geochat",
    title="GeoChat Specialist",
)


@app.post("/http/geochat")
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

        return _run_image(pil_image, prompt, max_new_tokens)

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


@app.post("/http/geochat/sar")
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

        result = _run_image(rgb_image, prompt, max_new_tokens)
        result["modality"] = "sar"
        return result

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


app = gr.mount_gradio_app(app, geochat_demo, path="/")