
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from teochat_engine import TEOChatEngine


API_KEY = os.environ.get("TEOCHAT_API_KEY")

if not API_KEY:
    raise RuntimeError("TEOCHAT_API_KEY is not set.")


app = FastAPI(
    title="TEOChat API",
    version="1.0.0",
    description="API service for TEOChat Earth Observation inference.",
)


engine = None


@app.on_event("startup")
def startup_event():
    global engine

    engine = TEOChatEngine(
        model_path=os.environ.get(
            "TEOCHAT_MODEL_PATH",
            "jirvin16/TEOChat",
        ),
        model_base=None,
        load_8bit=True,
        device="cuda",
    )


def verify_api_key(authorization: Optional[str]):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header must use Bearer authentication.",
        )

    token = authorization[7:]

    if token != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "TEOChat",
        "gpu": "cuda",
    }


@app.post("/analyze")
async def analyze(
    question: str = Form(...),
    image: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    verify_api_key(authorization)

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="TEOChat engine is not ready.",
        )

    suffix = Path(image.filename or "").suffix or ".png"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)
        contents = await image.read()
        tmp.write(contents)

    try:
        response = engine.analyze(
            image_paths=[str(temp_path)],
            instruction=question,
        )

        return JSONResponse(
            {
                "model": "TEOChat",
                "answer": response,
            }
        )

    finally:
        temp_path.unlink(missing_ok=True)
