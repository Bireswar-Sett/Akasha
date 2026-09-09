import os

from dotenv import load_dotenv

# IMPORTANT: Load environment variables before importing application modules.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router


app = FastAPI(
    title="AKASHA — Satellite Intelligence Assistant API",
    description=(
        "Backend API for satellite imagery analysis "
        "using AWS Cognito, S3, and Qwen."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS
#
# Origins are configurable via the ALLOWED_ORIGINS env var (comma-separated).
# The hardcoded defaults cover local development and the current Amplify URL.
# ---------------------------------------------------------------------------

_env_origins = os.getenv("ALLOWED_ORIGINS", "")
_extra_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # AWS Amplify
    "https://main.d5t6w8xnwognw.amplifyapp.com",
] + _extra_origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


app.include_router(
    api_router,
    prefix="/api",
)


@app.get("/")
def read_root():
    return {
        "status": "AKASHA API is running",
        "version": "1.0.0",
    }
