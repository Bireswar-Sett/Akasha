from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router


# MUST happen before importing routes/auth_service
load_dotenv()


app = FastAPI(
    title="AKASHA - Satellite Intelligence Assistant API",
    description="Backend API for remote sensing analysis orchestrating Qwen and HF specialist models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def read_root():
    return {"status": "AKASHA API is running"}