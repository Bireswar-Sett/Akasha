from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AKASHA - Satellite Intelligence Assistant API",
    description="Backend API for remote sensing analysis orchestrating Qwen and HF specialist models.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "AKASHA API is running"}
