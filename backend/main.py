from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from services.ai_service import analyze_image

app = FastAPI(
    title="AKASHA API",
    description="Earth Observation AI Assistant",
    version="1.0.0",
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def root():
    return {
        "message": "AKASHA API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "AKASHA Backend"
    }


# =========================
# ANALYZE
# =========================

@app.post("/api/analyze")
async def analyze(
    mode: str = Form(...),
    query: str = Form(...),
    t1_image: UploadFile = File(...),
    t2_image: UploadFile | None = File(None),
):

    # Read T1 image
    t1_content = await t1_image.read()

    # Read T2 image if provided
    t2_content = None

    if t2_image:
        t2_content = await t2_image.read()

    # Send T1 image to AI service
    ai_result = await analyze_image(
        image_bytes=t1_content,
        query=query,
        mode=mode,
    )

    return {
        "status": "success",
        "message": "Analysis completed",
        "data": ai_result,
    }