import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

app = FastAPI(
    title="SatQuery AI",
    description="Conversational Satellite Intelligence API for Smart India Hackathon 2026 / ISRO-SAC (PS ID 26167)",
    version="1.0.0",
)

# Enable CORS for local Vite dev server and external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for visual overlay serving
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Mount data/samples directory for serving sample satellite imagery
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)
app.mount("/data/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

# Include main API router
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "SatQuery AI",
        "description": "Conversational Satellite Intelligence API",
        "hackathon": "Smart India Hackathon 2026 (SIH 2026)",
        "organization": "ISRO-SAC",
        "ps_id": "26167",
        "status": "ready",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /api/query",
            "samples": "GET /api/samples",
            "docs": "/docs",
            "health": "/api/health"
        }
    }


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "specialists": ["VQA", "GROUNDING", "CHANGE_DETECTION", "FUSION"],
        "benchmarks_loaded": ["BigEarthNet.txt", "VRSBench", "RSVQA", "CDVQA"],
        "sensor_support": ["Sentinel-2 MSI", "Sentinel-1 SAR", "Cartosat-2S", "RISAT-1"]
    }
