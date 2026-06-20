import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import media, detect, offside

app = FastAPI(title="Offside Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(media.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
app.include_router(offside.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
