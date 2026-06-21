"""FastAPI backend for offside detection."""
import io
import os
import base64
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

load_dotenv()

from utils import (  # noqa: E402
    CLASES,
    MODEL_PATH,
    calcular_offside,
    cargar_modelo,
    detectar_objetos,
    dibujar_resultado,
)
from hough import detectar_punto_de_fuga  # noqa: E402

app = FastAPI(title="Offside Detection API", version="1.0.0")

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


@app.on_event("startup")
async def startup_event():
    global _model
    path = os.getenv("MODEL_PATH", str(MODEL_PATH))
    _model = cargar_modelo(path)


# ── helpers ──────────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _crop_sample(img: Image.Image, det: dict) -> Optional[str]:
    try:
        w, h = img.size
        x1, y1 = max(0, int(det["x1"])), max(0, int(det["y1"]))
        x2, y2 = min(w, int(det["x2"])), min(h, int(det["y2"]))
        if x2 <= x1 or y2 <= y1:
            return None
        return _pil_to_b64(img.crop((x1, y1, x2, y2)))
    except Exception:
        return None


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/api/extract-frame")
async def extract_frame(
    video: UploadFile = File(...),
    frame: int = Form(0),
):
    data = await video.read()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="No se pudo abrir el video.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_idx = max(0, min(frame, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=400, detail="No se pudo leer el frame.")
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return {"image_b64": _pil_to_b64(img), "total_frames": total, "fps": fps}
    finally:
        os.unlink(tmp_path)


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    iou: float = Form(0.7),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado aún.")

    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida.")

    detections = detectar_objetos(_model, img, iou=iou)
    detected_img = dibujar_resultado(img, detections, None, None)

    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    vp, grupo_a, grupo_b = detectar_punto_de_fuga(img_bgr)

    class_counts: dict = {}
    for d in detections:
        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

    team1 = [d for d in detections if d["class_id"] == 5]
    team2 = [d for d in detections if d["class_id"] == 6]
    team_samples = {
        "team1": _crop_sample(img, max(team1, key=lambda x: x["conf"])) if team1 else None,
        "team2": _crop_sample(img, max(team2, key=lambda x: x["conf"])) if team2 else None,
    }

    w, h = img.size
    return {
        "detections": detections,
        "image_b64": _pil_to_b64(img),
        "detected_image_b64": _pil_to_b64(detected_img),
        "vp": {"x": float(vp[0]), "y": float(vp[1])} if vp else None,
        "vp_lines": {
            "a": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in grupo_a],
            "b": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in grupo_b],
        },
        "team_samples": team_samples,
        "class_counts": class_counts,
        "image_size": {"width": w, "height": h},
    }


class VPFromLinesRequest(BaseModel):
    lines_a: List[List[float]]
    lines_b: List[List[float]]
    image_width: float
    image_height: float


@app.post("/api/vp-from-lines")
async def vp_from_lines(req: VPFromLinesRequest):
    from hough import _ransac_vp
    segs = [(x1, y1, x2, y2) for x1, y1, x2, y2 in req.lines_a + req.lines_b]
    angulos = []
    import numpy as np
    for x1, y1, x2, y2 in req.lines_a:
        angulos.append(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
    for x1, y1, x2, y2 in req.lines_b:
        angulos.append(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
    vp, _, _ = _ransac_vp(segs, angulos, req.image_width, req.image_height)
    return {"vp": {"x": float(vp[0]), "y": float(vp[1])} if vp else None}


class Detection(BaseModel):
    class_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class VPModel(BaseModel):
    x: float
    y: float


class OffsideRequest(BaseModel):
    image_b64: str
    detections: List[Detection]
    vp: VPModel
    attacking_team_id: int
    goal_on_right: Optional[bool] = None
    reference_point: str = "medio"


@app.post("/api/calculate-offside")
async def calculate_offside_endpoint(req: OffsideRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado aún.")

    try:
        img = _b64_to_pil(req.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen base64 inválida.")

    dets = [d.model_dump() for d in req.detections]
    vp = (req.vp.x, req.vp.y)
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    resultado = calcular_offside(
        vp=vp,
        detecciones=dets,
        equipo_atacante_id=req.attacking_team_id,
        imagen_shape=(h, w),
        gol_a_derecha=req.goal_on_right,
        punto_referencia=req.reference_point,
    )

    result_img = dibujar_resultado(img, dets, vp, resultado)

    atacantes_serial = [
        {
            "detection": det,
            "offside": bool(en_offside),
            "adelantado_al_defensor": bool(ad),
            "adelantado_a_pelota": bool(ap),
        }
        for det, en_offside, ad, ap in resultado.get("atacantes_resultado", [])
    ]

    resultado_serial = {
        "penultimo_defensor": resultado.get("penultimo_defensor"),
        "pelota": resultado.get("pelota"),
        "linea_foot": list(resultado["linea_foot"]) if resultado.get("linea_foot") else None,
        "linea_pelota_foot": list(resultado["linea_pelota_foot"]) if resultado.get("linea_pelota_foot") else None,
        "atacantes_resultado": atacantes_serial,
        "hay_offside": bool(resultado.get("hay_offside", False)),
        "advertencias": resultado.get("advertencias", []),
        "equipo_atacante_id": resultado.get("equipo_atacante_id"),
        "gol_a_derecha": resultado.get("gol_a_derecha"),
        "punto_referencia": resultado.get("punto_referencia", "medio"),
    }

    return {
        "result_image_b64": _pil_to_b64(result_img),
        "resultado": resultado_serial,
        "image_size": {"width": w, "height": h},
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
