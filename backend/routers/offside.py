import base64
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image

from backend.storage import get_info
from backend.model import (
    calcular_offside,
    dibujar_resultado,
    punto_de_fuga_desde_puntos,
)

router = APIRouter()


class VPRequest(BaseModel):
    points: list[list[float]]


class Detection(BaseModel):
    class_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class AnalyzeRequest(BaseModel):
    media_id: str
    frame_index: int = 0
    detections: list[Detection]
    vp: list[float]
    attacking_team_id: int
    direction: Optional[bool] = None
    ref_defender_idx: Optional[int] = None
    foot_point: str = "medio"


def _load_pil(media_id: str, frame_index: int) -> Image.Image:
    info = get_info(media_id)
    if not info:
        raise HTTPException(404, "Media not found")

    ext = info["ext"]
    video_exts = {"mp4", "avi", "mov", "mkv", "m4v"}

    if ext in video_exts:
        cap = cv2.VideoCapture(info["path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(0, min(frame_index, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(400, "Cannot read video frame")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        return Image.open(info["path"]).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


@router.post("/vanishing-point")
def vanishing_point(req: VPRequest):
    if len(req.points) != 4:
        raise HTTPException(400, "Exactly 4 points required")

    pts = [tuple(p) for p in req.points]
    vp = punto_de_fuga_desde_puntos(pts)

    if vp is None:
        return {"vp": None, "error": "Lines are parallel — choose points on converging lines"}

    return {"vp": list(vp)}


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    pil = _load_pil(req.media_id, req.frame_index)
    img_shape = np.array(pil).shape

    dets = [d.model_dump() for d in req.detections]
    vp = tuple(req.vp)

    result = calcular_offside(
        vp,
        dets,
        req.attacking_team_id,
        img_shape,
        gol_a_derecha=req.direction,
        ref_defender_idx=req.ref_defender_idx,
        punto_referencia=req.foot_point,
    )

    img_annotated = dibujar_resultado(pil, dets, vp, result)
    img_b64 = _pil_to_b64(img_annotated)

    # Serialize result (convert tuples to JSON-safe types)
    attackers = [
        {
            "detection": det,
            "offside": bool(en_offside),
        }
        for det, en_offside in result["atacantes_resultado"]
    ]

    penultimate = result.get("penultimo_defensor")
    linea_foot = result.get("linea_foot")

    return {
        "annotated_image_base64": img_b64,
        "hay_offside": result["hay_offside"],
        "gol_a_derecha": result["gol_a_derecha"],
        "advertencias": result["advertencias"],
        "atacantes_resultado": attackers,
        "penultimo_defensor": penultimate,
        "linea_foot": list(linea_foot) if linea_foot else None,
        "equipo_atacante_id": result["equipo_atacante_id"],
    }
