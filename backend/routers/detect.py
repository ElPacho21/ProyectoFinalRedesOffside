import base64
from io import BytesIO
from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image

from backend.storage import get_info
from backend.model import get_model, detectar_objetos, dibujar_resultado

router = APIRouter()


def _load_pil(media_id: str, frame_index: int = 0) -> Image.Image:
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


class DetectRequest(BaseModel):
    media_id: str
    frame_index: int = 0
    confidence: float = 0.25


@router.post("/detect")
def detect(req: DetectRequest):
    pil = _load_pil(req.media_id, req.frame_index)
    model = get_model()

    detections = detectar_objetos(model, pil, conf=req.confidence)

    img_annotated = dibujar_resultado(pil, detections, vp=None, resultado_offside=None)
    img_b64 = _pil_to_b64(img_annotated)

    conteo = Counter(d["class_name"] for d in detections)

    warnings = []
    if conteo.get("TEAM 1", 0) == 0 and conteo.get("TEAM 2", 0) == 0:
        warnings.append("No players detected from either team.")
    elif conteo.get("TEAM 1", 0) == 0 or conteo.get("TEAM 2", 0) == 0:
        warnings.append("Only one team detected. Offside result may be unreliable.")
    if conteo.get("GoalKeeper", 0) == 0:
        warnings.append("No goalkeeper detected. Offside line may be imprecise.")

    # Auto-suggest attacking team (closest to ball)
    pelotas = [d for d in detections if d["class_id"] == 0]
    team1 = [d for d in detections if d["class_id"] == 5]
    team2 = [d for d in detections if d["class_id"] == 6]
    suggested_team = None
    if pelotas and (team1 or team2):
        pelota = pelotas[0]
        px = (pelota["x1"] + pelota["x2"]) / 2
        py = (pelota["y1"] + pelota["y2"]) / 2

        def dist_min(team):
            if not team:
                return float("inf")
            return min(
                ((d["x1"] + d["x2"]) / 2 - px) ** 2 + ((d["y1"] + d["y2"]) / 2 - py) ** 2
                for d in team
            )

        suggested_team = 5 if dist_min(team1) <= dist_min(team2) else 6

    return {
        "detections": detections,
        "image_base64": img_b64,
        "counts": dict(conteo),
        "warnings": warnings,
        "suggested_team": suggested_team,
    }
