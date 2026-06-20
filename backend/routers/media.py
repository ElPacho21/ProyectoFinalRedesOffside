import base64
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image

from backend.storage import save_upload, get_info

router = APIRouter()

_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "m4v"}


def _pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    is_video = ext in _VIDEO_EXTS

    media_id = save_upload(file.filename, data)
    info = get_info(media_id)

    if is_video:
        cap = cv2.VideoCapture(info["path"])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Preview frame 0
        cap = cv2.VideoCapture(info["path"])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(400, "Cannot read video")
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        preview = _pil_to_b64(pil)

        return {
            "media_id": media_id,
            "is_video": True,
            "width": w,
            "height": h,
            "total_frames": total_frames,
            "fps": fps,
            "preview_base64": preview,
        }
    else:
        img = Image.open(BytesIO(data)).convert("RGB")
        w, h = img.size
        preview = _pil_to_b64(img)
        return {
            "media_id": media_id,
            "is_video": False,
            "width": w,
            "height": h,
            "preview_base64": preview,
        }


@router.get("/frame/{media_id}")
def get_frame(media_id: str, index: int = 0):
    info = get_info(media_id)
    if not info:
        raise HTTPException(404, "Media not found")

    cap = cv2.VideoCapture(info["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    index = max(0, min(index, total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, "Cannot read frame")

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w, h = pil.size
    ts = index / fps

    return {
        "frame_base64": _pil_to_b64(pil),
        "width": w,
        "height": h,
        "timestamp": ts,
        "total_frames": total,
        "fps": fps,
    }
