"""Test de integración del API FastAPI con el backend ONNX, vía TestClient
(sin levantar un servidor real). Verifica health + /api/detect end-to-end.

Uso:
    .venv/Scripts/python.exe dev/test_api.py <ruta_imagen>
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
img_path = Path(sys.argv[1]).resolve()  # absoluta ANTES del chdir
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

with TestClient(main.app) as client:
    r = client.get("/api/health")
    print("HEALTH:", r.status_code, r.json())
    assert r.status_code == 200 and r.json()["model_loaded"] is True

    with open(img_path, "rb") as f:
        r = client.post("/api/detect", files={"image": ("img.jpg", f, "image/jpeg")})
    print("DETECT:", r.status_code)
    assert r.status_code == 200, r.text
    data = r.json()
    print("  detections:", len(data["detections"]))
    print("  class_counts:", data["class_counts"])
    print("  vp:", data["vp"])
    print("  image_size:", data["image_size"])
    assert len(data["detections"]) > 0
    assert data["detected_image_b64"]

print("\nAPI TEST OK  (health + detect end-to-end sobre ONNX)")
