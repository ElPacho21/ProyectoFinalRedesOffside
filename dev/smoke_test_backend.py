"""Smoke test del backend con el pipeline ONNX integrado (cargar_modelo +
detectar_objetos, incluyendo clustering por color). Verifica que todo el camino
corre sin torch/ultralytics y produce detecciones.

Uso:
    .venv/Scripts/python.exe dev/smoke_test_backend.py <ruta_imagen>
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import utils  # noqa: E402

# Cortafuegos: asegura que NO se importó torch ni ultralytics
for mod in ("torch", "ultralytics"):
    assert mod not in sys.modules, f"¡{mod} fue importado! El backend no quedó limpio."

img_path = Path(sys.argv[1])
img = Image.open(img_path).convert("RGB")

model = utils.cargar_modelo()
print(f"Modelo ONNX cargado: imgsz={model.imgsz}, input={model.input_name}")

dets = utils.detectar_objetos(model, img)
print(f"\nDetecciones finales (post-umbral + keep-best + clustering): {len(dets)}")

counts = {}
for d in dets:
    counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
for name in sorted(counts):
    print(f"  {name:>12}: {counts[name]}")

# Verifica que dibujar_resultado tampoco rompe
out_img = utils.dibujar_resultado(img, dets, None, None)
print(f"\ndibujar_resultado OK -> {out_img.size}")
print("\nSMOKE TEST OK  (sin torch ni ultralytics)")
