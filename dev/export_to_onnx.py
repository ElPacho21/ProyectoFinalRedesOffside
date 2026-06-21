"""Exporta dev/modelo.pt a ONNX para el deploy del backend sin torch/ultralytics.

El modelo entrenado es un WeightedDetectionModel (custom solo en la loss de
entrenamiento). Para inferencia es un DetectionModel estándar de YOLO, así que
el ONNX exportado tiene el formato de salida estándar (1, 4+nc, num_anchors).

Uso:
    .venv/Scripts/python.exe dev/export_to_onnx.py
"""
import shutil
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "dev"
BACKEND = ROOT / "backend"

# --- registrar la clase custom para que torch pueda des-serializar el .pt ---
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))
from custom_models import WeightedDetectionModel  # noqa: E402

sys.modules["__main__"].WeightedDetectionModel = WeightedDetectionModel

from ultralytics import YOLO  # noqa: E402

PT_PATH = DEV / "modelo.pt"
IMGSZ = 640


def main():
    print(f"Cargando {PT_PATH} ...")
    model = YOLO(str(PT_PATH))
    print("Nombres de clases del modelo:", model.names)

    print(f"Exportando a ONNX (imgsz={IMGSZ}, dynamic=False, simplify=True) ...")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        dynamic=False,
        simplify=True,
        opset=12,
    )
    onnx_path = Path(onnx_path)
    print(f"ONNX generado en: {onnx_path}")

    dest = BACKEND / "modelo.onnx"
    shutil.copyfile(onnx_path, dest)
    print(f"Copiado a: {dest}")

    # --- inspeccionar shapes de entrada/salida ---
    m = onnx.load(str(dest))

    def _shape(t):
        return [d.dim_value if (d.dim_value > 0) else d.dim_param
                for d in t.type.tensor_type.shape.dim]

    print("\n=== ONNX I/O ===")
    for inp in m.graph.input:
        print(f"INPUT  {inp.name}: {_shape(inp)}")
    for out in m.graph.output:
        print(f"OUTPUT {out.name}: {_shape(out)}")

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"\nTamaño modelo.onnx: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
