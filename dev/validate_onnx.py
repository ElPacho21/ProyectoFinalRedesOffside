"""Compara el pipeline ultralytics (.pt) vs onnxruntime (.onnx) para asegurar
paridad de detecciones antes de migrar utils.py.

Verifica:
  - convención de canales (RGB vs BGR) que matchea ultralytics
  - letterbox correcto (probando una imagen no-cuadrada)
  - formato de la salida (1, 300, 6) = [x1, y1, x2, y2, conf, cls]

Uso:
    .venv/Scripts/python.exe dev/validate_onnx.py <ruta_imagen>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))
from custom_models import WeightedDetectionModel  # noqa: E402

sys.modules["__main__"].WeightedDetectionModel = WeightedDetectionModel

PT_PATH = DEV / "modelo.pt"
ONNX_PATH = ROOT / "backend" / "modelo.onnx"
IMGSZ = 640
MIN_CONF = 0.05
CLASES = {0: "Ball", 1: "Corner", 2: "GoalKeeper", 3: "Goal_Net",
          4: "Referee", 5: "TEAM 1", 6: "TEAM 2"}


# ── ONNX pipeline (lo que irá a utils.py) ─────────────────────────────────────

def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape - nw) / 2.0, (new_shape - nh) / 2.0
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, dw, dh


def onnx_infer(session, input_name, img_rgb, flip_to_bgr, min_conf=MIN_CONF):
    feed = img_rgb[..., ::-1] if flip_to_bgr else img_rgb
    padded, r, dw, dh = _letterbox(np.ascontiguousarray(feed), IMGSZ)
    blob = padded.astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
    out = session.run(None, {input_name: blob})[0][0]  # (300, 6)
    dets = []
    for row in out:
        x1, y1, x2, y2, conf, cls = row
        if conf < min_conf:
            continue
        X1 = (x1 - dw) / r
        Y1 = (y1 - dh) / r
        X2 = (x2 - dw) / r
        Y2 = (y2 - dh) / r
        dets.append((int(round(cls)), float(conf), X1, Y1, X2, Y2))
    return dets


def _fmt(dets):
    out = []
    for cls, conf, x1, y1, x2, y2 in sorted(dets, key=lambda d: (d[0], -d[1])):
        out.append(f"  {CLASES.get(cls, cls):>10} {conf:.3f}  "
                   f"[{x1:6.1f},{y1:6.1f},{x2:6.1f},{y2:6.1f}]")
    return "\n".join(out)


def _match_stats(ref, test):
    """Para cada det de ref, busca el más cercano en test (mismo cls) y mide
    error de caja y de confianza."""
    if not ref:
        return 0, 0.0, 0.0
    matched, max_box_err, max_conf_err = 0, 0.0, 0.0
    for c, cf, x1, y1, x2, y2 in ref:
        cand = [t for t in test if t[0] == c]
        if not cand:
            continue
        best = min(cand, key=lambda t: abs(t[2]-x1)+abs(t[3]-y1)+abs(t[4]-x2)+abs(t[5]-y2))
        box_err = max(abs(best[2]-x1), abs(best[3]-y1), abs(best[4]-x2), abs(best[5]-y2))
        matched += 1
        max_box_err = max(max_box_err, box_err)
        max_conf_err = max(max_conf_err, abs(best[1]-cf))
    return matched, max_box_err, max_conf_err


def run_ultra(model, img_np):
    res = model.predict(source=img_np, conf=MIN_CONF, iou=0.7,
                        agnostic_nms=True, verbose=False)
    dets = []
    if res and res[0].boxes is not None:
        b = res[0].boxes
        for i in range(len(b)):
            x1, y1, x2, y2 = b.xyxy[i].tolist()
            dets.append((int(b.cls[i].item()), float(b.conf[i].item()), x1, y1, x2, y2))
    return dets


def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    from ultralytics import YOLO
    model = YOLO(str(PT_PATH))
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    pil = Image.open(img_path).convert("RGB")
    for tag, img in [("CUADRADA (orig)", pil),
                     ("NO-CUADRADA (1280x720)", pil.resize((1280, 720)))]:
        img_np = np.array(img)
        ultra = run_ultra(model, img_np)
        onnx_bgr = onnx_infer(sess, input_name, img_np, flip_to_bgr=True)
        onnx_rgb = onnx_infer(sess, input_name, img_np, flip_to_bgr=False)

        print(f"\n{'='*70}\n{tag}  —  ultralytics={len(ultra)} dets\n{'='*70}")
        print("[ULTRALYTICS]")
        print(_fmt(ultra))
        for name, od in [("ONNX feed=BGR(flip)", onnx_bgr), ("ONNX feed=RGB", onnx_rgb)]:
            m, be, ce = _match_stats(ultra, od)
            print(f"\n[{name}]  n={len(od)}  matched={m}/{len(ultra)}  "
                  f"max_box_err={be:.2f}px  max_conf_err={ce:.4f}")
            print(_fmt(od))


if __name__ == "__main__":
    main()
