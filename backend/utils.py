"""Core offside detection utilities. Adapted from prod/utils.py for FastAPI backend."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from hough import detectar_punto_de_fuga, punto_de_fuga_desde_puntos  # noqa: F401

_DEV_DIR = Path(__file__).resolve().parent.parent / "dev"
try:
    if str(_DEV_DIR) not in sys.path:
        sys.path.insert(0, str(_DEV_DIR))
    from custom_models import WeightedDetectionModel
    sys.modules["__main__"].WeightedDetectionModel = WeightedDetectionModel
except Exception:
    class WeightedDetectionModel:
        pass
    sys.modules["__main__"].WeightedDetectionModel = WeightedDetectionModel

_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    _ROOT / "dev" / "modelo.pt"
    if (_ROOT / "dev" / "modelo.pt").exists()
    else _ROOT / "dev" / "YOLOv26m Manual Weighted" / "exp5_yolo26m_manual_weighted.pt"
)

CLASES = {
    0: "Ball",
    1: "Corner",
    2: "GoalKeeper",
    3: "Goal_Net",
    4: "Referee",
    5: "TEAM 1",
    6: "TEAM 2",
}

_COLORES_BGR = {
    0: (255, 255, 255),
    2: (0, 215, 255),
    4: (0, 165, 255),
    5: (180, 105, 255),
    6: (220, 220, 0),
}


def cargar_modelo(ruta=None):
    from ultralytics import YOLO
    return YOLO(str(ruta or MODEL_PATH))


def reasignar_equipos_por_color(detecciones, imagen_pil):
    img_np = np.array(imagen_pil.convert("RGB"))
    h_img, w_img = img_np.shape[:2]

    jugadores_idx = [i for i, d in enumerate(detecciones) if d["class_id"] in (5, 6)]
    if len(jugadores_idx) < 2:
        return detecciones

    colores = []
    for i in jugadores_idx:
        d = detecciones[i]
        x1 = max(0, int(d["x1"]))
        y1 = max(0, int(d["y1"]))
        x2 = min(w_img, int(d["x2"]))
        y2 = min(h_img, int(d["y2"]))
        bbox_h = max(1, y2 - y1)
        ty1 = y1 + int(bbox_h * 0.20)
        ty2 = y1 + int(bbox_h * 0.60)
        bw = max(1, x2 - x1)
        tx1 = x1 + int(bw * 0.25)
        tx2 = x1 + int(bw * 0.75)
        crop = img_np[ty1:ty2, tx1:tx2] if ty2 > ty1 and tx2 > tx1 else img_np[y1:y2, x1:x2]
        if crop.size == 0:
            colores.append(np.zeros(3, dtype=np.float32))
            continue
        crop_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        colores.append(crop_lab.reshape(-1, 3).mean(axis=0).astype(np.float32))

    colores_arr = np.array(colores, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, _ = cv2.kmeans(colores_arr, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    resultado = list(detecciones)
    for orden, idx in enumerate(jugadores_idx):
        d = dict(resultado[idx])
        new_id = 5 if int(labels[orden]) == 0 else 6
        d["class_id"] = new_id
        d["class_name"] = CLASES[new_id]
        resultado[idx] = d

    return resultado


def detectar_objetos(modelo, imagen_pil, conf=0.25, iou=0.7, agnostic_nms=True, clustering=True):
    img_np = np.array(imagen_pil.convert("RGB"))
    resultados = modelo.predict(
        source=img_np,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        verbose=False,
    )

    detecciones = []
    if resultados and resultados[0].boxes is not None:
        boxes = resultados[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf_i = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detecciones.append({
                "class_id": cls_id,
                "class_name": CLASES.get(cls_id, str(cls_id)),
                "conf": conf_i,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    if clustering:
        detecciones = reasignar_equipos_por_color(detecciones, imagen_pil)

    return detecciones


def pie_jugador(det, modo="medio"):
    if modo == "izquierdo":
        return (det["x1"], det["y2"])
    elif modo == "derecho":
        return (det["x2"], det["y2"])
    return ((det["x1"] + det["x2"]) / 2, det["y2"])


def _proyectar_en_base(foot, vp, h):
    fx, fy = foot
    vp_x, vp_y = vp
    if abs(fy - vp_y) < 1e-6:
        return fx
    t = (h - fy) / (vp_y - fy)
    return fx + t * (vp_x - fx)


def calcular_offside(
    vp, detecciones, equipo_atacante_id, imagen_shape,
    gol_a_derecha=None, ref_defender_idx=None, punto_referencia="medio"
):
    h, w = imagen_shape[:2]
    equipo_defensor_id = 6 if equipo_atacante_id == 5 else 5
    advertencias = []

    atacantes = [d for d in detecciones if d["class_id"] == equipo_atacante_id]
    equipo_def = [d for d in detecciones if d["class_id"] == equipo_defensor_id]
    _arqueros_all = [d for d in detecciones if d["class_id"] == 2]
    arqueros = [max(_arqueros_all, key=lambda d: d["conf"])] if _arqueros_all else []
    defensores = equipo_def + arqueros

    if not atacantes:
        advertencias.append("No se detectaron jugadores del equipo atacante seleccionado.")
    if not defensores:
        advertencias.append("No se detectaron defensores.")
        return {
            "penultimo_defensor": None,
            "linea_foot": None,
            "linea_pelota_foot": None,
            "atacantes_resultado": [(a, False) for a in atacantes],
            "hay_offside": False,
            "advertencias": advertencias,
            "equipo_atacante_id": equipo_atacante_id,
            "gol_a_derecha": True,
            "punto_referencia": punto_referencia,
        }
    if not arqueros:
        advertencias.append("No se detectó arquero — la referencia de offside puede ser imprecisa.")

    _pelotas_all = [d for d in detecciones if d["class_id"] == 0]
    pelota = max(_pelotas_all, key=lambda d: d["conf"]) if _pelotas_all else None
    if pelota is None:
        advertencias.append("No se detectó pelota — no se puede verificar si el atacante está detrás del balón.")

    def proj(det):
        return _proyectar_en_base(pie_jugador(det, modo=punto_referencia), vp, h)

    def proj_centro(det):
        cx = (det["x1"] + det["x2"]) / 2
        cy = (det["y1"] + det["y2"]) / 2
        return _proyectar_en_base((cx, cy), vp, h)

    if gol_a_derecha is None:
        todos = [d for d in detecciones if d["class_id"] in (2, 5, 6)]
        promedio_proj = np.mean([proj(d) for d in todos]) if todos else w / 2.0
        if arqueros:
            gol_a_derecha = bool(proj(arqueros[0]) > promedio_proj)
        else:
            gol_a_derecha = bool(np.mean([proj(d) for d in defensores]) > promedio_proj)

    defensores_ord = sorted(defensores, key=proj, reverse=gol_a_derecha)

    if ref_defender_idx is not None:
        if 0 <= ref_defender_idx < len(defensores_ord):
            penultimo = defensores_ord[ref_defender_idx]
        else:
            penultimo = defensores_ord[-1] if defensores_ord else None
            advertencias.append(f"No hay suficientes defensores para el índice {ref_defender_idx + 1}.")
    else:
        if arqueros:
            if len(defensores_ord) < 2:
                penultimo = defensores_ord[0]
                advertencias.append("Solo un defensor detectado; se usa como referencia de offside.")
            else:
                penultimo = defensores_ord[1]
        else:
            penultimo = defensores_ord[0] if defensores_ord else None
            if penultimo is None:
                advertencias.append("No se detectaron defensores de campo.")

    if penultimo is not None:
        penultimo_proj = proj(penultimo)
        penultimo_foot = pie_jugador(penultimo, modo=punto_referencia)
    else:
        penultimo_proj = 0
        penultimo_foot = None

    pelota_proj = proj_centro(pelota) if pelota is not None else None
    pelota_foot = (
        ((pelota["x1"] + pelota["x2"]) / 2, (pelota["y1"] + pelota["y2"]) / 2)
        if pelota is not None else None
    )

    atacantes_resultado = []
    salvados_por_pelota = 0
    for atk in atacantes:
        atk_proj = proj(atk)
        adelantado_al_defensor = (atk_proj > penultimo_proj) if gol_a_derecha else (atk_proj < penultimo_proj)
        adelantado_a_pelota = True
        if pelota_proj is not None:
            adelantado_a_pelota = (atk_proj > pelota_proj) if gol_a_derecha else (atk_proj < pelota_proj)
        en_offside = adelantado_al_defensor and adelantado_a_pelota
        if adelantado_al_defensor and not adelantado_a_pelota:
            salvados_por_pelota += 1
        atacantes_resultado.append((atk, en_offside))

    hay_offside = any(os for _, os in atacantes_resultado)
    linea_pelota_foot = pelota_foot if salvados_por_pelota > 0 else None

    return {
        "penultimo_defensor": penultimo,
        "linea_foot": penultimo_foot,
        "linea_pelota_foot": linea_pelota_foot,
        "atacantes_resultado": atacantes_resultado,
        "hay_offside": hay_offside,
        "advertencias": advertencias,
        "equipo_atacante_id": equipo_atacante_id,
        "gol_a_derecha": gol_a_derecha,
        "punto_referencia": punto_referencia,
    }


def _puntos_en_borde(fx, fy, vp_x, vp_y, w, h):
    dx, dy = vp_x - fx, vp_y - fy
    puntos = set()
    candidatos = [
        (0 - fy,  dy, lambda t: (fx + t * dx, 0.0)),
        (h - fy,  dy, lambda t: (fx + t * dx, float(h))),
        (0 - fx,  dx, lambda t: (0.0, fy + t * dy)),
        (w - fx,  dx, lambda t: (float(w), fy + t * dy)),
    ]
    for t_num, t_den, xy_fn in candidatos:
        if abs(t_den) < 1e-9:
            continue
        t = t_num / t_den
        x, y = xy_fn(t)
        if -1 <= x <= w + 1 and -1 <= y <= h + 1:
            puntos.add((round(x, 1), round(y, 1)))
    pts = sorted(puntos)
    return pts[:2] if len(pts) >= 2 else list(pts)


def dibujar_resultado(imagen_pil, detecciones, vp, resultado_offside):
    img = cv2.cvtColor(np.array(imagen_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    equipo_atk_id = resultado_offside.get("equipo_atacante_id") if resultado_offside else None
    atk_offside_map = {}
    if resultado_offside:
        for det, en_offside in resultado_offside.get("atacantes_resultado", []):
            atk_offside_map[(round(det["x1"]), round(det["y1"]))] = en_offside

    _class_counters = {}
    for det in detecciones:
        cls_id = det["class_id"]
        if cls_id not in _COLORES_BGR:
            continue
        _class_counters[cls_id] = _class_counters.get(cls_id, 0) + 1
        n = _class_counters[cls_id]
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        key = (round(det["x1"]), round(det["y1"]))

        if cls_id == equipo_atk_id and key in atk_offside_map:
            en_offside = atk_offside_map[key]
            color = (0, 0, 220) if en_offside else (0, 200, 50)
            grosor = 3
            label = f"#{n} OFFSIDE" if en_offside else f"#{n} ONSIDE"
        else:
            color = _COLORES_BGR[cls_id]
            grosor = 2
            label = f"#{n} {det['class_name']}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    if resultado_offside and resultado_offside.get("linea_foot") and vp:
        fx, fy = resultado_offside["linea_foot"]
        pts = _puntos_en_borde(fx, fy, vp[0], vp[1], w, h)
        if len(pts) >= 2:
            pt1 = (int(round(pts[0][0])), int(round(pts[0][1])))
            pt2 = (int(round(pts[1][0])), int(round(pts[1][1])))
            cv2.line(img, pt1, pt2, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "Linea offside",
                        (min(pt1[0], pt2[0]) + 5, max(min(pt1[1], pt2[1]) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (int(fx), int(fy)), 8, (0, 140, 255), -1, cv2.LINE_AA)

    if resultado_offside and resultado_offside.get("linea_pelota_foot") and vp:
        bx, by = resultado_offside["linea_pelota_foot"]
        pts = _puntos_en_borde(bx, by, vp[0], vp[1], w, h)
        if len(pts) >= 2:
            pt1 = (int(round(pts[0][0])), int(round(pts[0][1])))
            pt2 = (int(round(pts[1][0])), int(round(pts[1][1])))
            cv2.line(img, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "Linea pelota",
                        (min(pt1[0], pt2[0]) + 5, max(min(pt1[1], pt2[1]) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (int(bx), int(by)), 6, (255, 200, 0), -1, cv2.LINE_AA)

    if resultado_offside and vp:
        punto_ref = resultado_offside.get("punto_referencia", "medio")
        for det, en_offside in resultado_offside.get("atacantes_resultado", []):
            if en_offside:
                fx, fy = pie_jugador(det, modo=punto_ref)
                pts = _puntos_en_borde(fx, fy, vp[0], vp[1], w, h)
                if len(pts) >= 2:
                    pt1 = (int(round(pts[0][0])), int(round(pts[0][1])))
                    pt2 = (int(round(pts[1][0])), int(round(pts[1][1])))
                    cv2.line(img, pt1, pt2, (0, 0, 220), 2, cv2.LINE_AA)
                cv2.circle(img, (int(fx), int(fy)), 6, (0, 0, 220), -1, cv2.LINE_AA)

    if vp:
        vx, vy = int(round(vp[0])), int(round(vp[1]))
        if 0 <= vx < w and 0 <= vy < h:
            cv2.circle(img, (vx, vy), 10, (0, 0, 180), -1, cv2.LINE_AA)
            cv2.circle(img, (vx, vy), 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"VP ({vp[0]:.0f},{vp[1]:.0f})",
                        (vx + 14, max(vy - 14, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
