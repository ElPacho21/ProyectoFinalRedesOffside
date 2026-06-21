"""Core offside detection utilities. Adapted from prod/utils.py for FastAPI backend.

La inferencia corre sobre ONNX Runtime (sin torch ni ultralytics) para que el
backend quede liviano en deploys serverless. El modelo es un YOLO26m exportado
end-to-end (NMS-free): su salida ONNX es (1, 300, 6) =
[x1, y1, x2, y2, conf, class] en coordenadas de píxel del input con letterbox.
"""
import os
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from hough import detectar_punto_de_fuga, punto_de_fuga_desde_puntos  # noqa: F401

MODEL_PATH = Path(__file__).resolve().parent / "modelo.onnx"

CLASES = {
    0: "Ball",
    1: "Corner",
    2: "GoalKeeper",
    3: "Goal_Net",
    4: "Referee",
    5: "TEAM 1",
    6: "TEAM 2",
}

# Per-class confidence thresholds (configurable via env)
CONF_POR_CLASE = {
    0: float(os.getenv("CONF_BALL",       "0.05")),  # Ball — ultra-low, keep best only
    1: float(os.getenv("CONF_CORNER",     "0.25")),  # Corner flag
    2: float(os.getenv("CONF_GOALKEEPER", "0.25")),  # GoalKeeper
    3: float(os.getenv("CONF_GOAL_NET",   "0.05")),  # Goal_Net — ultra-low, keep best only
    4: float(os.getenv("CONF_REFEREE",    "0.25")),  # Referee
    5: float(os.getenv("CONF_TEAM",       "0.30")),  # TEAM 1
    6: float(os.getenv("CONF_TEAM",       "0.30")),  # TEAM 2
}

# Classes where only the single highest-confidence detection is kept
_KEEP_BEST_ONLY = {0, 3}  # Ball, Goal_Net

_COLORES_BGR = {
    0: (255, 255, 255),   # Ball — white
    1: (0, 255, 200),     # Corner — cyan-green
    2: (0, 215, 255),     # GoalKeeper — yellow
    3: (0, 100, 255),     # Goal_Net — orange-red
    4: (0, 165, 255),     # Referee — orange
    5: (180, 105, 255),   # TEAM 1 — pink
    6: (220, 220, 0),     # TEAM 2 — cyan
}


class _OnnxModel:
    """Wrapper liviano sobre onnxruntime con lo que necesita la inferencia."""

    def __init__(self, ruta):
        self.session = ort.InferenceSession(
            str(ruta), providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # input shape [1, 3, H, W]; H == W para este modelo
        self.imgsz = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640


def cargar_modelo(ruta=None):
    return _OnnxModel(str(ruta or MODEL_PATH))


def _letterbox(img, new_shape, color=(114, 114, 114)):
    """Redimensiona manteniendo aspect ratio y rellena hasta new_shape×new_shape.
    Réplica de LetterBox(auto=False) de ultralytics. Devuelve (img, r, dw, dh)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape - nw) / 2.0, (new_shape - nh) / 2.0
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return (
        cv2.copyMakeBorder(resized, top, bottom, left, right,
                           cv2.BORDER_CONSTANT, value=color),
        r, dw, dh,
    )


def _inferir(modelo, img_rgb, min_conf):
    """Corre el modelo ONNX y devuelve detecciones crudas en coordenadas de la
    imagen original: lista de (class_id, conf, x1, y1, x2, y2).

    ultralytics, al recibir un array numpy en predict(), lo trata como BGR (hace
    im[..., ::-1] internamente), por lo que el modelo fue entrenado/usado viendo
    BGR. Reproducimos ese orden de canales para paridad exacta con el pipeline
    original (validado: bit-exact en 640×640 nativo, <1.2px reescalado)."""
    feed = np.ascontiguousarray(img_rgb[..., ::-1])  # RGB → BGR
    padded, r, dw, dh = _letterbox(feed, modelo.imgsz)
    blob = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    blob = np.ascontiguousarray(blob)
    out = modelo.session.run(None, {modelo.input_name: blob})[0][0]  # (300, 6)

    h, w = img_rgb.shape[:2]
    dets = []
    for x1, y1, x2, y2, conf, cls in out:
        if conf < min_conf:
            continue
        x1 = min(max((x1 - dw) / r, 0.0), w)
        y1 = min(max((y1 - dh) / r, 0.0), h)
        x2 = min(max((x2 - dw) / r, 0.0), w)
        y2 = min(max((y2 - dh) / r, 0.0), h)
        dets.append((int(round(cls)), float(conf),
                     float(x1), float(y1), float(x2), float(y2)))
    return dets


# --- Tuning for color-based team clustering (env-overridable) ---
# Peso del canal L (luminosidad) en la distancia de color. <1 prioriza
# tono/croma (a,b de LAB) sobre brillo → más robusto a sombras (mejora #5).
_L_WEIGHT = float(os.getenv("COLOR_L_WEIGHT", "0.5"))
# Distancia máxima (espacio LAB ponderado) para considerar que un "jugador"
# coincide con el color del arquero/árbitro y excluirlo (mejora #3).
_ANCLA_DIST = float(os.getenv("COLOR_ANCLA_DIST", "28.0"))
# Distancia máxima de un jugador a su centroide de equipo; si la supera se
# considera intruso y se descarta del equipo (mejora #1).
_OUTLIER_DIST = float(os.getenv("COLOR_OUTLIER_DIST", "55.0"))
# Distancia LAB para considerar que un píxel es césped (vs el ref de césped
# muestreado alrededor del jugador). Camiseta verde de otro tono sobrevive.
_GRASS_DIST = float(os.getenv("COLOR_GRASS_DIST", "22.0"))


def _torso_crop(d, img_np):
    """Recorta la franja central del torso (evita césped/piernas/cabeza)."""
    h_img, w_img = img_np.shape[:2]
    x1 = max(0, int(d["x1"])); y1 = max(0, int(d["y1"]))
    x2 = min(w_img, int(d["x2"])); y2 = min(h_img, int(d["y2"]))
    bbox_h = max(1, y2 - y1); bw = max(1, x2 - x1)
    ty1 = y1 + int(bbox_h * 0.20); ty2 = y1 + int(bbox_h * 0.60)
    tx1 = x1 + int(bw * 0.25); tx2 = x1 + int(bw * 0.75)
    if ty2 > ty1 and tx2 > tx1:
        return img_np[ty1:ty2, tx1:tx2]
    return img_np[y1:y2, x1:x2]


def _grass_lab(d, img_np):
    """Estima el color del césped muestreando franjas a izq/der/abajo del
    bbox del jugador. Devuelve mediana LAB o None si no hay margen.
    Permite distinguir césped de una camiseta verde de otro tono."""
    h_img, w_img = img_np.shape[:2]
    x1 = max(0, int(d["x1"])); y1 = max(0, int(d["y1"]))
    x2 = min(w_img, int(d["x2"])); y2 = min(h_img, int(d["y2"]))
    bw = max(1, x2 - x1); bh = max(1, y2 - y1)
    m = max(2, int(bw * 0.30))  # ancho de la franja lateral
    yc1 = y1 + int(bh * 0.40); yc2 = y2  # mitad inferior (suele ser césped)
    muestras = []
    izq = img_np[yc1:yc2, max(0, x1 - m):x1]
    der = img_np[yc1:yc2, x2:min(w_img, x2 + m)]
    aba = img_np[y2:min(h_img, y2 + m), x1:x2]
    for patch in (izq, der, aba):
        if patch.size:
            muestras.append(patch.reshape(-1, 3))
    if not muestras:
        return None
    rgb = np.concatenate(muestras, axis=0).reshape(1, -1, 3).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return np.median(lab, axis=0).astype(np.float32)


def _color_camiseta(crop_rgb, grass_lab=None):
    """Color dominante de la camiseta: mediana LAB sobre píxeles que NO
    coinciden con el césped muestreado localmente (mejora #4). Si no hay ref
    de césped, usa todo el crop. None si crop vacío."""
    if crop_rgb is None or crop_rgb.size == 0:
        return None
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    if grass_lab is not None:
        dist = np.linalg.norm(lab - grass_lab.astype(np.float32), axis=1)
        valido = dist > _GRASS_DIST
        # Si casi todo matchea césped (jugador chico/lejano), usar todo el crop.
        if valido.sum() >= max(10, int(0.10 * valido.size)):
            lab = lab[valido]
    if lab.size == 0:
        return None
    return np.median(lab, axis=0).astype(np.float32)


def _feat(col):
    """Vector de color ponderado para distancias/kmeans (mejora #5)."""
    return np.array([col[0] * _L_WEIGHT, col[1], col[2]], dtype=np.float32)


def reasignar_equipos_por_color(detecciones, imagen_pil):
    img_np = np.array(imagen_pil.convert("RGB"))

    jugadores_idx = [i for i, d in enumerate(detecciones) if d["class_id"] in (5, 6)]
    if len(jugadores_idx) < 2:
        return detecciones

    resultado = [dict(d) for d in detecciones]

    # --- Mejora #3: anclas de color desde arquero/árbitro ya detectados ---
    anclas = []  # (class_id, feat)
    for d in detecciones:
        if d["class_id"] in (2, 4):
            col = _color_camiseta(_torso_crop(d, img_np), _grass_lab(d, img_np))
            if col is not None:
                anclas.append((d["class_id"], _feat(col)))

    # Color de cada jugador
    feats = {}  # idx -> feat
    for idx in jugadores_idx:
        d = detecciones[idx]
        col = _color_camiseta(_torso_crop(d, img_np), _grass_lab(d, img_np))
        feats[idx] = _feat(col) if col is not None else np.zeros(3, dtype=np.float32)

    # --- Mejora #2: kmeans k=2 sobre TODOS los jugadores → 2 equipos ---
    # k=2 fijo: nunca parte un equipo real en un 3er cluster espurio.
    feat_arr = np.array([feats[i] for i in jugadores_idx], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centros = cv2.kmeans(
        feat_arr, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    lbl_a_equipo = {0: 5, 1: 6}

    for orden, idx in enumerate(jugadores_idx):
        lbl = int(labels[orden])
        f = feat_arr[orden]
        d_own = float(np.linalg.norm(f - centros[lbl]))

        # --- Mejora #3: ¿más parecido a arquero/árbitro que a su propio equipo? ---
        mejor_id, mejor_d = None, np.inf
        for cid, af in anclas:
            dist = float(np.linalg.norm(f - af))
            if dist < mejor_d:
                mejor_d, mejor_id = dist, cid
        if mejor_id is not None and mejor_d < _ANCLA_DIST and mejor_d < d_own:
            # Solo si el match al ancla es mejor que su propio equipo → no es jugador.
            resultado[idx]["class_id"] = mejor_id
            resultado[idx]["class_name"] = CLASES[mejor_id]
            continue

        # --- Mejora #1: rechazo de outlier por distancia al centroide del equipo ---
        if d_own > _OUTLIER_DIST:
            resultado[idx]["class_id"] = 4
            resultado[idx]["class_name"] = CLASES[4]
            continue

        new_id = lbl_a_equipo[lbl]
        resultado[idx]["class_id"] = new_id
        resultado[idx]["class_name"] = CLASES[new_id]

    return resultado


def detectar_objetos(modelo, imagen_pil, conf=None, iou=0.7, agnostic_nms=True, clustering=True):
    # iou/agnostic_nms se ignoran: el modelo es end-to-end (NMS-free).
    # Run at the lowest per-class threshold so nothing is missed before filtering
    min_conf = min(CONF_POR_CLASE.values())
    img_np = np.array(imagen_pil.convert("RGB"))
    raw = _inferir(modelo, img_np, min_conf)

    detecciones = []
    for cls_id, conf_i, x1, y1, x2, y2 in raw:
        umbral = CONF_POR_CLASE.get(cls_id, min_conf)
        if conf_i < umbral:
            continue
        detecciones.append({
            "class_id": cls_id,
            "class_name": CLASES.get(cls_id, str(cls_id)),
            "conf": conf_i,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    # For certain classes keep only the single highest-confidence detection
    filtered = []
    best_per_class: dict = {}
    for d in detecciones:
        cid = d["class_id"]
        if cid in _KEEP_BEST_ONLY:
            if cid not in best_per_class or d["conf"] > best_per_class[cid]["conf"]:
                best_per_class[cid] = d
        else:
            filtered.append(d)
    filtered.extend(best_per_class.values())
    detecciones = filtered

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
        atacantes_resultado.append((atk, en_offside, adelantado_al_defensor, adelantado_a_pelota))

    hay_offside = any(os for _, os, *_ in atacantes_resultado)
    linea_pelota_foot = pelota_foot if salvados_por_pelota > 0 else None

    return {
        "penultimo_defensor": penultimo,
        "pelota": pelota,
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
        for det, en_offside, *_ in resultado_offside.get("atacantes_resultado", []):
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
            cv2.putText(img, "Línea offside",
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
            cv2.putText(img, "Línea pelota",
                        (min(pt1[0], pt2[0]) + 5, max(min(pt1[1], pt2[1]) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (int(bx), int(by)), 6, (255, 200, 0), -1, cv2.LINE_AA)

    if resultado_offside and vp:
        punto_ref = resultado_offside.get("punto_referencia", "medio")
        for det, en_offside, *_ in resultado_offside.get("atacantes_resultado", []):
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
            cv2.putText(img, f"Punto de fuga ({vp[0]:.0f},{vp[1]:.0f})",
                        (vx + 14, max(vy - 14, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
