"""
Funciones auxiliares para la app de detección de offside.
Carga del modelo, detección, geometría de offside y visualización.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Importar HoughLines desde dev/ (funciones de punto de fuga)
_DEV_DIR = Path(__file__).resolve().parent.parent / "dev"
sys.path.insert(0, str(_DEV_DIR))
from HoughLines import detectar_punto_de_fuga, punto_de_fuga_desde_puntos  # noqa: E402

# Registrar WeightedDetectionModel en __main__ para evitar errores de deserialización al cargar modelos entrenados con pesos
try:
    from dev.custom_models import WeightedDetectionModel
    sys.modules['__main__'].WeightedDetectionModel = WeightedDetectionModel
except Exception:
    try:
        from custom_models import WeightedDetectionModel
        sys.modules['__main__'].WeightedDetectionModel = WeightedDetectionModel
    except Exception:
        class WeightedDetectionModel:
            pass
        sys.modules['__main__'].WeightedDetectionModel = WeightedDetectionModel


# Rutas: buscamos primero dev/modelo.pth (nombre canónico del TP), luego el archivo real
_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    _ROOT / "dev" / "modelo.pt"
    if (_ROOT / "dev" / "modelo.pt").exists()
    else _ROOT / "dev" / "YOLOv26m Manual Weighted" / "exp5_yolo26m_manual_weighted.pt"
)

# Clases del dataset según data/raw/data.yaml
CLASES = {
    0: "Ball",
    1: "Corner",
    2: "GoalKeeper",
    3: "Goal_Net",
    4: "Referee",
    5: "TEAM 1",
    6: "TEAM 2",
}

# Colores BGR para dibujar bboxes sin resultado de offside
_COLORES_BGR = {
    0: (255, 255, 255),  # Ball: blanco
    2: (0, 215, 255),    # GoalKeeper: dorado
    4: (0, 165, 255),    # Referee: naranja
    5: (0,   0, 220),    # TEAM 1: rojo
    6: (220, 220,  0),   # TEAM 2: amarillo
}


def cargar_modelo(ruta=None):
    """Carga el modelo YOLOv8 con Ultralytics. Usar con @st.cache_resource en la app."""
    from ultralytics import YOLO
    return YOLO(str(ruta or MODEL_PATH))


def detectar_objetos(modelo, imagen_pil, conf=0.25, iou=0.7, agnostic_nms=True):
    """
    Ejecuta inferencia YOLOv8 sobre una PIL.Image con parámetros NMS personalizados.
    Ultralytics aplica internamente el mismo preprocesamiento que en entrenamiento
    (resize a 640, normalización), así que no hay que reinventarlo.

    Retorna lista de dicts: {class_id, class_name, conf, x1, y1, x2, y2}
    """
    img_np = np.array(imagen_pil.convert("RGB"))
    resultados = modelo.predict(
        source=img_np,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        verbose=False
    )

    detecciones = []
    if resultados and resultados[0].boxes is not None:
        boxes = resultados[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detecciones.append({
                "class_id":   cls_id,
                "class_name": CLASES.get(cls_id, str(cls_id)),
                "conf":       conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return detecciones


def pie_jugador(det):
    """
    Punto de contacto con el césped: centro inferior del bounding box.
    Es la posición que se compara en el algoritmo de offside.
    """
    return ((det["x1"] + det["x2"]) / 2, det["y2"])


def _proyectar_en_base(foot, vp, h):
    """
    Extiende la recta foot→VP hasta y=h (borde inferior de la imagen).
    El valor x resultante sirve para ordenar jugadores por posición relativa
    en el campo, corrigiendo la perspectiva. Funciona incluso si el VP
    cae fuera de la imagen (arriba o a los lados).
    """
    fx, fy = foot
    vp_x, vp_y = vp
    if abs(fy - vp_y) < 1e-6:
        return fx
    t = (h - fy) / (vp_y - fy)
    return fx + t * (vp_x - fx)


def calcular_offside(vp, detecciones, equipo_atacante_id, imagen_shape, gol_a_derecha=None, ref_defender_idx=None):
    """
    Calcula el veredicto de offside geométrico usando el punto de fuga.

    Algoritmo:
      1. Separa jugadores en atacantes y defensores (otro equipo + arquero).
      2. Proyecta el pie de cada defensor hacia el VP para ordenarlos
         a lo largo del eje del campo, corrigiendo perspectiva.
      3. Identifica la dirección del gol a partir de la posición del arquero o la selección manual.
      4. El penúltimo defensor (2do desde el gol) define la línea de offside.
      5. Cada atacante cuya proyección supere la del penúltimo defensor es OFFSIDE.

    Nota sobre la pelota (reglamento): un atacante nunca está en offside si está
    detrás de la pelota. Esa verificación adicional se puede agregar con la posición
    de Ball del mismo dict de detecciones si se desea ser más preciso.

    Retorna dict con: penultimo_defensor, linea_foot, atacantes_resultado,
                      hay_offside, advertencias, equipo_atacante_id, gol_a_derecha.
    """
    h, w = imagen_shape[:2]
    equipo_defensor_id = 6 if equipo_atacante_id == 5 else 5
    advertencias = []

    atacantes  = [d for d in detecciones if d["class_id"] == equipo_atacante_id]
    equipo_def = [d for d in detecciones if d["class_id"] == equipo_defensor_id]
    arqueros   = [d for d in detecciones if d["class_id"] == 2]
    defensores = equipo_def + arqueros

    if not atacantes:
        advertencias.append("No se detectaron jugadores del equipo atacante seleccionado.")
    if not defensores:
        advertencias.append("No se detectaron defensores.")
        return {
            "penultimo_defensor":  None,
            "linea_foot":          None,
            "atacantes_resultado": [(a, False) for a in atacantes],
            "hay_offside":         False,
            "advertencias":        advertencias,
            "equipo_atacante_id":  equipo_atacante_id,
            "gol_a_derecha":       True,
        }
    if not arqueros:
        advertencias.append("No se detectó arquero — la referencia de offside puede ser imprecisa.")

    def proj(det):
        return _proyectar_en_base(pie_jugador(det), vp, h)

    # Determinar en qué lado está el gol
    if gol_a_derecha is None:
        # Comparamos la proyección contra el promedio de todos los jugadores del campo.
        todos = [d for d in detecciones if d["class_id"] in (2, 5, 6)]
        promedio_proj = np.mean([proj(d) for d in todos]) if todos else w / 2.0

        if arqueros:
            gol_a_derecha = bool(proj(arqueros[0]) > promedio_proj)
        else:
            # Sin arquero: estimamos por la posición promedio de los defensores
            gol_a_derecha = bool(np.mean([proj(d) for d in defensores]) > promedio_proj)

    # Ordenar defensores desde el gol hacia el centro del campo
    defensores_ord = sorted(defensores, key=proj, reverse=gol_a_derecha)

    if ref_defender_idx is not None:
        if 0 <= ref_defender_idx < len(defensores_ord):
            penultimo = defensores_ord[ref_defender_idx]
        else:
            penultimo = defensores_ord[-1] if defensores_ord else None
            advertencias.append(f"No hay suficientes defensores para usar el índice {ref_defender_idx + 1}.")
    else:
        # El penúltimo defensor es el 2do desde el gol.
        # Si detectamos arquero, él ocupa el 1er lugar (índice 0), por lo que el penúltimo es el índice 1.
        # Si NO detectamos arquero, asumimos que está en el arco (detrás de los defensores de campo),
        # por lo que el último defensor de campo (índice 0) actúa como el penúltimo general.
        if arqueros:
            if len(defensores_ord) < 2:
                penultimo = defensores_ord[0]
                advertencias.append("Solo un defensor detectado; se usa como referencia de offside.")
            else:
                penultimo = defensores_ord[1]
        else:
            if defensores_ord:
                penultimo = defensores_ord[0]
            else:
                penultimo = None
                advertencias.append("No se detectaron defensores de campo.")

    if penultimo is not None:
        penultimo_proj = proj(penultimo)
        penultimo_foot = pie_jugador(penultimo)
    else:
        penultimo_proj = 0
        penultimo_foot = None

    # Evaluar cada atacante
    atacantes_resultado = []
    for atk in atacantes:
        atk_proj  = proj(atk)
        en_offside = (atk_proj > penultimo_proj) if gol_a_derecha else (atk_proj < penultimo_proj)
        atacantes_resultado.append((atk, en_offside))

    hay_offside = any(os for _, os in atacantes_resultado)

    return {
        "penultimo_defensor":  penultimo,
        "linea_foot":          penultimo_foot,
        "atacantes_resultado": atacantes_resultado,
        "hay_offside":         hay_offside,
        "advertencias":        advertencias,
        "equipo_atacante_id":  equipo_atacante_id,
        "gol_a_derecha":       gol_a_derecha,
    }


def _puntos_en_borde(fx, fy, vp_x, vp_y, w, h):
    """
    Devuelve hasta dos puntos donde la recta (fx,fy)→(vp_x,vp_y)
    cruza los bordes del rectángulo [0,w] × [0,h].
    """
    dx, dy = vp_x - fx, vp_y - fy
    puntos = set()

    candidatos = [
        (0 - fy,  dy, lambda t: (fx + t * dx, 0.0)),    # borde superior
        (h - fy,  dy, lambda t: (fx + t * dx, float(h))),  # borde inferior
        (0 - fx,  dx, lambda t: (0.0, fy + t * dy)),    # borde izquierdo
        (w - fx,  dx, lambda t: (float(w), fy + t * dy)),  # borde derecho
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
    """
    Dibuja bboxes, línea de offside, punto de fuga y etiquetas sobre la imagen.
    Si resultado_offside es None, solo dibuja las detecciones.
    Retorna PIL.Image anotada.
    """
    img = cv2.cvtColor(np.array(imagen_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    equipo_atk_id = resultado_offside.get("equipo_atacante_id") if resultado_offside else None

    # Mapa de clave → en_offside para los atacantes detectados
    atk_offside_map = {}
    if resultado_offside:
        for det, en_offside in resultado_offside.get("atacantes_resultado", []):
            atk_offside_map[(round(det["x1"]), round(det["y1"]))] = en_offside

    # --- Bboxes ---
    for det in detecciones:
        cls_id = det["class_id"]
        if cls_id not in _COLORES_BGR:
            continue
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        key = (round(det["x1"]), round(det["y1"]))

        if cls_id == equipo_atk_id and key in atk_offside_map:
            en_offside = atk_offside_map[key]
            color  = (0, 0, 220) if en_offside else (0, 200, 50)
            grosor = 3
            label  = "OFFSIDE" if en_offside else "ONSIDE"
        else:
            color  = _COLORES_BGR[cls_id]
            grosor = 2
            label  = det["class_name"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # --- Línea de offside ---
    if resultado_offside and resultado_offside.get("linea_foot") and vp:
        fx, fy = resultado_offside["linea_foot"]
        pts_borde = _puntos_en_borde(fx, fy, vp[0], vp[1], w, h)
        if len(pts_borde) >= 2:
            pt1 = (int(round(pts_borde[0][0])), int(round(pts_borde[0][1])))
            pt2 = (int(round(pts_borde[1][0])), int(round(pts_borde[1][1])))
            cv2.line(img, pt1, pt2, (0, 140, 255), 2, cv2.LINE_AA)
            label_x = min(pt1[0], pt2[0]) + 5
            label_y = min(pt1[1], pt2[1]) - 6
            cv2.putText(img, "Linea offside", (label_x, max(label_y, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
        # Marcar el pie del penúltimo defensor
        cv2.circle(img, (int(fx), int(fy)), 8, (0, 140, 255), -1, cv2.LINE_AA)

    # --- Punto de fuga (solo si cae dentro de la imagen) ---
    if vp:
        vx, vy = int(round(vp[0])), int(round(vp[1]))
        if 0 <= vx < w and 0 <= vy < h:
            cv2.circle(img, (vx, vy), 10, (0, 0, 180), -1, cv2.LINE_AA)
            cv2.circle(img, (vx, vy), 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"VP ({vp[0]:.0f},{vp[1]:.0f})",
                        (vx + 14, max(vy - 14, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def imagen_pil_a_bgr(imagen_pil):
    """Convierte PIL.Image a numpy array BGR para funciones de OpenCV/HoughLines."""
    return cv2.cvtColor(np.array(imagen_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
