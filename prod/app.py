"""
App Streamlit — Detección de Offside en Fútbol
Semana 4 del TP Integrador de Redes Neuronales Profundas.

Flujo:
  1. Usuario sube imagen → detección con YOLOv8
  2. Cálculo del punto de fuga (automático o manual con 4 puntos)
  3. Selección del equipo atacante
  4. Algoritmo geométrico de offside → imagen anotada + veredicto
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import streamlit as st
from PIL import Image

# Agregar raíz del repo al path (necesario cuando Streamlit corre desde prod/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    cargar_modelo,
    detectar_objetos,
    calcular_offside,
    dibujar_resultado,
    imagen_pil_a_bgr,
    detectar_punto_de_fuga,
    punto_de_fuga_desde_puntos,
)

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Detección de Offside",
    page_icon="⚽",
    layout="wide",
)
st.title("⚽ Detección de Offside en Fútbol")
st.caption("TP Integrador — Redes Neuronales Profundas")

# ---------------------------------------------------------------------------
# Carga del modelo (cacheada para no recargar en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_modelo():
    return cargar_modelo()

with st.spinner("Cargando modelo YOLOv8..."):
    modelo = get_modelo()

# ---------------------------------------------------------------------------
# Paso 1: Subida de imagen y configuración de parámetros
# ---------------------------------------------------------------------------
st.header("1. Subí una imagen del partido")
imagen_subida = st.file_uploader(
    "Formatos aceptados: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
)

if imagen_subida is None:
    st.info("Subí una imagen para comenzar el análisis.")
    st.stop()

# Parámetros de detección de YOLO
col_conf, col_iou, col_agnostic = st.columns(3)
with col_conf:
    conf = st.slider(
        "Umbral de confianza (Confidence)",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Confianza mínima requerida para registrar una detección."
    )
with col_iou:
    iou = st.slider(
        "Umbral de IoU (NMS)",
        min_value=0.10,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Superposición máxima permitida entre cajas de detección."
    )
with col_agnostic:
    st.write("") # Espaciadores verticales para centrado
    st.write("") 
    agnostic_nms = st.checkbox(
        "NMS Agnóstico de Clase",
        value=True,
        help="Evita que se dibujen cajas duplicadas de distintos equipos sobre un mismo jugador."
    )

imagen_pil = Image.open(imagen_subida).convert("RGB")
ancho_orig, alto_orig = imagen_pil.size

# ---------------------------------------------------------------------------
# Paso 2: Detección con YOLOv8
# Cacheamos en session_state para evitar redetectar si los parámetros o la imagen no cambiaron.
# ---------------------------------------------------------------------------
if (
    st.session_state.get("last_filename") != imagen_subida.name
    or st.session_state.get("last_conf") != conf
    or st.session_state.get("last_iou") != iou
    or st.session_state.get("last_agnostic") != agnostic_nms
):
    st.session_state.last_filename  = imagen_subida.name
    st.session_state.last_conf      = conf
    st.session_state.last_iou       = iou
    st.session_state.last_agnostic  = agnostic_nms
    st.session_state.detecciones    = None
    st.session_state.vp_manual      = None
    st.session_state.resultado      = None

if st.session_state.detecciones is None:
    with st.spinner("Detectando jugadores y objetos..."):
        st.session_state.detecciones = detectar_objetos(
            modelo,
            imagen_pil,
            conf=conf,
            iou=iou,
            agnostic_nms=agnostic_nms
        )

detecciones = st.session_state.detecciones

# Mostrar imagen con detecciones y resumen
st.header("2. Detecciones del modelo")
col_img, col_info = st.columns([3, 1])

with col_img:
    img_det = dibujar_resultado(imagen_pil, detecciones, vp=None, resultado_offside=None)
    st.image(img_det, caption="Bounding boxes detectados", width="stretch")

with col_info:
    st.subheader("Objetos detectados")
    conteo = Counter(d["class_name"] for d in detecciones)

    if not conteo:
        st.error("No se detectó ningún objeto en la imagen.")
    else:
        for nombre, n in sorted(conteo.items()):
            st.write(f"- **{nombre}**: {n}")

    # Advertencias de casos límite
    if conteo.get("TEAM 1", 0) == 0 and conteo.get("TEAM 2", 0) == 0:
        st.error("No se detectaron jugadores de ningún equipo. El análisis de offside no es posible.")
    elif conteo.get("TEAM 1", 0) == 0 or conteo.get("TEAM 2", 0) == 0:
        st.warning("Solo se detectó un equipo. El resultado de offside puede ser poco confiable.")
    if conteo.get("GoalKeeper", 0) == 0:
        st.warning("No se detectó arquero. La línea de offside puede ser imprecisa.")

# ---------------------------------------------------------------------------
# Paso 3: Punto de fuga
# ---------------------------------------------------------------------------
st.header("3. Punto de fuga (VP)")

# Intentar detección automática con Hough
imagen_bgr = imagen_pil_a_bgr(imagen_pil)
vp_auto, _, _ = detectar_punto_de_fuga(imagen_bgr)

if vp_auto:
    st.success(f"Punto de fuga detectado automáticamente: ({vp_auto[0]:.0f}, {vp_auto[1]:.0f})")
    usar_manual = st.checkbox("Quiero ajustar el VP manualmente", value=False)
else:
    st.warning(
        "No se pudo detectar el punto de fuga automáticamente "
        "(común en imágenes de broadcast). Marcá 4 puntos manualmente."
    )
    usar_manual = True

vp = vp_auto  # puede ser sobreescrito por el modo manual

if usar_manual:
    st.info(
        "**Instrucciones:** hacé clic en 4 puntos sobre la imagen. "
        "Los puntos **1 y 2** deben estar sobre una línea del campo (ej: línea lateral). "
        "Los puntos **3 y 4** sobre otra línea paralela (ej: la línea del área). "
        "El VP se calcula como intersección de esas dos rectas."
    )

    # Intentar usar el canvas interactivo
    _canvas_disponible = False
    try:
        from streamlit_drawable_canvas import st_canvas
        _canvas_disponible = True
    except ImportError:
        pass

    if _canvas_disponible:
        DISPLAY_W = min(700, ancho_orig)
        DISPLAY_H = int(DISPLAY_W * alto_orig / ancho_orig)

        st.write("**Hacé clic sobre la imagen** para marcar los 4 puntos (aparecen como círculos rojos):")
        canvas_result = st_canvas(
            background_image=imagen_pil,
            drawing_mode="point",
            point_display_radius=7,
            stroke_color="#FF3333",
            fill_color="#FF3333",
            width=DISPLAY_W,
            height=DISPLAY_H,
            key="canvas_vp",
        )

        if canvas_result.json_data:
            objetos = canvas_result.json_data.get("objects", [])
            n_pts = len(objetos)
            st.write(f"Puntos marcados: **{n_pts}/4**")

            if n_pts >= 4:
                # Los objetos son círculos pequeños; el centro es (left + radius, top + radius)
                scale_x = ancho_orig / DISPLAY_W
                scale_y = alto_orig / DISPLAY_H
                pts = []
                for obj in objetos[:4]:
                    r  = obj.get("radius", 0)
                    cx = (obj["left"] + r) * scale_x
                    cy = (obj["top"]  + r) * scale_y
                    pts.append((cx, cy))

                vp_calculado = punto_de_fuga_desde_puntos(pts)
                if vp_calculado:
                    st.session_state.vp_manual = vp_calculado
                    st.success(
                        f"VP calculado desde puntos manuales: "
                        f"({vp_calculado[0]:.0f}, {vp_calculado[1]:.0f})"
                    )
                else:
                    st.error(
                        "Las dos líneas que trazaste son paralelas. "
                        "Elegí puntos sobre líneas que converjan (no líneas paralelas entre sí)."
                    )

    else:
        # Fallback: 4 pares de inputs numéricos
        st.write("Ingresá las coordenadas en píxeles de los 4 puntos (2 por línea del campo):")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p1x = st.number_input("Punto 1 X", min_value=0, max_value=ancho_orig, value=0, key="p1x")
            p1y = st.number_input("Punto 1 Y", min_value=0, max_value=alto_orig,  value=0, key="p1y")
        with c2:
            p2x = st.number_input("Punto 2 X", min_value=0, max_value=ancho_orig, value=0, key="p2x")
            p2y = st.number_input("Punto 2 Y", min_value=0, max_value=alto_orig,  value=0, key="p2y")
        with c3:
            p3x = st.number_input("Punto 3 X", min_value=0, max_value=ancho_orig, value=0, key="p3x")
            p3y = st.number_input("Punto 3 Y", min_value=0, max_value=alto_orig,  value=0, key="p3y")
        with c4:
            p4x = st.number_input("Punto 4 X", min_value=0, max_value=ancho_orig, value=0, key="p4x")
            p4y = st.number_input("Punto 4 Y", min_value=0, max_value=alto_orig,  value=0, key="p4y")

        if st.button("Calcular VP desde estos 4 puntos"):
            pts = [(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)]
            vp_calculado = punto_de_fuga_desde_puntos(pts)
            if vp_calculado:
                st.session_state.vp_manual = vp_calculado
                st.success(f"VP calculado: ({vp_calculado[0]:.0f}, {vp_calculado[1]:.0f})")
            else:
                st.error("Las líneas son paralelas. Elegí puntos diferentes.")

    # Usar el VP manual si está disponible
    if st.session_state.get("vp_manual"):
        vp = st.session_state.vp_manual

if vp is None:
    st.info("Marcá el punto de fuga para continuar con el análisis.")
    st.stop()

# ---------------------------------------------------------------------------
# Paso 4: Selección del equipo atacante
# ---------------------------------------------------------------------------
st.header("4. Equipo atacante")

# Sugerencia por proximidad del equipo a la pelota
pelotas  = [d for d in detecciones if d["class_id"] == 0]
team1    = [d for d in detecciones if d["class_id"] == 5]
team2    = [d for d in detecciones if d["class_id"] == 6]
sugerencia = None

if pelotas and (team1 or team2):
    pelota = pelotas[0]
    px = (pelota["x1"] + pelota["x2"]) / 2
    py = (pelota["y1"] + pelota["y2"]) / 2

    def dist_min(equipo):
        if not equipo:
            return float("inf")
        return min(
            ((d["x1"] + d["x2"]) / 2 - px) ** 2 + ((d["y1"] + d["y2"]) / 2 - py) ** 2
            for d in equipo
        )

    sugerencia = "TEAM 1" if dist_min(team1) <= dist_min(team2) else "TEAM 2"
    st.info(f"Sugerencia: **{sugerencia}** parece estar más cerca de la pelota.")

opciones    = ["TEAM 1", "TEAM 2"]
default_idx = opciones.index(sugerencia) if sugerencia in opciones else 0
equipo_atacante    = st.radio("¿Qué equipo está atacando?", opciones, index=default_idx, horizontal=True)
equipo_atacante_id = 5 if equipo_atacante == "TEAM 1" else 6

# Controles avanzados para ajustar dirección y defensor de referencia
with st.expander("Ajustes avanzados de Offside (Dirección y Referencia)"):
    col_dir, col_ref = st.columns(2)
    with col_dir:
        dir_ataque = st.radio(
            "Dirección del ataque:",
            ["Detectar automáticamente 🔄", "Ataca hacia la derecha ➡️", "Ataca hacia la izquierda ⬅️"],
            index=0,
            help="Fuerza la dirección en la que el equipo seleccionado está atacando."
        )
    with col_ref:
        ref_defensor = st.selectbox(
            "Jugador de referencia para la línea:",
            [
                "Detectar automáticamente 🔄",
                "1er jugador más cercano al arco (Último)",
                "2do jugador más cercano al arco (Penúltimo)",
                "3er jugador más cercano al arco"
            ],
            index=0,
            help="Elige explícitamente cuál defensor (ordenados desde su propia línea de meta) define la línea de offside."
        )

# ---------------------------------------------------------------------------
# Paso 5: Cálculo de offside
# ---------------------------------------------------------------------------
st.header("5. Resultado de offside")

if st.button("Calcular offside", type="primary", use_container_width=True):
    # Mapeo de dirección del ataque
    gol_a_derecha_val = None
    if dir_ataque == "Ataca hacia la derecha ➡️":
        gol_a_derecha_val = True
    elif dir_ataque == "Ataca hacia la izquierda ⬅️":
        gol_a_derecha_val = False

    # Mapeo del defensor de referencia
    ref_defender_idx_val = None
    if ref_defensor == "1er jugador más cercano al arco (Último)":
        ref_defender_idx_val = 0
    elif ref_defensor == "2do jugador más cercano al arco (Penúltimo)":
        ref_defender_idx_val = 1
    elif ref_defensor == "3er jugador más cercano al arco":
        ref_defender_idx_val = 2

    resultado = calcular_offside(
        vp,
        detecciones,
        equipo_atacante_id,
        np.array(imagen_pil).shape,
        gol_a_derecha=gol_a_derecha_val,
        ref_defender_idx=ref_defender_idx_val,
    )
    st.session_state.resultado             = resultado
    st.session_state.vp_para_resultado     = vp
    st.session_state.equipo_para_resultado = equipo_atacante_id
    st.session_state.last_dir_ataque       = dir_ataque
    st.session_state.last_ref_defensor     = ref_defensor

# Mostrar resultado si está disponible (y corresponde a la imagen y configuración actuales)
if (
    st.session_state.get("resultado") is not None
    and st.session_state.get("last_filename") == imagen_subida.name
    and st.session_state.get("equipo_para_resultado") == equipo_atacante_id
    and st.session_state.get("last_dir_ataque") == dir_ataque
    and st.session_state.get("last_ref_defensor") == ref_defensor
):
    resultado  = st.session_state.resultado
    vp_final   = st.session_state.vp_para_resultado

    # Advertencias del algoritmo
    for adv in resultado.get("advertencias", []):
        st.warning(adv)

    # Imagen anotada con línea de offside
    img_final = dibujar_resultado(imagen_pil, detecciones, vp_final, resultado)

    col_res, col_det = st.columns([3, 1])
    with col_res:
        st.image(img_final, caption="Imagen con línea de offside anotada", width="stretch")

    with col_det:
        # Veredicto global
        if resultado["hay_offside"]:
            st.error("🚩 OFFSIDE")
        else:
            st.success("✅ ONSIDE (posición legal)")

        st.divider()

        # Detalle por atacante
        st.subheader("Detalle por jugador")
        for det, en_offside in resultado.get("atacantes_resultado", []):
            estado = "🔴 OFFSIDE" if en_offside else "🟢 ONSIDE"
            st.write(f"{det['class_name']} (conf {det['conf']:.0%}): {estado}")

        if not resultado.get("atacantes_resultado"):
            st.write("No hay atacantes para evaluar.")
