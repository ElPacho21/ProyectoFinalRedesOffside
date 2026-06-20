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
import os
import base64
import tempfile
from pathlib import Path
from collections import Counter

import cv2
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
st.header("1. Subí una imagen o video del partido")
imagen_subida = st.file_uploader(
    "Formatos aceptados: JPG, JPEG, PNG, MP4, AVI, MOV, MKV",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv", "m4v"],
)

if imagen_subida is None:
    st.info("Subí una imagen o video para comenzar el análisis.")
    st.stop()

# --- Soporte de video ---
_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "m4v"}
_file_ext = imagen_subida.name.rsplit(".", 1)[-1].lower()
is_video = _file_ext in _VIDEO_EXTS
frame_idx = 0

if is_video:
    # Guardar en archivo temporal (OpenCV necesita path, no BytesIO)
    _vid_key = (imagen_subida.name, imagen_subida.size)
    if st.session_state.get("_temp_vid_key") != _vid_key:
        _tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"offside_{abs(hash(imagen_subida.name)) % 10**9}.{_file_ext}"
        )
        imagen_subida.seek(0)
        with open(_tmp_path, "wb") as _f:
            _f.write(imagen_subida.read())
        st.session_state._temp_vid_path = _tmp_path
        st.session_state._temp_vid_key  = _vid_key
        st.session_state.frame_idx      = 0   # reset al cargar nuevo video

    _cap = cv2.VideoCapture(st.session_state._temp_vid_path)
    _total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _fps          = _cap.get(cv2.CAP_PROP_FPS) or 25.0
    _cap.release()
    _max_frame = max(0, _total_frames - 1)

    # Botones ◀ ▶ + slider enlazado a session state
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    def _prev_frame():
        st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)

    def _next_frame():
        st.session_state.frame_idx = min(_max_frame, st.session_state.frame_idx + 1)

    frame_idx = st.session_state.frame_idx

    # Extraer frame primero para mostrar preview
    _cap = cv2.VideoCapture(st.session_state._temp_vid_path)
    _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    _ret, _frame_bgr = _cap.read()
    _cap.release()
    if not _ret:
        st.error("No se pudo leer el frame seleccionado.")
        st.stop()
    imagen_pil = Image.fromarray(cv2.cvtColor(_frame_bgr, cv2.COLOR_BGR2RGB))

    # Preview primero
    st.image(imagen_pil, use_container_width=True)

    # Controles debajo de la preview
    _col_prev, _col_slider, _col_next = st.columns([1, 20, 1])
    with _col_prev:
        st.button("◀", on_click=_prev_frame, use_container_width=True)
    with _col_next:
        st.button("▶", on_click=_next_frame, use_container_width=True)
    with _col_slider:
        frame_idx = st.slider(
            "Frame",
            0, _max_frame,
            key="frame_idx",
            label_visibility="collapsed",
        )

    _ts = frame_idx / _fps
    st.caption(
        f"**{_total_frames}** frames · **{_fps:.1f}** FPS · "
        f"Frame **{frame_idx}** / {_max_frame} · "
        f"{int(_ts // 60):02d}:{_ts % 60:05.2f}s"
    )
else:
    imagen_pil = Image.open(imagen_subida).convert("RGB")

ancho_orig, alto_orig = imagen_pil.size
_cache_key = f"{imagen_subida.name}::f{frame_idx}" if is_video else imagen_subida.name

# Parámetros de detección de YOLO
conf = st.slider(
    "Umbral de confianza (Confidence)",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Confianza mínima requerida para registrar una detección."
)

# ---------------------------------------------------------------------------
# Paso 2: Detección con YOLOv8
# Cacheamos en session_state para evitar redetectar si los parámetros o la imagen no cambiaron.
# ---------------------------------------------------------------------------
if (
    st.session_state.get("last_cache_key") != _cache_key
    or st.session_state.get("last_conf") != conf
):
    _filename_changed = st.session_state.get("last_filename") != imagen_subida.name
    st.session_state.last_cache_key = _cache_key
    st.session_state.last_filename  = imagen_subida.name
    st.session_state.last_conf      = conf
    st.session_state.detecciones    = None
    st.session_state.resultado      = None
    if _filename_changed:
        st.session_state.vp_manual      = None
        st.session_state.canvas_vp_reset = st.session_state.get("canvas_vp_reset", 0) + 1

if st.session_state.detecciones is None:
    with st.spinner("Detectando jugadores y objetos..."):
        st.session_state.detecciones = detectar_objetos(
            modelo,
            imagen_pil,
            conf=conf,
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

st.info(
    "**Instrucciones:** hacé clic en 4 puntos sobre la imagen. "
    "Los puntos **1 y 2** deben estar sobre una línea del campo (ej: línea lateral). "
    "Los puntos **3 y 4** sobre otra línea paralela (ej: la línea del área). "
    "El VP se calcula como intersección de esas dos rectas."
)

vp = None

# Intentar usar el canvas interactivo
_canvas_disponible = False
try:
    from streamlit_drawable_canvas import st_canvas
    _canvas_disponible = True
except ImportError:
    pass

if _canvas_disponible:
    DISPLAY_W = min(1100, ancho_orig)
    DISPLAY_H = int(DISPLAY_W * alto_orig / ancho_orig)

    # El canvas vive en un iframe (componente custom de Streamlit), así que CSS del padre
    # no llega. Inyectamos JS que accede al iframe por same-origin y aplica cursor X.
    _x_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
        '<line x1="2" y1="2" x2="18" y2="18" stroke="white" stroke-width="4" stroke-linecap="round"/>'
        '<line x1="18" y1="2" x2="2" y2="18" stroke="white" stroke-width="4" stroke-linecap="round"/>'
        '<line x1="2" y1="2" x2="18" y2="18" stroke="black" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="18" y1="2" x2="2" y2="18" stroke="black" stroke-width="2" stroke-linecap="round"/>'
        '</svg>'
    )
    _cursor_url = (
        "url('data:image/svg+xml;base64,"
        + base64.b64encode(_x_svg.encode()).decode()
        + "') 10 10, crosshair"
    )
    st.markdown(
        f"""
        <script>
        (function() {{
            if (window._vpXCursorInterval) return;
            var cursorUrl = "{_cursor_url}";
            window._vpXCursorInterval = setInterval(function() {{
                document.querySelectorAll('iframe').forEach(function(iframe) {{
                    try {{
                        var doc = iframe.contentDocument || iframe.contentWindow.document;
                        doc.querySelectorAll('canvas').forEach(function(c) {{
                            c.style.setProperty('cursor', cursorUrl, 'important');
                        }});
                    }} catch(e) {{}}
                }});
            }}, 200);
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    if "canvas_vp_reset" not in st.session_state:
        st.session_state.canvas_vp_reset = 0

    st.write("**Hacé clic sobre la imagen** para marcar los 4 puntos (aparecen como círculos rojos):")
    _bg_canvas = imagen_pil.resize((DISPLAY_W, DISPLAY_H))
    canvas_result = st_canvas(
        background_image=_bg_canvas,
        drawing_mode="point",
        point_display_radius=3,
        stroke_color="#FF3333",
        fill_color="#FF3333",
        width=DISPLAY_W,
        height=DISPLAY_H,
        key=f"canvas_vp_{st.session_state.canvas_vp_reset}",
    )

    if canvas_result.json_data:
        objetos = canvas_result.json_data.get("objects", [])
        n_pts = len(objetos)

        if n_pts > 4:
            st.error(f"Máximo 4 puntos permitidos. Marcaste {n_pts}.")
            if st.button("Limpiar canvas"):
                st.session_state.canvas_vp_reset += 1
                st.rerun()
        else:
            st.write(f"Puntos marcados: **{n_pts}/4**")

            if n_pts == 4:
                # Los objetos son círculos pequeños; el centro es (left + radius, top + radius)
                scale_x = ancho_orig / DISPLAY_W
                scale_y = alto_orig / DISPLAY_H
                pts = []
                for obj in objetos:
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
    col_dir, col_ref, col_pie = st.columns(3)
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
    with col_pie:
        ref_pie = st.selectbox(
            "Punto de referencia del jugador:",
            [
                "Punto medio inferior (Centro) ⏺️",
                "Punto izquierdo inferior (Izquierda) ◀️",
                "Punto derecho inferior (Derecha) ▶️"
            ],
            index=0,
            help="Determina qué parte del bounding box del jugador se usa para calcular su posición (útil según la pose del cuerpo)."
        )

# ---------------------------------------------------------------------------
# Paso 5: Cálculo de offside
# ---------------------------------------------------------------------------
st.header("5. Resultado de offside")

# Para videos con VP ya conocido, auto-calcular al cambiar de frame
_btn_calc = st.button("Calcular offside", type="primary", use_container_width=True)
_auto_calc = is_video and vp is not None and st.session_state.get("resultado") is None

if _btn_calc or _auto_calc:
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

    # Mapeo del punto de referencia (pie)
    punto_referencia_val = "medio"
    if ref_pie == "Punto izquierdo inferior (Izquierda) ◀️":
        punto_referencia_val = "izquierdo"
    elif ref_pie == "Punto derecho inferior (Derecha) ▶️":
        punto_referencia_val = "derecho"

    resultado = calcular_offside(
        vp,
        detecciones,
        equipo_atacante_id,
        np.array(imagen_pil).shape,
        gol_a_derecha=gol_a_derecha_val,
        ref_defender_idx=ref_defender_idx_val,
        punto_referencia=punto_referencia_val,
    )
    st.session_state.resultado             = resultado
    st.session_state.vp_para_resultado     = vp
    st.session_state.equipo_para_resultado = equipo_atacante_id
    st.session_state.last_dir_ataque       = dir_ataque
    st.session_state.last_ref_defensor     = ref_defensor
    st.session_state.last_ref_pie          = ref_pie

# Mostrar resultado si está disponible (y corresponde al frame/imagen y configuración actuales)
if (
    st.session_state.get("resultado") is not None
    and st.session_state.get("last_cache_key") == _cache_key
    and st.session_state.get("equipo_para_resultado") == equipo_atacante_id
    and st.session_state.get("last_dir_ataque") == dir_ataque
    and st.session_state.get("last_ref_defensor") == ref_defensor
    and st.session_state.get("last_ref_pie") == ref_pie
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
