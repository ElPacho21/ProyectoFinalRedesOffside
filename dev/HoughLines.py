import cv2
import numpy as np


def _interseccion(seg_a, seg_b):
    """Intersección entre dos segmentos tratados como líneas infinitas.
    Retorna (x, y) o None si son paralelas o casi paralelas."""
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-6:
        return None
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    return (x1 + t * dx1, y1 + t * dy1)


def detectar_punto_de_fuga(imagen_bgr):
    """Detecta el punto de fuga de las líneas blancas del campo de fútbol.

    Retorna (vp, grupo_a, grupo_b):
        vp      — (x, y) punto de fuga como floats
        grupo_a — lista de (x1,y1,x2,y2) con ángulo positivo
        grupo_b — lista de (x1,y1,x2,y2) con ángulo negativo
    Retorna (None, [], []) si no hay suficientes líneas.
    """
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    mask_verde = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    mask_blanca = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    mask_final = cv2.bitwise_and(mask_blanca, mask_verde)

    edges = cv2.Canny(mask_final, 50, 150)

    lineas = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=80,
                             minLineLength=60,
                             maxLineGap=20)
    if lineas is None:
        return None, [], []

    # Filtrar líneas casi horizontales (<10°) o casi verticales (>80°)
    # — esas no contribuyen a un punto de fuga útil en el horizonte
    segs, angulos = [], []
    for l in lineas:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        abs_ang = abs(ang)
        if abs_ang < 10 or abs_ang > 80:
            continue
        segs.append((x1, y1, x2, y2))
        angulos.append(ang)

    grupo_a = [s for s, a in zip(segs, angulos) if a > 0]  # pendiente positiva en imagen
    grupo_b = [s for s, a in zip(segs, angulos) if a < 0]  # pendiente negativa

    if not grupo_a or not grupo_b:
        return None, grupo_a, grupo_b

    h, w = imagen_bgr.shape[:2]
    intersecciones = []
    for sa in grupo_a:
        for sb in grupo_b:
            pt = _interseccion(sa, sb)
            if pt is None:
                continue
            # Descartar intersecciones numéricamente explosivas (>10× dimensiones)
            if abs(pt[0]) > 10 * w or abs(pt[1]) > 10 * h:
                continue
            intersecciones.append(pt)

    if not intersecciones:
        return None, grupo_a, grupo_b

    vp = tuple(np.median(intersecciones, axis=0))
    return vp, grupo_a, grupo_b


def visualizar_punto_de_fuga(imagen_bgr, max_lineas=15):
    """Dibuja el punto de fuga y las líneas que convergen en él.

    Verde  — líneas grupo A (pendiente positiva)
    Amarillo — líneas grupo B (pendiente negativa)
    Blanco semitransparente — prolongaciones hasta el VP
    Rojo   — punto de fuga

    Retorna la imagen anotada (BGR), o None si no se detectó VP.
    """
    vp, grupo_a, grupo_b = detectar_punto_de_fuga(imagen_bgr)
    if vp is None:
        return None

    out = imagen_bgr.copy()
    overlay = out.copy()
    vp_int = (int(round(vp[0])), int(round(vp[1])))

    COLOR_A = (0, 210, 0)       # verde
    COLOR_B = (0, 210, 210)     # amarillo

    def draw_line_and_extension(seg, color):
        x1, y1, x2, y2 = seg
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))

        # Segmento detectado — grueso y sólido
        cv2.line(out, pt1, pt2, color, 2, cv2.LINE_AA)

        # Prolongación desde el extremo más cercano al VP hasta el propio VP
        d1 = (x1 - vp[0]) ** 2 + (y1 - vp[1]) ** 2
        d2 = (x2 - vp[0]) ** 2 + (y2 - vp[1]) ** 2
        nearest = pt1 if d1 < d2 else pt2
        cv2.line(overlay, nearest, vp_int, color, 1, cv2.LINE_AA)

    for seg in grupo_a[:max_lineas]:
        draw_line_and_extension(seg, COLOR_A)
    for seg in grupo_b[:max_lineas]:
        draw_line_and_extension(seg, COLOR_B)

    # Mezclar overlay (prolongaciones) semitransparente sobre out
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    # Punto de fuga: relleno rojo + borde blanco + cruz
    cv2.circle(out, vp_int, 10, (0, 0, 220), -1, cv2.LINE_AA)
    cv2.circle(out, vp_int, 12, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(out, vp_int, (255, 255, 255),
                   cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)

    # Etiqueta de coordenadas (borde negro + texto blanco)
    h, w = out.shape[:2]
    label = f"VP ({vp[0]:.0f}, {vp[1]:.0f})"
    lx = int(np.clip(vp_int[0] + 16, 0, w - 200))
    ly = int(np.clip(vp_int[1] - 16, 20, h - 10))
    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)

    return out



def punto_de_fuga_desde_puntos(pts):
    """Calcula el punto de fuga a partir de 4 puntos: los dos primeros
    definen la línea A, los dos últimos la línea B.
    Retorna (x, y) o None si las líneas son paralelas."""
    (p1, p2, p3, p4) = pts
    return _interseccion((p1[0], p1[1], p2[0], p2[1]),
                         (p3[0], p3[1], p4[0], p4[1]))


def _dibujar_vp_manual(ax, pts, vp, w, h):
    """Dibuja sobre `ax` las dos líneas, sus prolongaciones y el VP.
    Amplía los límites del eje si el VP cae fuera de la imagen."""

    def linea_extendida(p1, p2, color):
        # Segmento clickeado sólido + prolongación punteada hasta el VP
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2.5)
        d1 = (p1[0] - vp[0]) ** 2 + (p1[1] - vp[1]) ** 2
        d2 = (p2[0] - vp[0]) ** 2 + (p2[1] - vp[1]) ** 2
        cercano = p1 if d1 < d2 else p2
        ax.plot([cercano[0], vp[0]], [cercano[1], vp[1]],
                color=color, lw=1.2, ls="--", alpha=0.8)

    linea_extendida(pts[0], pts[1], "lime")
    linea_extendida(pts[2], pts[3], "gold")

    xs, ys = zip(*pts)
    ax.scatter(xs, ys, c=["lime", "lime", "gold", "gold"],
               s=60, edgecolors="black", zorder=5)
    ax.scatter([vp[0]], [vp[1]], c="red", s=160, marker="X",
               edgecolors="white", linewidths=1.5, zorder=6)
    ax.annotate(f"VP ({vp[0]:.0f}, {vp[1]:.0f})",
                xy=vp, xytext=(vp[0] + 15, vp[1] - 15),
                color="white", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.85))

    # Si el VP queda fuera de la imagen, ampliar la vista para mostrarlo
    margen = 40
    ax.set_xlim(min(0, vp[0] - margen), max(w, vp[0] + margen))
    ax.set_ylim(max(h, vp[1] + margen), min(0, vp[1] - margen))  # eje y invertido


def seleccionar_punto_de_fuga_manual(imagen_bgr, guardar_en=None):
    """Interfaz interactiva: el usuario hace 4 clics sobre la imagen
    (2 puntos de una línea del campo, 2 de otra paralela en la cancha)
    y se calcula el punto de fuga como intersección de ambas.

    Click derecho deshace el último punto. Retorna (x, y) o None."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    h, w = imagen_bgr.shape[:2]
    img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title("Hacé 4 clics — puntos 1-2: línea A (verde) · puntos 3-4: línea B (amarilla)\n"
                 "Click derecho = deshacer último punto")
    ax.axis("off")

    pts = plt.ginput(4, timeout=0)
    if len(pts) < 4:
        plt.close(fig)
        print("Selección cancelada: se necesitan 4 puntos.")
        return None

    vp = punto_de_fuga_desde_puntos(pts)
    if vp is None:
        plt.close(fig)
        print("Las dos líneas son paralelas en la imagen — no hay punto de fuga.")
        return None

    _dibujar_vp_manual(ax, pts, vp, w, h)
    ax.set_title(f"Punto de fuga: ({vp[0]:.1f}, {vp[1]:.1f}) — cerrá la ventana para terminar")
    fig.canvas.draw()

    if guardar_en:
        fig.savefig(guardar_en, dpi=110, bbox_inches="tight")
        print(f"Guardado: {guardar_en}")

    plt.show()
    return vp


# Clases del dataset (data/raw/data.yaml) — solo las que participan del offside
CLASES_JUGADOR = {
    2: ("GoalKeeper", "deepskyblue"),
    5: ("TEAM 1",     "red"),
    6: ("TEAM 2",     "white"),
}


def cargar_jugadores_de_labels(labels_path, w, h):
    """Lee un .txt YOLO del dataset y devuelve los jugadores como
    [(class_id, x1, y1, x2, y2), ...] en píxeles. Ignora pelota, árbitros, etc."""
    jugadores = []
    from pathlib import Path
    labels_path = Path(labels_path)
    if not labels_path.exists():
        print(f"ADVERTENCIA: labels no encontrado: {labels_path}")
        return jugadores
    for line in labels_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls not in CLASES_JUGADOR:
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        jugadores.append((cls, x1, y1, x2, y2))
    return jugadores


def graficar_jugadores_y_vp(imagen_bgr, labels_path, vp, guardar_en=None, mostrar=True):
    """Dibuja los bboxes de los jugadores (del dataset) y una línea desde los
    pies de cada uno hacia el punto de fuga. La línea pies→VP es la proyección
    en imagen de la línea de offside (paralela a la línea de gol en la cancha).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    h, w = imagen_bgr.shape[:2]
    jugadores = cargar_jugadores_de_labels(labels_path, w, h)
    if not jugadores:
        print("No hay jugadores en las labels — nada que dibujar.")
        return None

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    vistos = set()
    for cls, x1, y1, x2, y2 in jugadores:
        nombre, color = CLASES_JUGADOR[cls]

        # Bbox del jugador
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        # Punto de pies = centro inferior del bbox (lo que toca el césped)
        pie = ((x1 + x2) / 2, y2)
        ax.scatter([pie[0]], [pie[1]], c=color, s=30, edgecolors="black",
                   zorder=5, linewidths=0.8)

        # Línea pies → punto de fuga
        label = nombre if nombre not in vistos else None
        vistos.add(nombre)
        ax.plot([pie[0], vp[0]], [pie[1], vp[1]],
                color=color, lw=1.3, alpha=0.85, label=label)

    # Punto de fuga
    ax.scatter([vp[0]], [vp[1]], c="red", s=160, marker="X",
               edgecolors="white", linewidths=1.5, zorder=6)
    ax.annotate(f"VP ({vp[0]:.0f}, {vp[1]:.0f})",
                xy=vp, xytext=(vp[0] + 15, vp[1] - 15),
                color="white", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.85))

    # Ampliar la vista si el VP cae fuera de la imagen
    margen = 40
    ax.set_xlim(min(0, vp[0] - margen), max(w, vp[0] + margen))
    ax.set_ylim(max(h, vp[1] + margen), min(0, vp[1] - margen))

    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title(f"Líneas de offside: pies de cada jugador → punto de fuga "
                 f"({len(jugadores)} jugadores)")

    if guardar_en:
        fig.savefig(guardar_en, dpi=110, bbox_inches="tight")
        print(f"Guardado: {guardar_en}")
    if mostrar:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    IMG_PATH = "data/raw/test/images/frame0-00-04-67_jpg.rf.207af35afe3e4aad5e1f275c34e9c95a.jpg"
    LBL_PATH = "data/raw/test/labels/frame0-00-04-67_jpg.rf.207af35afe3e4aad5e1f275c34e9c95a.txt"

    img = cv2.imread(IMG_PATH)
    if img is None:
        print("ERROR: imagen no encontrada")
    else:
        # 1. El usuario selecciona el VP con 4 clics (2 por línea)
        vp = seleccionar_punto_de_fuga_manual(img, guardar_en="dev/vp_manual.png")
        if vp is not None:
            print(f"Punto de fuga: ({vp[0]:.1f}, {vp[1]:.1f})")
            # 2. Dibujar líneas desde los jugadores del dataset hacia el VP
            graficar_jugadores_y_vp(img, LBL_PATH, vp,
                                    guardar_en="dev/vp_jugadores.png")