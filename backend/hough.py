"""Vanishing point detection from soccer field lines."""
import cv2
import numpy as np


def _interseccion(seg_a, seg_b):
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
    """Detects vanishing point from white field lines on green grass.
    Returns (vp, grupo_a, grupo_b) or (None, [], []) if not enough lines."""
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    mask_verde = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    mask_blanca = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    mask_final = cv2.bitwise_and(mask_blanca, mask_verde)
    edges = cv2.Canny(mask_final, 50, 150)
    lineas = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=20
    )
    if lineas is None:
        return None, [], []

    segs, angulos = [], []
    for l in lineas:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        abs_ang = abs(ang)
        if abs_ang < 10 or abs_ang > 80:
            continue
        segs.append((x1, y1, x2, y2))
        angulos.append(ang)

    grupo_a = [s for s, a in zip(segs, angulos) if a > 0]
    grupo_b = [s for s, a in zip(segs, angulos) if a < 0]

    if not grupo_a or not grupo_b:
        return None, grupo_a, grupo_b

    h, w = imagen_bgr.shape[:2]
    intersecciones = []
    for sa in grupo_a:
        for sb in grupo_b:
            pt = _interseccion(sa, sb)
            if pt is None:
                continue
            if abs(pt[0]) > 10 * w or abs(pt[1]) > 10 * h:
                continue
            intersecciones.append(pt)

    if not intersecciones:
        return None, grupo_a, grupo_b

    vp = tuple(np.median(intersecciones, axis=0))
    return vp, grupo_a, grupo_b


def punto_de_fuga_desde_puntos(pts):
    """Calculates VP from 4 points: first two define line A, last two line B."""
    (p1, p2, p3, p4) = pts
    return _interseccion(
        (p1[0], p1[1], p2[0], p2[1]),
        (p3[0], p3[1], p4[0], p4[1]),
    )
