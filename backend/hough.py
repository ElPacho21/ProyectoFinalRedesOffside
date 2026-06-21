"""Vanishing point detection from soccer field lines."""
import random
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


def _dist_punto_a_linea(pt, seg):
    """Perpendicular distance from pt to the infinite line through seg."""
    x1, y1, x2, y2 = seg
    px, py = pt
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return float('inf')
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / np.hypot(dx, dy)


def _ransac_vp(segs, angulos, w, h, n_iter=500):
    """RANSAC vanishing point estimation.
    Tries ALL pairs of lines (no group split) — works even when all lines
    are from the same parallel family (e.g., both sides of penalty area).
    Returns (vp, inliers_a, inliers_b) split by angle sign for display."""
    if len(segs) < 2:
        return None, [], []

    thresh = np.hypot(w, h) * 0.012

    best_vp       = None
    best_count    = 0
    best_inliers  = []

    indices = list(range(len(segs)))
    rng = random.Random(42)  # fixed seed → deterministic results

    for _ in range(n_iter):
        i, j = rng.sample(indices, 2)
        vp = _interseccion(segs[i], segs[j])
        if vp is None:
            continue
        if abs(vp[0]) > 8 * w or abs(vp[1]) > 8 * h:
            continue

        inliers = [s for s in segs if _dist_punto_a_linea(vp, s) < thresh]
        count   = len(inliers)

        # Prefer VP outside image — inside VPs usually come from the
        # perpendicular line family, not the depth-direction family
        vp_inside = (0 <= vp[0] <= w) and (0 <= vp[1] <= h)
        score = count * (0.25 if vp_inside else 1.0)

        if score > best_count:
            best_count   = score
            best_vp      = vp
            best_inliers = inliers

    if best_vp is None or not best_inliers:
        return None, [], []

    # Refine with median of all pairwise intersections among inliers
    pts = []
    for i in range(len(best_inliers)):
        for j in range(i + 1, len(best_inliers)):
            pt = _interseccion(best_inliers[i], best_inliers[j])
            if pt and abs(pt[0]) < 8 * w and abs(pt[1]) < 8 * h:
                pts.append(pt)

    if pts:
        best_vp = tuple(np.median(pts, axis=0))

    # Split inliers by angle sign for frontend color coding
    seg_to_ang = dict(zip(segs, angulos))
    ia = [s for s in best_inliers if seg_to_ang.get(s, 0) >= 0]
    ib = [s for s in best_inliers if seg_to_ang.get(s, 0) < 0]
    return best_vp, ia, ib


def detectar_punto_de_fuga(imagen_bgr):
    """Detects vanishing point from white field lines on green grass.
    Returns (vp, grupo_a, grupo_b) or (None, [], []) if not enough lines."""
    h, w = imagen_bgr.shape[:2]
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    # Green field mask — dilated to cover adjacent white lines
    mask_verde = cv2.inRange(hsv, (25, 30, 30), (90, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask_campo = cv2.dilate(mask_verde, kernel, iterations=2)

    # White lines mask (low saturation, high value)
    mask_blanca = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
    mask_lineas = cv2.bitwise_and(mask_blanca, mask_campo)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_lineas = cv2.morphologyEx(mask_lineas, cv2.MORPH_CLOSE, k2)

    # Run LSD on full grayscale — no pre-mask so we don't miss lines
    # due to mask errors near goal nets / shadows
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lsd_lines = lsd.detect(gray)[0]

    def _seg_on_field(x1, y1, x2, y2, n_samples=16):
        """True if segment is on the field — sample green grass on both sides."""
        ih, iw = mask_campo.shape
        xs = np.linspace(x1, x2, n_samples)
        ys = np.linspace(y1, y2, n_samples)
        dx, dy = x2 - x1, y2 - y1
        seg_len = max(np.hypot(dx, dy), 1e-6)
        pnx, pny = -dy / seg_len, dx / seg_len
        green_max = 0.0
        for sign in (1, -1):
            xg = np.clip((xs + pnx * sign * 8).astype(int), 0, iw - 1)
            yg = np.clip((ys + pny * sign * 8).astype(int), 0, ih - 1)
            green_max = max(green_max, mask_campo[yg, xg].mean() / 255)
        return green_max

    segs, angulos = [], []
    n_raw = len(lsd_lines) if lsd_lines is not None else 0
    n_len = n_ang = n_color = 0
    if lsd_lines is not None:
        for l in lsd_lines:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length < 20:
                continue
            n_len += 1
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            abs_ang = abs(ang)
            # Filter near-horizontal in both directions (0° and 180°)
            if abs_ang < 10 or abs_ang > 170:
                continue
            # Filter advertising/scoreboard lines: their LOWER endpoint is still near
            # the top. A real field line extends well into the field (max y > 20%).
            if max(y1, y2) < h * 0.20:
                continue
            n_ang += 1
            green_score = _seg_on_field(x1, y1, x2, y2)
            if green_score > 0.30:
                n_color += 1
                segs.append((x1, y1, x2, y2))
                angulos.append(ang)
            elif length > 40:
                print(f"  FAILED field ang={ang:.1f}° len={length:.0f} green={green_score:.2f} ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    print(f"[hough] LSD raw={n_raw} len≥20={n_len} ang_ok={n_ang} color_ok={n_color} → final={len(segs)}")
    for s, a in zip(segs, angulos):
        print(f"  seg ang={a:.1f}° ({s[0]:.0f},{s[1]:.0f})-({s[2]:.0f},{s[3]:.0f})")

    # Deduplicate: LSD detects both edges of each white line (opposite angle signs,
    # nearly same position). Two segments are duplicates if they are nearly parallel
    # (angles differ by ≤15° ignoring sign) AND midpoints are within 20px.
    def _ang_diff(a1, a2):
        d = abs(abs(a1) - abs(a2))
        return min(d, 180 - d)

    merged, merged_angs = [], []
    used = [False] * len(segs)
    for i in range(len(segs)):
        if used[i]: continue
        s1 = segs[i]
        mx1 = (s1[0] + s1[2]) / 2; my1 = (s1[1] + s1[3]) / 2
        l1 = np.hypot(s1[2]-s1[0], s1[3]-s1[1])
        for j in range(i + 1, len(segs)):
            if used[j]: continue
            s2 = segs[j]
            mx2 = (s2[0] + s2[2]) / 2; my2 = (s2[1] + s2[3]) / 2
            parallel = _ang_diff(angulos[i], angulos[j]) < 15
            close = np.hypot(mx1 - mx2, my1 - my2) < 20
            if parallel and close:
                l2 = np.hypot(s2[2]-s2[0], s2[3]-s2[1])
                if l2 > l1:
                    segs[i] = s2; angulos[i] = angulos[j]
                    mx1 = mx2; my1 = my2; l1 = l2
                used[j] = True
        merged.append(segs[i])
        merged_angs.append(angulos[i])
    segs, angulos = merged, merged_angs

    if len(segs) < 2:
        return None, [], []

    vp, grupo_a, grupo_b = _ransac_vp(segs, angulos, w, h)
    return vp, grupo_a, grupo_b


def punto_de_fuga_desde_puntos(pts):
    """Calculates VP from 4 points: first two define line A, last two line B."""
    (p1, p2, p3, p4) = pts
    return _interseccion(
        (p1[0], p1[1], p2[0], p2[1]),
        (p3[0], p3[1], p4[0], p4[1]),
    )
