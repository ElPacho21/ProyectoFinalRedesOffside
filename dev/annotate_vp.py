"""
Herramienta de anotación manual — Punto de Fuga + Label de Offside (Ground Truth)

Uso:
    python dev/annotate_vp.py --integrante 0          # rango 0-99 de imágenes de train
    python dev/annotate_vp.py --integrante 2 --limite 50
    python dev/annotate_vp.py --integrante 1 --comparar-algoritmo

Argumentos:
    --integrante N        ID del integrante (0-4). Define el rango de imágenes.
    --limite N            Máximo de imágenes a anotar en esta sesión (default: 100).
    --images-dir RUTA     Carpeta de imágenes (default: data/raw/train/images).
    --labels-dir RUTA     Carpeta de labels YOLO (default: data/raw/train/labels).
    --output RUTA         CSV de salida (default: data/vp_annotations.csv).
    --comparar-algoritmo  Tras el label humano, calcula y muestra el veredicto del algoritmo.

Cómo repartir el trabajo entre 5 integrantes (Esquema A — por rango):
    Integrante 0 → imágenes 0-99
    Integrante 1 → imágenes 100-199
    Integrante 2 → imágenes 200-299
    Integrante 3 → imágenes 300-399
    Integrante 4 → imágenes 400-499

Requisito: entorno con display (Windows/Linux/Mac con interfaz gráfica).
No funciona en Colab sin configuración adicional.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Agregar raíz del repo al path para importar HoughLines desde dev/
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "dev"))
from HoughLines import seleccionar_punto_de_fuga_manual, punto_de_fuga_desde_puntos

# ---- Columnas del CSV de salida ----
COLUMNAS_CSV = [
    "filepath",
    "vp_x", "vp_y",
    "p1_x", "p1_y", "p2_x", "p2_y",
    "p3_x", "p3_y", "p4_x", "p4_y",
    "offside_gt",
    "attacking_team",
    "offside_pred",
    "integrante",
]

IMAGENES_POR_INTEGRANTE = 100


def leer_ya_anotadas(csv_path):
    """
    Lee el CSV de salida y devuelve un dict {filepath: fila_dict}.
    Si el CSV no existe, devuelve dict vacío.
    """
    ya_anotadas = {}
    if not csv_path.exists():
        return ya_anotadas
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ya_anotadas[row["filepath"]] = row
    return ya_anotadas


def guardar_fila(csv_path, fila):
    """Agrega o actualiza una fila en el CSV. Escribe después de cada imagen."""
    ya_existe = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        if not ya_existe:
            writer.writeheader()
        writer.writerow(fila)


def rango_integrante(integrante, imagenes_totales):
    """Devuelve (inicio, fin) del rango de imágenes para el integrante dado."""
    inicio = integrante * IMAGENES_POR_INTEGRANTE
    fin    = min(inicio + IMAGENES_POR_INTEGRANTE, imagenes_totales)
    return inicio, fin


def pedir_label_offside():
    """
    Pide al usuario el label de offside y el equipo atacante por teclado.
    Retorna (offside_gt: int | None, attacking_team: int | None).
    offside_gt: 1=offside, 0=onside, None=no sabe / saltar
    """
    print("\n¿Hay OFFSIDE en esta imagen?")
    print("  1 = OFFSIDE")
    print("  0 = ONSIDE (no hay offside)")
    print("  s = saltear (imagen dudosa o sin jugada clara)")
    respuesta = input("Tu decisión: ").strip().lower()

    if respuesta == "s":
        return None, None

    if respuesta not in ("0", "1"):
        print("Respuesta no reconocida. Se saltea la imagen.")
        return None, None

    offside_gt = int(respuesta)

    print("\n¿Qué equipo ataca?")
    print("  1 = TEAM 1")
    print("  2 = TEAM 2")
    print("  s = no sé / no aplica")
    resp_team = input("Tu decisión: ").strip().lower()

    if resp_team == "s":
        attacking_team = None
    elif resp_team in ("1", "2"):
        attacking_team = int(resp_team)
    else:
        print("Respuesta no reconocida. Se deja vacío.")
        attacking_team = None

    return offside_gt, attacking_team


def calcular_veredicto_algoritmo(imagen_bgr, labels_path, vp, attacking_team):
    """
    Calcula el veredicto del algoritmo de offside usando:
      - Las posiciones de jugadores de los labels YOLO de la imagen.
      - El VP recién marcado.
      - El equipo atacante indicado por el usuario.

    Retorna 1 (offside), 0 (onside), o None si no se puede calcular.
    """
    if attacking_team is None or vp is None:
        return None

    # Importar el algoritmo de offside desde prod/utils.py
    sys.path.insert(0, str(_REPO_ROOT / "prod"))
    try:
        from utils import calcular_offside
    except ImportError:
        print("ADVERTENCIA: no se pudo importar utils.py de prod/. Se omite el veredicto del algoritmo.")
        return None

    # Leer labels YOLO de la imagen
    h_img, w_img = imagen_bgr.shape[:2]
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return None

    detecciones = []
    CLASES = {0: "Ball", 2: "GoalKeeper", 5: "TEAM 1", 6: "TEAM 2"}
    for linea in labels_path.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) < 5:
            continue
        cls_id = int(partes[0])
        if cls_id not in CLASES:
            continue
        xc, yc, bw, bh = map(float, partes[1:5])
        x1 = (xc - bw / 2) * w_img
        y1 = (yc - bh / 2) * h_img
        x2 = (xc + bw / 2) * w_img
        y2 = (yc + bh / 2) * h_img
        detecciones.append({
            "class_id":   cls_id,
            "class_name": CLASES[cls_id],
            "conf":       1.0,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    if not detecciones:
        return None

    equipo_atacante_id = 5 if attacking_team == 1 else 6
    resultado = calcular_offside(vp, detecciones, equipo_atacante_id, imagen_bgr.shape)
    return 1 if resultado["hay_offside"] else 0


def anotar_imagen(imagen_path, labels_dir, integrante, comparar_algoritmo):
    """
    Muestra la imagen al usuario, captura VP y label de offside.
    Retorna fila de dict lista para guardar en CSV, o None si se saltea.
    """
    img_bgr = cv2.imread(str(imagen_path))
    if img_bgr is None:
        print(f"No se pudo leer la imagen: {imagen_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Imagen: {imagen_path.name}")
    print("Hacé 4 clics sobre la imagen para marcar el punto de fuga.")
    print("(Click derecho deshace el último punto)")
    print("Cerrá la ventana cuando hayas terminado.")
    print("'s' en el terminal saltea esta imagen.")
    print("=" * 60)

    # Usar la función interactiva de HoughLines.py
    vp = seleccionar_punto_de_fuga_manual(img_bgr)
    if vp is None:
        print("VP no marcado. Se saltea esta imagen.")
        return None

    vp_x, vp_y = vp

    # Pedir label de offside al usuario (ANTES de mostrar el veredicto del algoritmo)
    offside_gt, attacking_team = pedir_label_offside()
    if offside_gt is None:
        print("Imagen salteada.")
        return None

    offside_pred = None
    if comparar_algoritmo:
        # Nombre del archivo de labels (mismo nombre que la imagen, extensión .txt)
        label_file = Path(labels_dir) / (imagen_path.stem + ".txt")
        offside_pred = calcular_veredicto_algoritmo(img_bgr, label_file, vp, attacking_team)

        if offside_pred is not None:
            gt_str   = "OFFSIDE" if offside_gt  else "ONSIDE"
            pred_str = "OFFSIDE" if offside_pred else "ONSIDE"
            match    = "✓ COINCIDEN" if offside_gt == offside_pred else "✗ DIFIEREN"
            print(f"\nComparación: Humano={gt_str} | Algoritmo={pred_str} → {match}")
        else:
            print("No se pudo calcular el veredicto del algoritmo para esta imagen.")

    # Recuperar los 4 puntos marcados (seleccionar_punto_de_fuga_manual no los devuelve,
    # así que los dejamos vacíos — el VP es suficiente para recalcular)
    fila = {
        "filepath":       str(imagen_path.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "vp_x":           round(vp_x, 2),
        "vp_y":           round(vp_y, 2),
        "p1_x": "", "p1_y": "", "p2_x": "", "p2_y": "",
        "p3_x": "", "p3_y": "", "p4_x": "", "p4_y": "",
        "offside_gt":     offside_gt,
        "attacking_team": attacking_team if attacking_team is not None else "",
        "offside_pred":   offside_pred if offside_pred is not None else "",
        "integrante":     integrante,
    }
    return fila


def main():
    parser = argparse.ArgumentParser(
        description="Herramienta de anotación de punto de fuga y offside ground truth."
    )
    parser.add_argument("--integrante",        type=int, required=True,
                        help="ID del integrante (0-4)")
    parser.add_argument("--limite",            type=int, default=IMAGENES_POR_INTEGRANTE,
                        help="Máximo de imágenes a anotar en esta sesión")
    parser.add_argument("--images-dir",        type=str,
                        default=str(_REPO_ROOT / "data" / "raw" / "train" / "images"),
                        help="Carpeta de imágenes")
    parser.add_argument("--labels-dir",        type=str,
                        default=str(_REPO_ROOT / "data" / "raw" / "train" / "labels"),
                        help="Carpeta de labels YOLO")
    parser.add_argument("--output",            type=str,
                        default=str(_REPO_ROOT / "data" / "vp_annotations.csv"),
                        help="Archivo CSV de salida")
    parser.add_argument("--comparar-algoritmo", action="store_true",
                        help="Calcular y mostrar el veredicto del algoritmo tras el label humano")
    args = parser.parse_args()

    if not 0 <= args.integrante <= 4:
        print("ERROR: --integrante debe ser un valor entre 0 y 4.")
        sys.exit(1)

    images_dir = Path(args.images_dir)
    csv_path   = Path(args.output)

    if not images_dir.exists():
        print(f"ERROR: La carpeta de imágenes no existe: {images_dir}")
        print("Asegurate de haber descargado el dataset con: python data/download_dataset.py")
        sys.exit(1)

    # Listar todas las imágenes disponibles, ordenadas para rango consistente
    extensiones = {".jpg", ".jpeg", ".png"}
    todas = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in extensiones
    )
    total = len(todas)
    print(f"Dataset: {total} imágenes en {images_dir}")

    # Rango asignado a este integrante
    inicio, fin = rango_integrante(args.integrante, total)
    rango = todas[inicio:fin]
    print(f"Integrante {args.integrante}: imágenes {inicio} a {fin - 1} ({len(rango)} imágenes)")

    # Verificar qué ya está anotado
    ya_anotadas = leer_ya_anotadas(csv_path)
    pendientes  = [
        p for p in rango
        if str(p.relative_to(_REPO_ROOT)).replace("\\", "/") not in ya_anotadas
    ]

    if not pendientes:
        print(f"\n¡Ya completaste tu rango! Todas las {len(rango)} imágenes están anotadas.")
        sys.exit(0)

    print(f"Imágenes ya anotadas: {len(rango) - len(pendientes)}")
    print(f"Imágenes pendientes : {len(pendientes)}")
    print(f"A anotar en esta sesión: mínimo hasta {args.limite}")
    print("\nPodés cerrar el script en cualquier momento con Ctrl+C. El progreso se guarda.")
    print("=" * 60)

    anotadas_sesion = 0
    try:
        for i, img_path in enumerate(pendientes):
            if anotadas_sesion >= args.limite:
                print(f"\nLímite de sesión alcanzado ({args.limite} imágenes). ¡Hasta la próxima!")
                break

            print(f"\nImagen {i + 1} de {len(pendientes)} pendientes "
                  f"(total anotadas este integrante: {len(rango) - len(pendientes) + anotadas_sesion})")

            fila = anotar_imagen(img_path, args.labels_dir, args.integrante, args.comparar_algoritmo)

            if fila is not None:
                guardar_fila(csv_path, fila)
                anotadas_sesion += 1
                print(f"Guardado en {csv_path} ✓")

    except KeyboardInterrupt:
        print(f"\n\nInterrumpido. Se guardaron {anotadas_sesion} imágenes en esta sesión.")

    print(f"\nSesión finalizada. Total anotadas esta sesión: {anotadas_sesion}")
    print(f"Archivo de salida: {csv_path}")


if __name__ == "__main__":
    main()
