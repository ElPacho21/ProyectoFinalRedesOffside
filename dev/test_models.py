# -*- coding: utf-8 -*-
"""
Script de prueba de modelos entrenados para detección de objetos.

Este script permite cargar cualquiera de los tres modelos YOLO entrenados (*.pt),
realizar inferencia sobre una imagen de prueba (ya sea especificada o elegida
aleatoriamente del set de test) y visualizar las detecciones con colores y etiquetas.

Uso:
    # Ejecutar con el modelo 1 (YOLOv8s Full) y una imagen aleatoria del set de test
    python dev/test_models.py --model 1

    # Ejecutar con el modelo 3 (YOLOv8m Weighted) y una imagen específica
    python dev/test_models.py --model 3 --image data/raw/test/images/mi_imagen.jpg

    # Ver opciones disponibles
    python dev/test_models.py --help
"""

import os
import sys
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Asegurar que el directorio raíz del proyecto esté en el path de Python
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO

# Rutas de los modelos entrenados en el proyecto
MODEL_PATHS = {
    1: {
        "name": "Exp1: YOLOv8s Full Fine-Tune",
        "path": _ROOT / "dev" / "YOLOv8s Full" / "exp1_yolov8s_full.pt"
    },
    2: {
        "name": "Exp2: YOLOv8s Freeze Backbone",
        "path": _ROOT / "dev" / "YOLOv8s Freeze" / "exp2_yolov8s_frozen.pt"
    },
    3: {
        "name": "Exp3: YOLOv8m Weighted",
        "path": _ROOT / "dev" / "YOLOv8m Weighted" / "exp3_yolov8m_weighted.pt"
    }
}

# Paleta de colores premium para cada clase de detección
CLASS_COLORS = {
    "Ball": "#FF5722",        # Naranja vibrante
    "Corner": "#FFEB3B",      # Amarillo
    "GoalKeeper": "#00BCD4",  # Cyan / Celeste
    "Goal_Net": "#9C27B0",    # Púrpura
    "Referee": "#E91E63",     # Rosado / Magenta
    "TEAM 1": "#F44336",      # Rojo / Carmesí
    "TEAM 2": "#7986CB"       # Azul suave para contraste
}


def main():
    parser = argparse.ArgumentParser(description="Prueba e inferencia de modelos YOLO entrenados (detección pura).")
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Número de experimento/modelo a usar: 1 (YOLOv8s Full), 2 (YOLOv8s Freeze), 3 (YOLOv8m Weighted). Por defecto: 1."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta de la imagen a testear. Si no se especifica, se seleccionará una aleatoria del set de test."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Umbral de confianza para las predicciones de YOLO. Por defecto: 0.25."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Ruta de guardado para la imagen resultante anotada. Por defecto: dev/test_result_model<N>.png."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra la imagen interactiva usando matplotlib si está en un entorno con interfaz gráfica."
    )

    args = parser.parse_args()

    # 1. Cargar el modelo seleccionado
    model_info = MODEL_PATHS[args.model]
    model_path = model_info["path"]
    model_name = model_info["name"]

    print("=" * 60)
    print(f" Cargando modelo: {model_name}")
    print(f" Ruta de pesos: {model_path}")
    print("=" * 60)

    if not model_path.exists():
        print(f"ERROR: No se encontró el archivo de pesos en {model_path}")
        print("Por favor, asegúrate de que el archivo exista en la ubicación correspondiente.")
        return

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"ERROR al cargar el modelo con ultralytics: {e}")
        return

    # Obtener nombres de las clases del modelo
    names = model.names
    print(f"Clases cargadas ({len(names)}): {list(names.values())}")

    # 2. Seleccionar la imagen de prueba
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: La imagen especificada no existe en: {image_path}")
            return
    else:
        # Intentar buscar en data/raw/test/images
        test_dir = _ROOT / "data" / "raw" / "test" / "images"
        if test_dir.exists():
            test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if test_images:
                image_path = random.choice(test_images)
                print(f"Seleccionada imagen aleatoria de test: {image_path.name}")
            else:
                print(f"ERROR: No se encontraron imágenes en {test_dir}")
                return
        else:
            print(f"ERROR: Carpeta de test no encontrada en {test_dir}. Especifica una imagen con --image.")
            return

    # 3. Leer imagen y correr inferencia
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"ERROR: No se pudo leer la imagen en {image_path}")
        return

    h, w, _ = img_bgr.shape
    print(f"Resolución de la imagen: {w}x{h}")

    print(f"Corriendo inferencia con conf={args.conf}...")
    results = model.predict(source=img_bgr, conf=args.conf, verbose=False)[0]

    # Contar detecciones
    detections_summary = {}
    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = names.get(cls_id, f"Clase_{cls_id}")
        detections_summary[cls_name] = detections_summary.get(cls_name, 0) + 1

    print("\n--- Resumen de Detecciones ---")
    if not detections_summary:
        print("  No se detectaron objetos con el umbral especificado.")
    for cls_name, count in sorted(detections_summary.items()):
        print(f"  * {cls_name}: {count}")
    print("------------------------------\n")

    # 4. Configurar el gráfico de Matplotlib para la visualización
    fig, ax = plt.subplots(figsize=(12, 8))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis("off")

    # 5. Dibujar cajas de detección de YOLO
    for box in results.boxes:
        # Coordenadas, clase y conf
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        cls_name = names.get(cls_id, f"Clase_{cls_id}")

        # Color correspondiente a la clase
        color_hex = CLASS_COLORS.get(cls_name, "#4CAF50")
        
        # Bbox del objeto
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color_hex, facecolor="none"
        )
        ax.add_patch(rect)

        # Label con texto y confianza
        label_str = f"{cls_name} {conf:.2f}"
        ax.text(
            x1, y1 - 6, label_str,
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor=color_hex, alpha=0.8, edgecolor="none", boxstyle="square,pad=0.2")
        )

    # Forzar límites del eje coincidiendo con la resolución original
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # 6. Título y Guardado
    title_str = f"Detección de Objetos | Modelo: {model_name} | Imagen: {image_path.name}"
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=10)

    # Configurar ruta de guardado
    if args.save_path:
        out_path = Path(args.save_path)
    else:
        out_path = _ROOT / "dev" / f"test_result_model{args.model}.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"¡Resultados guardados exitosamente en: {out_path}!")
    print("=" * 60)

    if args.show:
        try:
            # Mostrar la imagen guardada usando la app por defecto del sistema
            if os.name == "nt":
                os.startfile(str(out_path))
            else:
                import subprocess
                subprocess.run(["xdg-open", str(out_path)])
        except Exception as e:
            print(f"No se pudo mostrar la imagen de forma interactiva: {e}")


if __name__ == "__main__":
    main()
