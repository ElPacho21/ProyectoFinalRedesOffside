import os
import re
import shutil
import sys
import yaml
from pathlib import Path
from roboflow import Roboflow

# Agregar el directorio raíz del repo a sys.path para poder importar src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

from data.download_dataset import download_dataset

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_NEW_DIR = DATA_DIR / "raw_new"
COMBINED_DIR = DATA_DIR / "combined"

NEW_DATASET_URL = "https://universe.roboflow.com/fcaicu-aivisionstats/football_object-detection/dataset/4"
CLASS_NAMES = ['Ball', 'Corner', 'GoalKeeper', 'Goal_Net', 'Referee', 'TEAM 1', 'TEAM 2']


def parse_roboflow_url(url: str) -> tuple[str, str, int]:
    """Extrae (workspace, project, version) de una URL de Roboflow Universe."""
    pattern = r"universe\.roboflow\.com/([^/]+)/([^/]+)/dataset/(\d+)"
    m = re.search(pattern, url)
    if not m:
        raise ValueError(
            f"URL no reconocida: {url!r}\n"
            "Formato esperado: https://universe.roboflow.com/{{workspace}}/{{project}}/dataset/{{version}}"
        )
    return m.group(1), m.group(2), int(m.group(3))


def merge_split(split: str, src_original: Path, src_new: Path, dest_combined: Path):
    """Fusiona las imágenes y etiquetas de un split específico, aplicando el remapeo de clases."""
    # Directorios de origen
    orig_img_dir = src_original / split / "images"
    orig_lbl_dir = src_original / split / "labels"
    
    new_img_dir = src_new / split / "images"
    new_lbl_dir = src_new / split / "labels"
    
    # Directorios de destino
    dest_img_dir = dest_combined / split / "images"
    dest_lbl_dir = dest_combined / split / "labels"
    
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copiar archivos originales tal cual
    if orig_img_dir.exists():
        for img_ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_file in orig_img_dir.glob(img_ext):
                shutil.copy(img_file, dest_img_dir / img_file.name)
    if orig_lbl_dir.exists():
        for lbl_file in orig_lbl_dir.glob("*.txt"):
            shutil.copy(lbl_file, dest_lbl_dir / lbl_file.name)
            
    # 2. Copiar archivos nuevos renombrándolos para evitar colisiones
    # y remapear IDs de clases en las etiquetas
    if new_img_dir.exists():
        for img_ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_file in new_img_dir.glob(img_ext):
                shutil.copy(img_file, dest_img_dir / f"new_{img_file.name}")
                
    if new_lbl_dir.exists():
        for lbl_file in new_lbl_dir.glob("*.txt"):
            lines = lbl_file.read_text().splitlines()
            mapped_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    # Remapeo de clases debido a diferencias en el orden del data.yaml nuevo
                    # Nuevo dataset: ['ball', 'corner', 'goal', 'goalkeeper', 'referee', 'team A', 'team B']
                    # original dataset: ['Ball', 'Corner', 'GoalKeeper', 'Goal_Net', 'Referee', 'TEAM 1', 'TEAM 2']
                    #
                    # Mapeo:
                    # 0 (ball)       -> 0 (Ball)
                    # 1 (corner)     -> 1 (Corner)
                    # 2 (goal)       -> 3 (Goal_Net)
                    # 3 (goalkeeper) -> 2 (GoalKeeper)
                    # 4 (referee)    -> 4 (Referee)
                    # 5 (team A)     -> 5 (TEAM 1)
                    # 6 (team B)     -> 6 (TEAM 2)
                    if cls_id == 2:
                        cls_id = 3
                    elif cls_id == 3:
                        cls_id = 2
                    
                    mapped_lines.append(f"{cls_id} {' '.join(parts[1:])}")
                else:
                    mapped_lines.append(line)
            
            dest_lbl_file = dest_lbl_dir / f"new_{lbl_file.name}"
            dest_lbl_file.write_text("\n".join(mapped_lines) + "\n")


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        print("Error: ROBOFLOW_API_KEY no configurado en el entorno o en el archivo .env.")
        sys.exit(1)
        
    # 1. Asegurar que el dataset original está descargado
    if not RAW_DIR.exists() or not any(RAW_DIR.rglob("*.jpg")):
        print("Descargando dataset original...")
        download_dataset(api_key=api_key, raw_dir=RAW_DIR)
    else:
        print("Dataset original ya disponible.")
        
    # 2. Descargar el nuevo dataset
    if not RAW_NEW_DIR.exists() or not any(RAW_NEW_DIR.rglob("*.jpg")):
        print(f"Descargando nuevo dataset desde {NEW_DATASET_URL}...")
        workspace, project_name, version_num = parse_roboflow_url(NEW_DATASET_URL)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)
        version.download("yolov8", location=str(RAW_NEW_DIR))
    else:
        print("Nuevo dataset ya disponible.")
        
    # 3. Preparar directorio combinado limpio
    if COMBINED_DIR.exists():
        print(f"Limpiando directorio combinado existente en {COMBINED_DIR}...")
        shutil.rmtree(COMBINED_DIR)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 4. Fusionar los splits
    splits = ["train", "valid", "test"]
    for split in splits:
        print(f"Fusionando split: {split}...")
        merge_split(split, RAW_DIR, RAW_NEW_DIR, COMBINED_DIR)
        
    # 5. Generar data.yaml combinado
    print("Generando data.yaml combinado...")
    data_yaml = {
        "path": str(COMBINED_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(COMBINED_DIR / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
        
    # 6. Generar archivos CSV combinados
    print("Generando archivos CSV combinados...")
    from src.dataset import generate_csvs
    generate_csvs(
        raw_dir=COMBINED_DIR,
        data_dir=COMBINED_DIR,
        class_names=CLASS_NAMES,
        base_dir=BASE_DIR
    )
    
    print("\n¡Fusión completada con éxito!")
    print(f"Dataset combinado disponible en: {COMBINED_DIR}")


if __name__ == "__main__":
    main()
