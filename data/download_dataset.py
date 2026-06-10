from roboflow import Roboflow
import os
import re
import yaml
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

RAW_DIR = Path(__file__).parent / "raw"

# Defaults usados cuando no se pasa URL
_DEFAULT_WORKSPACE = "animals-67mq4"
_DEFAULT_PROJECT   = "team-separation"
_DEFAULT_VERSION   = 5


def parse_roboflow_url(url: str) -> tuple[str, str, int]:
    """Extrae (workspace, project, version) de una URL de Roboflow Universe.

    Formato esperado: https://universe.roboflow.com/{workspace}/{project}/dataset/{version}
    """
    pattern = r"universe\.roboflow\.com/([^/]+)/([^/]+)/dataset/(\d+)"
    m = re.search(pattern, url)
    if not m:
        raise ValueError(
            f"URL no reconocida: {url!r}\n"
            "Formato esperado: https://universe.roboflow.com/{{workspace}}/{{project}}/dataset/{{version}}"
        )
    return m.group(1), m.group(2), int(m.group(3))


def load_class_names(raw_dir: Path) -> list[str]:
    yaml_path = raw_dir / "data.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["names"]


def print_summary(raw_dir: Path) -> None:
    class_names = load_class_names(raw_dir)
    splits = ["train", "valid", "test"]
    print("\n" + "=" * 50)
    print("RESUMEN DEL DATASET DESCARGADO")
    print("=" * 50)
    print(f"\nClases ({len(class_names)}): {class_names}")
    for split in splits:
        img_dir = raw_dir / split / "images"
        lbl_dir = raw_dir / split / "labels"
        if not img_dir.exists():
            continue
        n_imgs = len(list(img_dir.glob("*")))
        print(f"\n{split}/ — {n_imgs} imágenes")
        if lbl_dir.exists():
            counts = {c: 0 for c in class_names}
            for lbl_file in lbl_dir.glob("*.txt"):
                for line in lbl_file.read_text().splitlines():
                    if line.strip():
                        cls_id = int(line.split()[0])
                        counts[class_names[cls_id]] += 1
            for cls, n in counts.items():
                print(f"  └─ {cls}: {n} anotaciones")
    print("=" * 50)


def download_dataset(url: str = None, api_key: str = None, raw_dir: Path = None) -> Path:
    """Descarga dataset desde Roboflow en formato YOLOv8.

    Args:
        url:     URL de Roboflow Universe. Si es None usa el dataset por defecto.
        api_key: Roboflow API key. Si es None lee ROBOFLOW_API_KEY del entorno.
        raw_dir: Directorio destino. Si es None usa data/raw/ relativo a este script.

    Returns:
        Path al directorio raw descargado.
    """
    if raw_dir is None:
        raw_dir = RAW_DIR

    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        api_key = input("Ingresá tu Roboflow API key: ").strip()

    if url:
        workspace, project_name, version_num = parse_roboflow_url(url)
    else:
        workspace, project_name, version_num = _DEFAULT_WORKSPACE, _DEFAULT_PROJECT, _DEFAULT_VERSION

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    version.download("yolov8", location=str(raw_dir))

    print_summary(raw_dir)
    return raw_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descarga dataset de Roboflow en formato YOLOv8")
    parser.add_argument("--url", default=None, help="URL de Roboflow Universe")
    parser.add_argument("--api-key", default=None, help="Roboflow API key (o env ROBOFLOW_API_KEY)")
    args = parser.parse_args()

    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.jpg")):
        print("Dataset ya descargado. Salteando descarga.")
        print_summary(RAW_DIR)
    else:
        download_dataset(url=args.url, api_key=args.api_key)
