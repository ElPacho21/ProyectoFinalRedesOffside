from roboflow import Roboflow
import os
import yaml
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

RAW_DIR = Path(__file__).parent / "raw"


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


def download_dataset() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        api_key = input("Ingresá tu Roboflow API key: ").strip()

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("animals-67mq4").project("team-separation")
    version = project.version(5)
    # Formato yolov8: images/ + labels/ por split, y data.yaml con nombres de clase
    version.download("yolov8", location=str(RAW_DIR))

    print_summary(RAW_DIR)


if __name__ == "__main__":
    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.jpg")):
        print("Dataset ya descargado. Salteando descarga.")
        print_summary(RAW_DIR)
    else:
        download_dataset()
