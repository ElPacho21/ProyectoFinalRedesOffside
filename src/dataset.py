"""Clases y utilidades compartidas de datos para detección de offside.

Importar desde notebook o pipeline:
    from src.dataset import DetectionDataset, detection_collate_fn, build_transforms, generate_csvs
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

BASE_DIR = Path(__file__).resolve().parent.parent

_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


class DetectionDataset(Dataset):
    """Dataset de detección de objetos. Lee imágenes y anotaciones desde CSV.

    CSV columns: filepath, class_name, class_id, x_center, y_center, width, height
    Sin transform: devuelve (PIL.Image, dict).
    Con transform: devuelve (Tensor[C,H,W] float [0,1], dict con boxes/labels tensors).
    """

    def __init__(self, csv_path: Path, class_names: list[str], transform=None, base_dir: Path = None):
        self.base_dir    = Path(base_dir) if base_dir else BASE_DIR
        self.transform   = transform
        self.class_names = class_names

        df = pd.read_csv(csv_path)
        self.samples = []
        for filepath, group in df.groupby("filepath", sort=False):
            self.samples.append({
                "filepath": filepath,
                "boxes":    group[["x_center", "y_center", "width", "height"]].values.tolist(),
                "labels":   group["class_id"].values.tolist(),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s      = self.samples[idx]
        img    = Image.open(self.base_dir / s["filepath"]).convert("RGB")
        boxes  = s["boxes"]
        labels = s["labels"]

        if self.transform:
            result = self.transform(image=np.array(img), bboxes=boxes, class_labels=labels)
            img_out    = torch.from_numpy(result["image"]).permute(2, 0, 1).float() / 255.0
            out_boxes  = result["bboxes"]
            out_labels = result["class_labels"]
            target = {
                "boxes":  torch.tensor(out_boxes, dtype=torch.float32).reshape(-1, 4)
                          if out_boxes else torch.zeros((0, 4)),
                "labels": torch.tensor(out_labels, dtype=torch.int64),
            }
            return img_out, target
        else:
            return img, {"boxes": boxes, "labels": labels, "filepath": s["filepath"]}


def detection_collate_fn(batch):
    """Apila imágenes y agrupa targets como lista (variable número de boxes por imagen)."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_transforms(img_size: int = 640):
    """Devuelve (train_transform, val_test_transform) con albumentations.

    train_transform: resize + pad + augmentations (flip, brillo, hue limitado, blur).
    val_test_transform: solo resize + pad (determinista).
    """
    train_tf = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=114),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # hue limitado a ±5° para no alterar colores de camiseta
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )
    val_tf = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=114),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )
    return train_tf, val_tf


def generate_csvs(raw_dir: Path, data_dir: Path, class_names: list[str], base_dir: Path = None) -> None:
    """Parsea labels YOLO y genera train.csv, val.csv, test.csv en data_dir."""
    if base_dir is None:
        base_dir = BASE_DIR

    for csv_name, split_folder in _SPLIT_MAP.items():
        img_dir = raw_dir / split_folder / "images"
        lbl_dir = raw_dir / split_folder / "labels"
        if not img_dir.exists():
            print(f"  Advertencia: {img_dir} no existe, saltando split '{csv_name}'.")
            continue

        records = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            rel_path = str(img_path.relative_to(base_dir)).replace("\\", "/")
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        records.append({
                            "filepath":   rel_path,
                            "class_name": class_names[cls_id],
                            "class_id":   cls_id,
                            "x_center":   xc,
                            "y_center":   yc,
                            "width":      w,
                            "height":     h,
                        })

        df = pd.DataFrame(records)
        out = data_dir / f"{csv_name}.csv"
        df.to_csv(out, index=False)
        n_imgs = df["filepath"].nunique()
        print(f"{csv_name}.csv guardado — {len(df)} anotaciones | {n_imgs} imágenes")
        print(df.groupby("class_name").size().to_string())
        print()
