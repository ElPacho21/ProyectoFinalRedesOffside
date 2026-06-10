"""Pipeline completo de datos para detección de offside.

Uso:
    from src.pipeline import build_dataloaders

    loaders = build_dataloaders(
        url="https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5",
        api_key="TU_KEY",   # o env ROBOFLOW_API_KEY
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]
    class_names  = loaders["class_names"]

CLI smoke-test:
    python -m src.pipeline "https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5"
"""

import os
import sys
import random
import subprocess
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import DetectionDataset, detection_collate_fn, build_transforms, generate_csvs

_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR / "data"
_RAW_DIR  = _DATA_DIR / "raw"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_download(url: Optional[str], api_key: Optional[str]) -> None:
    script = _DATA_DIR / "download_dataset.py"
    cmd = [sys.executable, str(script)]
    if url:
        cmd += ["--url", url]
    if api_key:
        cmd += ["--api-key", api_key]

    env = os.environ.copy()
    if api_key:
        env["ROBOFLOW_API_KEY"] = api_key

    subprocess.run(cmd, check=True, env=env)


def build_dataloaders(
    url: str,
    api_key: Optional[str] = None,
    batch_size: int = 8,
    img_size: int = 640,
    seed: int = 42,
    num_workers: Optional[int] = None,
    force_download: bool = False,
) -> dict:
    """Pipeline completo: descarga → CSVs → Dataset → DataLoader.

    Args:
        url:            URL de Roboflow Universe del dataset.
        api_key:        Roboflow API key. Si es None lee ROBOFLOW_API_KEY del entorno.
        batch_size:     Tamaño de batch (default 8).
        img_size:       Tamaño de imagen cuadrada en px (default 640).
        seed:           Semilla de reproducibilidad (default 42).
        num_workers:    Workers del DataLoader. Auto-detecta SO si None.
        force_download: Si True, re-descarga aunque raw/ ya exista.

    Returns:
        dict con claves: "train", "val", "test" (DataLoaders), "class_names", "num_classes".
    """
    _set_seed(seed)

    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    # ── 1. Descarga ──────────────────────────────────────────────────────────
    if force_download or not _RAW_DIR.exists() or not any(_RAW_DIR.rglob("*.jpg")):
        print("Descargando dataset...")
        _run_download(url=url, api_key=api_key)
    else:
        print("Dataset ya disponible. (force_download=False para re-descargar)")

    # ── 2. Metadatos de clases ───────────────────────────────────────────────
    with open(_RAW_DIR / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    num_classes = data_cfg["nc"]
    print(f"Clases ({num_classes}): {class_names}")

    # ── 3. CSVs ──────────────────────────────────────────────────────────────
    csv_paths = {s: _DATA_DIR / f"{s}.csv" for s in ("train", "val", "test")}
    if not all(p.exists() for p in csv_paths.values()):
        print("Generando CSVs...")
        generate_csvs(_RAW_DIR, _DATA_DIR, class_names)
    else:
        print("CSVs ya existen.")

    # ── 4. Transforms ────────────────────────────────────────────────────────
    train_tf, val_tf = build_transforms(img_size)

    # ── 5. Datasets ──────────────────────────────────────────────────────────
    train_ds = DetectionDataset(csv_paths["train"], class_names, transform=train_tf)
    val_ds   = DetectionDataset(csv_paths["val"],   class_names, transform=val_tf)
    test_ds  = DetectionDataset(csv_paths["test"],  class_names, transform=val_tf)

    print(f"Train: {len(train_ds)} imgs | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── 6. DataLoaders ───────────────────────────────────────────────────────
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=detection_collate_fn,
        pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=detection_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=detection_collate_fn,
    )

    print(
        f"Loaders — train: {len(train_loader)} batches | "
        f"val: {len(val_loader)} | test: {len(test_loader)}"
    )

    return {
        "train":       train_loader,
        "val":         val_loader,
        "test":        test_loader,
        "class_names": class_names,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de datos — smoke test")
    parser.add_argument("url", nargs="?", default="https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5", help="URL de Roboflow Universe")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    loaders = build_dataloaders(
        url=args.url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        force_download=args.force_download,
    )

    imgs, targets = next(iter(loaders["train"]))
    print(f"\nSmoke test OK:")
    print(f"  Batch shape : {tuple(imgs.shape)}")
    print(f"  dtype       : {imgs.dtype} | rango: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  Targets     : {len(targets)}")
