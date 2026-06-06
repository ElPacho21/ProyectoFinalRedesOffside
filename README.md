# TP Integrador — Detección de Jugadores para Offside en Fútbol

Trabajo Práctico Integrador de la materia *Redes Neuronales Profundas (Ingeniería en Sistemas de Información)*.

El objetivo es entrenar un modelo de detección de objetos (YOLOv8s) sobre imágenes de broadcast de fútbol para localizar jugadores, árbitros y la pelota mediante bounding boxes — información necesaria para determinar la línea de offside.

## Integrantes

- Facundo Pacci
- Nicolás Ocampo
- Valentino Isgro
- Bruno Lucero
- Juan Pablo Costa

## Aplicación desplegada

[TBD — se completará en la Semana 4]

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python >= 3.9 |
| Deep Learning | PyTorch + YOLOv8 (Ultralytics) |
| Augmentations | Albumentations |
| Dataset | Roboflow Universe |
| App Web | Streamlit |

## Dataset

- **Nombre:** Team Separation — Roboflow Universe v5
- **URL:** https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5
- **Tarea:** Detección de objetos (bounding boxes YOLO)
- **Clases (7):** `Ball`, `Corner`, `GoalKeeper`, `Goal_Net`, `Referee`, `TEAM 1`, `TEAM 2`
- **Imágenes:** 1.200 (640×640 px, broadcast de fútbol)

## Instrucciones para ejecutar localmente

```bash
git clone <url-del-repo>
cd ProyectoFinalRedesOffside
pip install -r requirements.txt
export ROBOFLOW_API_KEY="tu_api_key"
python data/download_dataset.py
jupyter notebook dev/01_dataset_preparation.ipynb
```

> En Windows, reemplazar `export` por `set`:
> ```cmd
> set ROBOFLOW_API_KEY=tu_api_key
> ```

## Estructura del repositorio

```
ProyectoFinalRedesOffside/
├── .gitignore
├── README.md
├── data/
│   ├── README.md               ← Descripción del dataset e instrucciones de descarga
│   ├── download_dataset.py     ← Script de descarga automática vía Roboflow
│   ├── train.csv               ← Anotaciones split train (una fila por bounding box)
│   ├── val.csv                 ← Anotaciones split val
│   └── test.csv                ← Anotaciones split test
├── dev/
│   ├── 01_dataset_preparation.ipynb
│   ├── augmentation_examples.png
│   ├── class_distribution.png
│   └── train_batch_sample.png
└── prod/
    └── .gitkeep                ← Reservado para la app web (Semana 4)
```
