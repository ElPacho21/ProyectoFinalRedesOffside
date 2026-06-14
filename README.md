# Detección de Jugadores para Offside en Fútbol

Trabajo Práctico Integrador de la materia *Redes Neuronales Profundas (Ingeniería en Sistemas de Información)*.

El objetivo es entrenar un modelo de detección de objetos (YOLOv8s) sobre imágenes de broadcast de fútbol para localizar jugadores, árbitros y la pelota mediante bounding boxes, información necesaria para determinar la línea de offside.

## Integrantes

- Facundo Pacci
- Nicolás Ocampo
- Valentino Isgro
- Bruno Lucero
- Juan Pablo Costa

## Aplicación desplegada

[TBD — completar con la URL de Streamlit Cloud luego del deploy]

> Para desplegar en Streamlit Cloud: conectar el repo en https://share.streamlit.io,
> seleccionar `prod/app.py` como archivo principal.

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

### App web (Semana 4)

```bash
git clone <url-del-repo>
cd ProyectoFinalRedesOffside
pip install -r prod/requirements.txt
streamlit run prod/app.py
```

### Notebooks de entrenamiento (Semanas 1-3)

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY="tu_api_key"   # Windows: set ROBOFLOW_API_KEY=tu_api_key
python data/download_dataset.py
jupyter notebook dev/01_dataset_preparation.ipynb
```

### Herramienta de anotación manual (VP + ground truth de offside)

```bash
# Cada integrante corre con su ID (0-4):
python dev/annotate_vp.py --integrante 0

# Con comparación contra el algoritmo:
python dev/annotate_vp.py --integrante 0 --comparar-algoritmo
```

## Estructura del repositorio

```
ProyectoFinalRedesOffside/
├── .gitignore
├── README.md
├── requirements.txt            ← Dependencias para notebooks de entrenamiento
├── data/
│   ├── README.md
│   ├── download_dataset.py
│   ├── vp_annotations.csv      ← Anotaciones manuales de VP + ground truth offside
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── dev/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── HoughLines.py           ← Detección de punto de fuga (Hough + manual)
│   ├── annotate_vp.py          ← Herramienta de anotación manual VP + offside GT
│   └── YOLOv8m Weighted/
│       └── exp3_yolov8m_weighted.pt  ← Modelo entrenado (~52 MB)
└── prod/
    ├── app.py                  ← Interfaz Streamlit
    ├── utils.py                ← Lógica: modelo, inferencia, geometría de offside
    └── requirements.txt        ← Dependencias para la app web (versiones fijadas)
```
