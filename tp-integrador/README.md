# TP Integrador — Clasificación de Animales con Redes Neuronales Profundas

Trabajo Práctico Integrador de la materia *Redes Neuronales Profundas (Ingeniería en Sistemas de Información)*.

El objetivo es entrenar una red neuronal de visión por computadora para clasificar grupos/especies de animales y desplegarla en una aplicación web interactiva.

## Integrantes

- Integrante 1
- Integrante 2
- Integrante 3

## Aplicación desplegada

[TBD — se completará en la Semana 4]

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python >= 3.9 |
| Deep Learning | PyTorch + Torchvision |
| Dataset | Roboflow |
| App Web | Streamlit |

## Dataset

- **Nombre:** Team Separation (Animals) — Roboflow Universe v5
- **URL:** https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5

## Instrucciones para ejecutar localmente

```bash
git clone <url-del-repo>
cd tp-integrador
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
tp-integrador/
├── .gitignore
├── README.md
├── data/
│   ├── README.md               ← Instrucciones de descarga del dataset
│   ├── download_dataset.py     ← Script de descarga automática
│   ├── train.csv               ← Rutas + etiquetas (split train)
│   ├── val.csv                 ← Rutas + etiquetas (split valid)
│   └── test.csv                ← Rutas + etiquetas (split test)
├── dev/
│   ├── 01_dataset_preparation.ipynb
│   ├── augmentation_examples.png
│   └── train_batch_sample.png
└── prod/
    └── .gitkeep                ← Carpeta reservada para la app web
```
