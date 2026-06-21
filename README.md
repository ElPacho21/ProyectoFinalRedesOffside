# Detección de Jugadores para Offside en Fútbol

Trabajo Práctico Integrador de la materia *Redes Neuronales Profundas (Ingeniería en Sistemas de Información)*.

El objetivo es entrenar un modelo de detección de objetos (YOLOv26m) sobre imágenes de broadcast de fútbol para localizar jugadores, árbitros y la pelota mediante bounding boxes, información necesaria para determinar la línea de offside.

## Integrantes

- Facundo Pacci
- Nicolás Ocampo
- Valentino Isgro
- Bruno Lucero
- Juan Pablo Costa

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python >= 3.9 |
| Deep Learning | PyTorch + YOLOv26 (Ultralytics) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Dataset | Roboflow Universe |

## Dataset

- **Nombre:** Team Separation — Roboflow Universe v5
- **URL:** https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5
- **Tarea:** Detección de objetos (bounding boxes YOLO)
- **Clases (7):** `Ball`, `Corner`, `GoalKeeper`, `Goal_Net`, `Referee`, `TEAM 1`, `TEAM 2`
- **Imágenes:** 1.200 (640×640 px, broadcast de fútbol)

## Instrucciones para ejecutar localmente

### Requisitos previos

- Python >= 3.9
- Node.js >= 18
- Modelo entrenado en `dev/modelo.pt`

### Backend (FastAPI)

> Crear un archivo `.env` en `backend/` con el contenido de `.env.example`. Ajustar `MODEL_PATH` según la ubicación del modelo.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

El backend queda disponible en `http://localhost:8000`.

### Frontend (React + Vite)

Crear un archivo `.env` en `frontend/` copiando el contenido de `.env.example`. Ajustar `VITE_API_URL` si el backend corre en otro puerto o host.

```bash
cd frontend
npm install
npm run dev
```

La app queda disponible en `http://localhost:5173`.

### Variables de entorno (backend)

| Variable | Descripción | Default |
|---|---|---|
| `MODEL_PATH` | Ruta al archivo `.pt` del modelo | `../dev/modelo.pt` |
| `CORS_ORIGINS` | Orígenes permitidos | `http://localhost:5173` |
| `HOST` | Host del servidor | `0.0.0.0` |
| `PORT` | Puerto del servidor | `8000` |
| `CONF_BALL` | Umbral de confianza para la pelota | `0.05` |
| `CONF_GOAL_NET` | Umbral de confianza para la red del arco | `0.05` |
| `CONF_TEAM` | Umbral de confianza para jugadores | `0.30` |

## Estructura del repositorio

```
ProyectoFinalRedesOffside/
├── README.md
├── backend/
│   ├── main.py               ← API FastAPI (endpoints detect, calculate-offside, etc.)
│   ├── utils.py              ← Inferencia YOLO + lógica de offside
│   ├── hough.py              ← Detección de punto de fuga (LSD + RANSAC)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx           ← Flujo principal (4 pasos)
│   │   ├── components/       ← Step1Upload, Step2TeamConfig, Step3VP, Step4Results
│   │   └── api/client.js
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── download_dataset.py
│   └── vp_annotations.csv
└── dev/
    ├── 01_dataset_preparation.ipynb
    ├── 02_model_training.ipynb
    └── modelo.pt             ← Modelo final
```

## Notebooks de entrenamiento

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY="tu_api_key"   # Windows: set ROBOFLOW_API_KEY=tu_api_key
python data/download_dataset.py
jupyter notebook dev/01_dataset_preparation.ipynb
```

cd backend

# Crear el entorno virtual
python -m venv .venv

# Activarlo
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Levantar el servidor
uvicorn main:app --reload --host 0.0.0.0 --port 8000