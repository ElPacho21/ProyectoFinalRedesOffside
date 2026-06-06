# Dataset — Team Separation (Fútbol)

## 1. Descripción del dataset

**Team Separation** es un dataset de detección de objetos en partidos de fútbol, disponible en Roboflow Universe. Contiene imágenes de broadcast (640×640 px) con anotaciones en formato YOLO para 7 clases: jugadores de dos equipos, arquero, árbitro, pelota, redes y córners. El objetivo es entrenar un detector que localice cada objeto con bounding boxes — información usada para determinar la línea de offside.

## 2. Fuente y enlace

- **Plataforma:** Roboflow Universe
- **URL directa (versión 5):** https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5
- **Workspace:** `animals-67mq4`
- **Proyecto:** `team-separation`
- **Versión utilizada:** 5

## 3. Licencia

Figura en la página de Roboflow del proyecto: https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5

## 4. Clases

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | `Ball` | Pelota |
| 1 | `Corner` | Banderín de córner |
| 2 | `GoalKeeper` | Arquero |
| 3 | `Goal_Net` | Red del arco |
| 4 | `Referee` | Árbitro |
| 5 | `TEAM 1` | Jugadores equipo 1 |
| 6 | `TEAM 2` | Jugadores equipo 2 |

## 5. Estadísticas

| Split | Imágenes | Anotaciones | Proporción |
|-------|----------|-------------|------------|
| Train | 1.050    | 21.633      | 87,5 %     |
| Val   | 100      | 1.999       | 8,3 %      |
| Test  | 50       | 992         | 4,2 %      |
| **Total** | **1.200** | **24.624** | **100 %** |

## 6. Cómo obtener la API Key de Roboflow

1. Registrarse gratis en [roboflow.com](https://roboflow.com).
2. Iniciar sesión → menú de usuario (esquina superior derecha).
3. Ir a **Settings → API**.
4. Copiar la **Private API Key**.
5. Exportarla como variable de entorno antes de ejecutar el script.

## 7. Instrucciones de descarga

```bash
# 1. Instalar el paquete de Roboflow
pip install roboflow

# 2. Exportar la API key como variable de entorno
export ROBOFLOW_API_KEY="tu_api_key_aqui"
# En Windows CMD:        set ROBOFLOW_API_KEY=tu_api_key_aqui
# En Windows PowerShell: $env:ROBOFLOW_API_KEY="tu_api_key_aqui"

# 3. Ejecutar el script de descarga
python data/download_dataset.py
```

El script detecta si el dataset ya fue descargado y evita repetir la descarga.

## 8. Estructura de carpetas resultante

Tras ejecutar `download_dataset.py`, la carpeta `data/raw/` tendrá la siguiente estructura:

```
data/raw/
├── data.yaml          ← nombres de clases y rutas por split
├── train/
│   ├── images/        ← imágenes JPG 640×640
│   └── labels/        ← un .txt por imagen con anotaciones YOLO
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Formato de cada línea de anotación:** `class_id x_center y_center width height`
Coordenadas normalizadas en [0, 1]. Una imagen puede tener múltiples líneas (múltiples objetos).

> Las carpetas `data/raw/` están excluidas del control de versiones por `.gitignore`. Solo se versionan los CSVs (`train.csv`, `val.csv`, `test.csv`).
