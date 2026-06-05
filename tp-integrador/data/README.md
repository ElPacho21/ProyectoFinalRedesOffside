# Dataset — Team Separation (Animals)

## 1. Descripción del dataset

**Team Separation (Animals)** es un dataset de clasificación de imágenes de animales disponible en Roboflow Universe. Contiene fotografías de distintas especies/grupos de animales capturadas en diversas condiciones de iluminación y entorno. El objetivo es entrenar un modelo que distinga automáticamente entre las diferentes clases de animales presentes.

## 2. Fuente y enlace

- **Plataforma:** Roboflow Universe
- **URL directa (versión 5):** https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5
- **Workspace:** `animals-67mq4`
- **Proyecto:** `team-separation`
- **Versión utilizada:** 5

## 3. Licencia

La licencia del dataset figura en la página de Roboflow del proyecto. Consultarla en: https://universe.roboflow.com/animals-67mq4/team-separation/dataset/5

## 4. Estadísticas

| Split | Imágenes | Porcentaje |
|-------|----------|------------|
| Train | 1 050    | 88 %       |
| Valid | 100      | 8 %        |
| Test  | 50       | 4 %        |
| **Total** | **1 200** | **100 %** |

- **Número de clases:** se descubre dinámicamente al ejecutar `download_dataset.py` (ver notebook `dev/01_dataset_preparation.ipynb`).

## 5. Cómo obtener la API Key de Roboflow

1. Registrarse gratis en [roboflow.com](https://roboflow.com).
2. Iniciar sesión y abrir el menú de usuario (esquina superior derecha).
3. Ir a **Settings → API** (o directamente a https://app.roboflow.com/settings/api).
4. Copiar la **Private API Key**.
5. Exportarla como variable de entorno antes de ejecutar el script (ver sección siguiente).

## 6. Instrucciones de descarga (paso a paso)

```bash
# 1. Instalar el paquete de Roboflow
pip install roboflow

# 2. Exportar la API key como variable de entorno
export ROBOFLOW_API_KEY="tu_api_key_aqui"
# En Windows CMD:  set ROBOFLOW_API_KEY=tu_api_key_aqui
# En Windows PowerShell:  $env:ROBOFLOW_API_KEY="tu_api_key_aqui"

# 3. Ejecutar el script de descarga
python data/download_dataset.py
```

El script detecta si el dataset ya fue descargado y evita repetir la descarga.

## 7. Estructura de carpetas resultante

Tras ejecutar `download_dataset.py`, la carpeta `data/raw/` tendrá la siguiente estructura:

```
data/raw/
├── train/
│   ├── <clase_1>/
│   │   ├── imagen1.jpg
│   │   └── ...
│   └── <clase_N>/
│       └── ...
├── valid/
│   └── <clases>/
└── test/
    └── <clases>/
```

> Las carpetas `data/raw/` están excluidas del control de versiones por `.gitignore`. Solo se versionan los archivos CSV (`train.csv`, `val.csv`, `test.csv`).
