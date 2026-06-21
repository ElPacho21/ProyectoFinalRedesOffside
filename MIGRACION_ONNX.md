# Migración del backend a ONNX Runtime (deploy sin PyTorch)

Documenta el cambio del backend de inferencia de **PyTorch + Ultralytics** a
**ONNX Runtime + NumPy**, hecho para poder deployarlo en **Vercel** (serverless).

> Rama: `feat/onnx-inference`

---

## 1. Por qué

El backend hace inferencia con un modelo YOLO26m. El stack original
(`torch` + `torchvision` + `ultralytics`) pesa **~450–700 MB** descomprimido, muy
por encima del límite de tamaño de las funciones serverless de Vercel (250 MB
históricos, 500 MB ampliado). Ese stack **no entra**.

`onnxruntime` corre el mismo modelo entrenado con una fracción del peso. El
modelo se exporta una vez a ONNX y la inferencia se reimplementa en NumPy/OpenCV.

### Resultado de tamaño

| Componente | Tamaño aprox. |
|---|---|
| modelo.onnx | 78 MB |
| opencv-python-headless | ~55–70 MB (Linux) |
| onnxruntime | ~44 MB |
| numpy | ~34 MB |
| pillow | ~16 MB |
| pydantic + fastapi + uvicorn + transitivas | ~30 MB |
| **Bundle estimado** | **~260–340 MB** |

Queda **cómodo bajo los 500 MB** (~160–240 MB de margen). El deploy en Vercel es
viable.

---

## 2. El modelo es exportable sin pérdida

El `.pt` es un `WeightedDetectionModel` (definido en `dev/custom_models.py`), pero
**solo customiza la _loss_ de entrenamiento** (`init_criterion`). El forward / la
inferencia son los de un `DetectionModel` estándar de Ultralytics. Por lo tanto el
ONNX exportado captura el 100% del comportamiento de inferencia.

Además es un modelo **end-to-end (NMS-free)**: la salida ya son las detecciones
finales, sin necesidad de Non-Max Suppression.

```
INPUT  images:  [1, 3, 640, 640]
OUTPUT output0: [1, 300, 6]   →  300 detecciones × [x1, y1, x2, y2, conf, class]
```

Las coordenadas salen en el espacio del input con letterbox (640×640).

---

## 3. Cambios por archivo

| Archivo | Cambio |
|---|---|
| `backend/utils.py` | `cargar_modelo` ahora crea una `onnxruntime.InferenceSession`. Nuevos helpers `_letterbox` y `_inferir` (preproceso + decode). `detectar_objetos` usa `_inferir` en vez de `model.predict`. Se quitó el hack de importar `WeightedDetectionModel`. |
| `backend/main.py` | Carga del modelo **lazy** (`get_model()`), porque en serverless los eventos `startup`/lifespan no están garantizados. Se quitaron los guards `503` que dependían de la precarga. |
| `backend/hough.py` | Se eliminaron `print()` de debug (rompían en consola Windows cp1252 por caracteres Unicode, y eran ruido en logs). Lógica intacta. |
| `backend/requirements.txt` | Fuera `torch`, `torchvision`, `ultralytics`. Dentro `onnxruntime==1.27.0`. |
| `backend/.env` | `MODEL_PATH=modelo.onnx`. |
| `backend/vercel.json` | Entrypoint `main.py` (imports más robustos que `api/index.py`) + `includeFiles: modelo.onnx`. |
| `backend/modelo.onnx` | Modelo exportado (78 MB). **Nuevo, versionado.** |
| `backend/modelo.pt` | **Eliminado** (42 MB, ya no se usa). |
| `backend/pyproject.toml`, `uv.toml`, `api/index.py` | **Eliminados** (eran workarounds del índice de PyTorch / del entrypoint viejo). |
| `.gitignore` | Excepción `!backend/modelo.onnx` (el `*.onnx` global lo ignoraba). |

> La lógica de offside (`calcular_offside`), el clustering por color, el dibujo de
> resultados y la detección de punto de fuga (`hough.py`) **no cambiaron** — ya eran
> NumPy/OpenCV puro.

---

## 4. Cómo funciona la inferencia ahora

`utils._inferir(modelo, img_rgb, min_conf)`:

1. **Flip RGB → BGR.** Ultralytics, al recibir un array NumPy en `predict()`, lo
   trata como BGR (hace `im[..., ::-1]` internamente). El modelo fue entrenado/usado
   viendo BGR, así que replicamos ese orden de canales. **Crítico:** sin el flip se
   **invierten TEAM 1 ↔ TEAM 2** (validado, ver §5).
2. **Letterbox** a 640×640 manteniendo aspect ratio, padding gris (114), réplica de
   `LetterBox(auto=False)` de Ultralytics.
3. **Inferencia** ONNX → `(300, 6)`.
4. **Filtro** por `min_conf` y **des-letterbox** de las cajas a coordenadas de la
   imagen original (con clip a los bordes).

El resto (umbral por clase, _keep-best-only_ para Ball/Goal_Net, clustering) queda
igual que antes.

---

## 5. Validación de paridad

`dev/validate_onnx.py` compara el pipeline original (Ultralytics `.pt`) contra el
nuevo (`onnxruntime`) sobre la misma imagen:

| Caso | Resultado (feed BGR) |
|---|---|
| Imagen 640×640 nativa | **35/35 detecciones, error de caja 0.00 px, error de conf 0.0000** (bit-exact) |
| Imagen reescalada 1280×720 | 32/32 detecciones, error < 1.2 px (interpolación del resize) |

El feed RGB (sin flip) daba error de 270+ px y **equipos invertidos** → confirmó que
hay que replicar el flip BGR.

`dev/test_api.py` valida el flujo HTTP completo (`/api/health` + `/api/detect`) con
`TestClient`: 200 OK, detecciones correctas, imagen anotada.

---

## 6. Re-exportar el modelo (si se reentrena)

Si se entrena un `.pt` nuevo, regenerar el ONNX con el entorno de desarrollo
(la venv raíz, que tiene `torch` + `ultralytics` + `onnx`):

```bash
# coloca el nuevo .pt en dev/modelo.pt y luego:
.venv/Scripts/python.exe dev/export_to_onnx.py
```

El script registra `WeightedDetectionModel` para el des-pickle, exporta a
`imgsz=640, dynamic=False, simplify=True` y copia el resultado a
`backend/modelo.onnx`. Después conviene correr la validación:

```bash
.venv/Scripts/python.exe dev/validate_onnx.py data/raw/test/images/<alguna>.jpg
```

---

## 7. Deploy en Vercel

- **Entrypoint:** `backend/main.py` (expone `app`, ASGI). Todas las rutas se enrutan
  ahí vía `vercel.json`.
- **Python:** fijado a `3.12` (`backend/.python-version`) para asegurar wheels.
- **Modelo:** incluido en el bundle con `includeFiles: "modelo.onnx"`. En runtime se
  resuelve como `Path(__file__).parent / "modelo.onnx"` (junto a `utils.py`).
- **Carga lazy:** el modelo se carga en el primer request (no depende de `startup`).

### Notas

- `modelo.onnx` pesa 78 MB: GitHub lo acepta (< 100 MB) pero muestra un *warning* de
  archivo grande (> 50 MB) al pushear. Funciona igual.
- Cold start: cada arranque en frío recarga el modelo (~78 MB) y la inferencia en CPU
  tarda ~1–3 s por imagen. Vigilar el límite de duración de la función (10 s en Hobby).

---

## 8. Scripts de desarrollo (`dev/`)

| Script | Para qué |
|---|---|
| `export_to_onnx.py` | Exporta `dev/modelo.pt` → `backend/modelo.onnx`. |
| `validate_onnx.py` | Compara paridad Ultralytics vs ONNX en una imagen. |
| `smoke_test_backend.py` | Smoke test del pipeline integrado (sin torch). |
| `test_api.py` | Test del API FastAPI (`/api/health` + `/api/detect`). |

> Estos scripts usan `torch`/`ultralytics` y corren solo en el entorno de desarrollo,
> no en el deploy.

---

## 9. Cómo volver atrás

El backend con torch sigue en el historial de git (antes de esta rama). Para
revertir: restaurar `requirements.txt`, `utils.py`, `main.py` y `MODEL_PATH` a la
versión previa, y volver a usar `dev/modelo.pt`.
