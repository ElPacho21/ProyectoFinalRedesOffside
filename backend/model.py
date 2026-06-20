import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prod.utils import (  # noqa: E402
    cargar_modelo,
    detectar_objetos,
    calcular_offside,
    dibujar_resultado,
    punto_de_fuga_desde_puntos,
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = cargar_modelo()
    return _model


__all__ = [
    "get_model",
    "detectar_objetos",
    "calcular_offside",
    "dibujar_resultado",
    "punto_de_fuga_desde_puntos",
]
