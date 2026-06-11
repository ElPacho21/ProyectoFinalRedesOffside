"""
Custom DetectionTrainer que aplica class weights en la componente BCE
de la classification loss para mitigar class imbalance.

Uso:
    from src.weighted_trainer import build_weighted_trainer
    trainer_cls = build_weighted_trainer(class_counts, device="cuda")
    model.train(data=..., trainer=trainer_cls)

class_counts: dict o lista con cantidad de instancias por class_id,
              en el mismo orden que data.yaml["names"].
"""

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss


class _WeightedDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss con pos_weight por clase en la BCE de clasificación."""

    def __init__(self, model, class_weights: torch.Tensor):
        super().__init__(model)
        # Reemplazar BCEWithLogitsLoss con pos_weight
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=class_weights.to(next(model.parameters()).device),
            reduction="none",
        )


class _WeightedDetectionTrainer(DetectionTrainer):
    _class_weights: torch.Tensor = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        return model

    def get_validator(self):
        return super().get_validator()

    def _setup_criterion(self):
        criterion = _WeightedDetectionLoss(self.model, self._class_weights)
        return criterion


def build_weighted_trainer(class_counts, device: str = "cuda"):
    """
    Construye una subclase de DetectionTrainer con class weights calculados
    a partir de la frecuencia de cada clase.

    Args:
        class_counts: lista o dict {class_id: count} con instancias por clase.
        device: "cuda" o "cpu".

    Returns:
        Subclase de DetectionTrainer lista para pasar a model.train(trainer=...).
    """
    if isinstance(class_counts, dict):
        counts = [class_counts[i] for i in sorted(class_counts)]
    else:
        counts = list(class_counts)

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    # Inverse frequency: clases raras reciben peso mayor
    # w_i = total / (n_classes * count_i)
    total = counts_tensor.sum()
    n_classes = len(counts_tensor)
    weights = total / (n_classes * counts_tensor)
    # Normalizar para que media = 1 (no cambia la escala global del loss)
    weights = weights / weights.mean()

    print(f"Class weights calculados ({n_classes} clases):")
    for i, w in enumerate(weights.tolist()):
        print(f"  clase {i}: count={counts[i]:6d}  weight={w:.4f}")

    class WeightedTrainer(_WeightedDetectionTrainer):
        _class_weights = weights

    return WeightedTrainer
