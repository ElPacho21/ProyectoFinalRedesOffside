import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


# ============================================================
# PESOS MANUALES
# ============================================================

MANUAL_WEIGHTS = torch.tensor(
    [2.5, 0.4, 1.2, 0.6, 0.4, 1.0, 1.0],
    dtype=torch.float32,
)

MANUAL_WEIGHTS = MANUAL_WEIGHTS / MANUAL_WEIGHTS.mean()


# ============================================================
# LOSS CON BCE PONDERADA
# ============================================================

class WeightedDetectionLoss(v8DetectionLoss):

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        print("USANDO WEIGHTED BCE")

        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=model.class_weights.to(self.device),
            reduction="none",
        )


# ============================================================
# E2E LOSS PERSONALIZADA
# ============================================================

class WeightedE2ELoss(E2ELoss):

    def __init__(self, model):
        print("USANDO WEIGHTED E2E LOSS")

        super().__init__(
            model,
            loss_fn=WeightedDetectionLoss,
        )


# ============================================================
# MODELO PERSONALIZADO
# ============================================================

class WeightedDetectionModel(DetectionModel):

    class_weights = None

    def init_criterion(self):

        print("INIT CRITERION CUSTOM")

        orig_criterion = super().init_criterion()
        if isinstance(orig_criterion, E2ELoss):
            return WeightedE2ELoss(self)
        else:
            return WeightedDetectionLoss(self)


# ============================================================
# TRAINER
# ============================================================

class ManualWeightedTrainer(DetectionTrainer):

    class_weights = MANUAL_WEIGHTS

    def get_model(self, cfg=None, weights=None, verbose=True):

        model = WeightedDetectionModel(
            cfg,
            nc=self.data["nc"],
            verbose=verbose,
        )

        if weights:
            model.load(weights)

        model.class_weights = self.class_weights

        return model


# ============================================================
# BUILDER DE TRAINER CON PESOS DINÁMICOS
# ============================================================

def build_weighted_trainer(class_counts, device: str = "cuda"):
    """
    Construye una subclase de DetectionTrainer con class weights calculados
    a partir de la frecuencia de cada clase.
    """
    if isinstance(class_counts, dict):
        counts = [class_counts[i] for i in sorted(class_counts)]
    else:
        counts = list(class_counts)

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    total = counts_tensor.sum()
    n_classes = len(counts_tensor)
    weights = total / (n_classes * counts_tensor)
    weights = weights / weights.mean()

    print(f"Class weights calculados ({n_classes} clases):")
    for i, w in enumerate(weights.tolist()):
        print(f"  clase {i}: count={counts[i]:6d}  weight={w:.4f}")

    class DynamicWeightedTrainer(ManualWeightedTrainer):
        class_weights = weights

    return DynamicWeightedTrainer


# ============================================================
# INYECCIÓN EN __main__ PARA DESERIALIZACIÓN (PICKLE)
# ============================================================

# PyTorch guarda referencias relativas al módulo __main__ si las clases se definieron originalmente
# en un notebook. Al cargar los modelos (.pt), se buscan estas clases en __main__.
# Las inyectamos aquí de forma dinámica para evitar el error "module '__main__' has no attribute ...".
try:
    import sys
    main_module = sys.modules['__main__']
    main_module.WeightedDetectionModel = WeightedDetectionModel
    main_module.WeightedE2ELoss = WeightedE2ELoss
    main_module.WeightedDetectionLoss = WeightedDetectionLoss
    main_module.ManualWeightedTrainer = ManualWeightedTrainer
except Exception as e:
    pass


