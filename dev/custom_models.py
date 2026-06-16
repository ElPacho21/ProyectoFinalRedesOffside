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

        return WeightedE2ELoss(self)


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
