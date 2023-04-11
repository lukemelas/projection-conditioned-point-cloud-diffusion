from config.structured import ProjectConfig
from .model import ConditionalPointCloudDiffusionModel
from .model_coloring import PointCloudColoringModel
from .model_utils import set_requires_grad


def get_model(cfg: ProjectConfig):
    model = ConditionalPointCloudDiffusionModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model


def get_coloring_model(cfg: ProjectConfig):
    model = PointCloudColoringModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model
