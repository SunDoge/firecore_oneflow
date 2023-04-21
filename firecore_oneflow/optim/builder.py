import oneflow as flow
from oneflow import nn, optim
from firecore.config.instantiate import instantiate


def build_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    cfg.params.model = model
    optimizer = instantiate(cfg)
    return optimizer


def get_params(model: nn.Module):
    return model.parameters()
