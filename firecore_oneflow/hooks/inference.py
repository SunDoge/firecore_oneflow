from .base import BaseHook
from oneflow import nn
import oneflow as flow


class InferenceHook(BaseHook):
    def __init__(self) -> None:
        super().__init__()

    def before_epoch(self, model: nn.Module, **kwargs):
        flow.set_grad_enabled(False)
        model.eval()
