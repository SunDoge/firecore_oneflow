from .base import BaseHook
import oneflow as flow


class TrainingHook(BaseHook):
    def __init__(self) -> None:
        super().__init__()

    def before_epoch(self, **kwargs):
        flow.set_grad_enabled(True)
