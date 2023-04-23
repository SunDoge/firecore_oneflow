from oneflow import nn
from firecore_oneflow.model.base import BaseModel
from firecore.adapter import adapt, StrMapping
from typing import List


class BaseLoss(BaseModel):
    pass


nn.CrossEntropyLoss


class LossWrapper(nn.Module):
    def __init__(
        self,
        loss_fn: nn.Module,
        in_rules: StrMapping | None = None,
        out_name: str | None = None,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self._in_rules = in_rules
        self._out_name = out_name

    def forward(self, **kwargs):
        new_kwargs = adapt(kwargs, self._in_rules)
        loss = self.loss_fn(**new_kwargs)
        return {self._out_name: loss}
