from oneflow import nn
from firecore.adapter import adapt, StrMapping
from typing import Optional, List


class BaseModel(nn.Module):
    def __init__(
        self,
        in_rules: Optional[StrMapping] = None,
        out_rules: Optional[StrMapping] = None,
    ):
        super().__init__()
        self._in_rules = in_rules
        self._out_rules = out_rules

    def __call__(self, **kwargs):
        new_kwargs = adapt(kwargs, self._in_rules)
        outputs = super().__call__(**new_kwargs)
        new_outputs = adapt(outputs, self._out_rules)
        return new_outputs


class ModelWrapper(BaseModel):
    def __init__(
        self,
        model: nn.Module,
        in_rules: StrMapping | None = None,
        out_names: List[str] | None = None,
        out_rules: StrMapping | None = None,
    ):
        super().__init__(in_rules, out_rules)
        self.model = model
        self._out_names = out_names

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        if self._out_names is not None:
            if len(self._out_names) == 1:
                outputs = [outputs]
            new_outputs = {k: v for k, v in zip(self._out_names, outputs)}
            return new_outputs
        else:
            return outputs
