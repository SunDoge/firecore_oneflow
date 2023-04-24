from oneflow import Tensor
import oneflow as flow
from typing import Dict, Optional, Union
from firecore.adapter import adapt
import logging
from oneflow import nn
import oneflow as flow

logger = logging.getLogger(__name__)


class BaseMetric(nn.Module):
    def __init__(
        self,
        fmt: str = ".4f",
        in_rules: Optional[Dict[str, str]] = None,
        out_rules: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self._fmt = fmt
        self._in_rules = in_rules
        self._out_rules = out_rules
        self._cached_result: Optional[Dict[str, Tensor]] = None

    def update(self, **kwargs):
        pass

    def compute(self):
        pass

    def reset(self):
        pass

    @flow.no_grad()
    def update_adapted(self, **kwargs):
        # print(kwargs.keys())
        self._cached_result = None
        new_kwargs = adapt(kwargs, self._in_rules)
        self.update(**new_kwargs)

    @flow.no_grad()
    def compute_adapted(self):
        if self._cached_result is None:
            self._cached_result = self.compute()
        result = adapt(self._cached_result, self._out_rules)
        return result

    def display(self):
        result = self.compute_adapted()
        tmpl = "{:" + self._fmt + "}"
        new_out = {k: tmpl.format(v) for k, v in result.items()}
        return new_out
