from oneflow import Tensor
import oneflow as flow
from typing import Dict, Optional, Union
from firecore.adapter import adapt
import logging

logger = logging.getLogger(__name__)


class BaseMetric:
    def __init__(
        self,
        in_rules: Optional[Dict[str, str]] = None,
        out_rules: Optional[Dict[str, str]] = None,
    ) -> None:
        self._in_rules = in_rules
        self._out_rules = out_rules
        self._cached_result: Optional[Dict[str, Tensor]] = None

    def update(self, **kwargs):
        pass

    def compute(self):
        pass

    def reset(self):
        pass

    def update_adapted(self, **kwargs):
        self._cached_result = None
        new_kwargs = adapt(kwargs, self._in_rules)
        self.update(**new_kwargs)

    def compute_adapted(self):
        if self._cached_result is None:
            self._cached_result = self.compute()
        result = adapt(self._cached_result, self._out_rules)
        return result
