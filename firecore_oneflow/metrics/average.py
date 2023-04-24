from .base import BaseMetric
from typing import Dict
import oneflow as flow
from oneflow import Tensor


class Average(BaseMetric):
    def __init__(
        self,
        fmt: str = ".4f",
        in_rules: Dict[str, str] | None = None,
        out_rules: Dict[str, str] | None = None,
    ) -> None:
        super().__init__(fmt, in_rules, out_rules)

        self.register_buffer("sum", flow.tensor(0, dtype=flow.float))
        self.register_buffer("count", flow.tensor(0, dtype=flow.long))
        self.register_buffer("val", flow.tensor(0, dtype=flow.float))

    def update(self, val: Tensor, n: int, **kwargs):
        self.val.copy_(val)
        self.sum.add_(val)
        self.count.add_(n)

    def compute(self):
        avg = self.sum / self.count
        return {"avg": avg, "val": self.val}

    def reset(self):
        self.val.fill_(0)
        self.sum.fill_(0)
        self.count.fill_(0)
