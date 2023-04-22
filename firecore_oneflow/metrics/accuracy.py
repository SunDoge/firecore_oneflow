from .base import BaseMetric
from typing import List, Optional, Dict


class Accuracy(BaseMetric):
    def __init__(
        self,
        topk: List[int],
        **kwargs,
    ) -> None:
        super().__init__()

        self._topk = topk
