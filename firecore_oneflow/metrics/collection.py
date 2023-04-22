from typing import Dict
from .base import BaseMetric


class MetricCollection:
    def __init__(self, metrics: Dict[str, BaseMetric]) -> None:
        self._metrics = metrics

    
