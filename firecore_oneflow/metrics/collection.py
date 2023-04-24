from typing import Dict
from .base import BaseMetric
import oneflow as flow


class MetricCollection(BaseMetric):
    def __init__(self, metrics: Dict[str, BaseMetric], **kwargs) -> None:
        super().__init__(**kwargs)
        self._metrics = metrics
        print(metrics)

    def update(self, **kwargs):
        for metric in self._metrics.values():
            metric.update_adapted(**kwargs)

    def compute(self):
        outputs = {}
        for metric in self._metrics.values():
            outputs.update(metric.compute_adapted())
        return outputs

    def reset(self):
        for metric in self._metrics.values():
            metric.reset()

    def display(self):
        outputs = {}
        for metric in self._metrics.values():
            outputs.update(metric.display())
        return outputs
