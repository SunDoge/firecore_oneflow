from .base import BaseHook
from firecore_oneflow.metrics.collection import MetricCollection
from firecore.meter import Meter
import logging

logger = logging.getLogger(__name__)


class TextLoggerHook(BaseHook):
    def __init__(
        self,
        stage: str,
        interval: int = 100,
    ) -> None:
        super().__init__()
        self._stage = stage
        self._interval = interval
        self._rate_meter = Meter()
        logger.info('init text logger with stage %s', stage)

    def before_epoch(self, **kwargs):
        self._rate_meter.reset()

    def after_iter(
        self,
        metrics: MetricCollection,
        batch_idx: int,
        epoch_length: int,
        batch_size: int,
        **kwargs,
    ):
        self._rate_meter.step(n=batch_size)

        if (batch_idx + 1) % self._interval == 0:
            metric_outputs = metrics.display()
            rate = self._rate_meter.rate
            prefix = f"{self._stage} {batch_idx + 1}/{epoch_length} {rate:.1f} spl/s"

            logger.info(f"{prefix} {metric_outputs}")
            print(metric_outputs)

    def after_epoch(
        self, epoch: int, metrics: MetricCollection, max_epochs: int, **kwargs
    ):
        metric_outputs = metrics.display()

        rate = self._rate_meter.rate

        prefix = f"{self._stage} {epoch + 1}/{max_epochs} {rate:.1f} spl/s"

        logger.info(f"{prefix} {metric_outputs}")
