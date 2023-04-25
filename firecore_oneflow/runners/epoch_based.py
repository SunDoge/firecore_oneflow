from .base import BaseRunner
from typing import List
from firecore_oneflow.hooks.base import BaseHook
from .batch_processor import BatchProcessor
from icecream import ic
from firecore_oneflow.metrics.collection import MetricCollection
import oneflow as flow


class EpochBasedRunner(BaseRunner):
    def __init__(self, hooks: List[BaseHook], **kwargs) -> None:
        super().__init__(hooks, **kwargs)
        ic([(k, type(v)) for k, v in kwargs.items()])

    def step(self, epoch: int):
        self.call_method(self.start_loop, epoch=epoch)

    def start_loop(
        self,
        data_source,
        batch_processor: BatchProcessor,
        forward_fn,
        metrics: MetricCollection,
        epoch: int,
        **kwargs
    ):
        epoch_length = len(data_source)
        self.call_hook(
            "before_epoch",
            epoch=epoch,
            epoch_length=epoch_length,
        )

        for batch_idx, batch in enumerate(data_source):
            self.call_hook(
                "before_iter",
                epoch=epoch,
                batch_idx=batch_idx,
                epoch_length=epoch_length,
            )

            batch_as_dict = batch_processor.rename(batch)
            batch_size = batch_processor.get_batch_size(batch_as_dict)
            batch_on_device = batch_processor.to_global(batch_as_dict)

            self.call_hook(
                "before_forward",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                epoch_length=epoch_length,
                **batch_on_device,
            )

            outputs, losses = self.call_method(forward_fn, **batch_on_device)

            self.call_hook(
                "after_forward",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                epoch_length=epoch_length,
                **batch_on_device,
                **outputs,
                **losses,
            )

            metrics.update_adapted(
                **batch_on_device,
                **outputs,
                **losses,
                batch_size=batch_size,
                epoch_length=epoch_length,
            )

            self.call_hook(
                "after_iter",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                epoch_length=epoch_length,
                **batch_on_device,
                **outputs,
                **losses,
            )

        self.call_hook(
            "after_epoch",
            epoch=epoch,
            epoch_length=epoch_length,
        )
