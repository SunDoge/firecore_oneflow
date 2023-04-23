from .base import BaseRunner
from typing import List
from firecore_oneflow.hooks.base import BaseHook
from .batch_processor import BatchProcessor
from icecream import ic

class EpochBasedRunner(BaseRunner):

    def __init__(self, hooks: List[BaseHook], **kwargs) -> None:
        super().__init__(hooks, **kwargs)
        ic([(k, type(v)) for k,v in kwargs.items()])

    def step(self, epoch: int):
        return super().step(epoch)

    def start_loop(
        self,
        data_source,
        batch_processor: BatchProcessor,
        forward_fn,
        metrics,
        epoch: int
    ):

        self.call_hook(
            'before_iter',
            epoch=epoch,
        )

        for batch_idx, batch in enumerate(data_source):
            self.call_method(
                'before_iter',
                epoch=epoch,
                batch_idx=batch_idx,
            )

            batch_as_dict = batch_processor.rename(batch)
            batch_size = batch_processor.get_batch_size(batch_as_dict)
            batch_on_device = batch_processor.copy_host_to_device(
                batch_as_dict)

            self.call_hook(
                'before_forward',
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                **batch_on_device,
            )

            outputs, losses = self.call_method(
                forward_fn,
                **batch_on_device
            )

            self.call_hook(
                'after_forward',
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                **batch_on_device,
                **outputs,
                **losses
            )

            