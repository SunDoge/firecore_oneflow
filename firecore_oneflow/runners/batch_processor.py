from typing import Optional, List, Dict, Union
import oneflow as flow
from oneflow import Tensor
from firecore.adapter import adapt
import pysnooper


class BatchProcessor:
    def __init__(
        self,
        # device: flow.device = flow.device("cpu"),
        placement: flow.placement = flow.placement("cpu", [0]),
        sbp: flow.sbp = flow.sbp.broadcast,
        names: Optional[List[str]] = None,
        batch_size_key: Optional[str] = None,
        batch_size_index: int = 0,
        rules: Optional[Dict[str, str]] = None,
    ) -> None:
        self._placement = placement
        self._sbp = sbp
        self._names = names
        self._batch_size_key = batch_size_key
        self._batch_size_index = batch_size_index
        self._rules = rules

    def rename(
        self, batch: Union[List[Tensor], Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        if self._names:
            batch = self.name_batch(batch)

        if self._rules is not None:
            batch = adapt(batch, self._rules)

        return batch

    def name_batch(self, batch: List[Tensor]):
        assert isinstance(batch, list)
        assert len(self._names) == len(batch)
        return {k: v for k, v in zip(self._names, batch)}

    def get_batch_size(self, batch: Dict[str, Tensor]):
        if self._batch_size_key:
            tensor = batch[self._batch_size_key]
        else:
            tensor = next(iter(batch.values()))
        batch_size = tensor.size(self._batch_size_index)
        return batch_size

    # def copy_host_to_device(self, batch: Dict[str, Tensor]):
    #     return {k: v.to(self._device) for k, v in batch.items()}

    def to_global(self, batch: Dict[str, Tensor]):
        return {
            k: v.to_global(placement=self._placement, sbp=self._sbp)
            for k, v in batch.items()
        }
