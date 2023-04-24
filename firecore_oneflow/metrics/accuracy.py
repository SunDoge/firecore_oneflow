from .base import BaseMetric
from typing import List, Optional, Dict
import oneflow as flow
from oneflow import Tensor
from . import functional as F
import pysnooper


class Accuracy(BaseMetric):
    def __init__(
        self,
        topk: List[int],
        **kwargs,
    ) -> None:
        super().__init__()

        self._topk = topk

        self.register_buffer("count", flow.tensor(0, dtype=flow.long))
        self.register_buffer("sum", flow.zeros(len(topk), dtype=flow.float))

    # @pysnooper.snoop()
    def update(self, output: Tensor, target: Tensor, **kwargs):
        batch_size = target.size(0)

        corrects = F.topk_correct(output, target, topk=self._topk)
        item = flow.stack(corrects)

        self.sum.add_(item)
        self.count.add_(batch_size)

    def compute(self):
        result = {}

        acc = self.sum / self.count

        for i, k in enumerate(self._topk):
            result[f"acc{k}"] = acc[i]

        return result

    def reset(self):
        self.count.fill_(0)
        self.sum.fill_(0)


if __name__ == "__main__":
    acc = Accuracy([1, 5])
    output = flow.rand(4, 10) + flow.eye(4, 10) * 2.0
    output[:, 0] = 1.0
    target = flow.zeros(4, dtype=flow.long)
    acc.update(output, target)
    print(acc.display())
    acc.reset()
    # acc = acc.to_global(flow.placement("cuda", [0]), sbp=flow.sbp.partial_sum)
    
