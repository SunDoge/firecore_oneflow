from oneflow import nn
from firecore_oneflow.optim.builder import build_optimizer
from firecore.config.instantiate import instantiate


class GraphBuilder:
    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ) -> None:
        self.model = model

        if optimizer_cfg:
            self.optimizer = build_optimizer(optimizer_cfg, self.model)
        else:
            self.optimizer = None

        if lr_scheduler_cfg:
            lr_scheduler_cfg.optimizer = self.optimizer
            self.lr_scheduler = instantiate(lr_scheduler_cfg)
        else:
            self.lr_scheduler = None

    def build_train_graph(self):
        pass

    def build_test_graph(self):
        pass
