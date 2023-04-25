from .cifar_resnet import cifar_resnet20
from firecore.config import LazyCall
from flowvision import transforms as T
from oneflow import nn, optim
from firecore_oneflow.optim.builder import get_params
from oneflow.utils.data import DataLoader
from flowvision.datasets.cifar import CIFAR10
from firecore_oneflow.runners.epoch_based import EpochBasedRunner
from firecore_oneflow.model.builder import GraphBuilder
from firecore_oneflow import hooks
from firecore_oneflow.runners.batch_processor import BatchProcessor
from firecore_oneflow.metrics.collection import MetricCollection
from firecore_oneflow.metrics.accuracy import Accuracy
from firecore_oneflow.metrics.average import Average
from firecore_oneflow.losses import LossWrapper
from firecore_oneflow.model.base import ModelWrapper
from omegaconf import DictConfig


hparams = DictConfig(
    dict(
        max_epochs=200,
    )
)

model = LazyCall(ModelWrapper)(
    model=LazyCall(cifar_resnet20)(
        num_classes=10,
    ),
    in_rules={"image": "x"},
    out_names=["output"],
)
criterion = LazyCall(LossWrapper)(
    loss_fn=LazyCall(nn.CrossEntropyLoss)(),
    in_rules={"output": "input", "target": "target"},
    out_name="loss_cls",
)
optimizer = LazyCall(optim.SGD)(
    params=LazyCall(get_params)(model=None),
    lr=0.1,
)
lr_scheduler = LazyCall(optim.lr_scheduler.MultiStepLR)(milestones=[100, 150])


normalize = LazyCall(T.Normalize)(
    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
)

train_aug = LazyCall(T.Compose)(
    transforms=[
        LazyCall(T.RandomCrop)(size=32, padding=4),
        LazyCall(T.RandomHorizontalFlip)(),
        LazyCall(T.ToTensor)(),
        normalize,
    ]
)

test_aug = LazyCall(T.Compose)(
    transforms=[
        LazyCall(T.ToTensor)(),
        normalize,
    ]
)

train_set = LazyCall(CIFAR10)(
    root="data/cifar10", train=True, transform=train_aug, download=True
)

test_set = LazyCall(CIFAR10)(
    root="data/cifar10", train=False, transform=test_aug, download=True
)


train_loader = LazyCall(DataLoader)(
    dataset=train_set,
    batch_size=128,
    num_workers=2,
    drop_last=True,
    shuffle=True,
)

test_loader = LazyCall(DataLoader)(
    dataset=test_set,
    batch_size=128,
    num_workers=2,
    shuffle=False,
    drop_last=False,
)


loss_metric = (
    LazyCall(Average)(
        in_rules={"loss_cls": "val", "batch_size": "n"},
        out_rules={"avg": "loss"},
    ),
)

train_metrics = LazyCall(MetricCollection)(
    metrics=dict(
        loss=loss_metric,
    )
)
test_metrics = LazyCall(MetricCollection)(
    metrics=dict(
        acc=LazyCall(Accuracy)(topk=[1, 5]),
        loss=loss_metric,
    ),
)

batch_processor = LazyCall(BatchProcessor)(names=["image", "target"])

train_runner = LazyCall(EpochBasedRunner)(
    data_source=train_loader,
    metrics=train_metrics,
    batch_processor=batch_processor,
    hooks=[
        LazyCall(hooks.TrainingHook)(),
        LazyCall(hooks.TextLoggerHook)(stage="train", interval=1),
    ],
)

test_runner = LazyCall(EpochBasedRunner)(
    data_source=test_loader,
    metrics=test_metrics,
    batch_processor=batch_processor,
    hooks=[
        LazyCall(hooks.InferenceHook)(),
        LazyCall(hooks.TextLoggerHook)(stage="test", interval=1),
    ],
)
