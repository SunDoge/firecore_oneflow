from .cifar_resnet import cifar_resnet20
from firecore.config import LazyCall
from flowvision import transforms as T
from oneflow import nn, optim
from firecore_oneflow.optim.builder import get_params
from oneflow.utils.data import DataLoader
from flowvision.datasets.cifar import CIFAR10
from firecore_oneflow.runners.epoch_based import EpochBasedRunner
from firecore_oneflow.module.builder import GraphBuilder

model = LazyCall(cifar_resnet20)(
    num_classes=10,
)
criterion = LazyCall(nn.CrossEntropyLoss)()
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
    dataset=train_set, batch_size=128, num_workers=2, drop_last=True, shuffle=True
)

test_loader = LazyCall(DataLoader)(
    dataset=test_set,
    batch_size=128,
    num_workers=2,
    shuffle=False,
    drop_last=False,
)

train_runner = LazyCall(EpochBasedRunner)(hooks=[])

test_runner = LazyCall(EpochBasedRunner)(hooks=[])
