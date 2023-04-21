from .cifar_resnet import cifar_resnet20
from firecore.config import LazyCall
from flowvision import transforms as T
from oneflow import nn, optim
from firecore_oneflow.optim.builder import get_params


model = LazyCall(cifar_resnet20)(
    num_classes=10,
)
criterion = LazyCall(nn.CrossEntropyLoss)()
optimizer = LazyCall(optim.SGD)(
    params=LazyCall(get_params)(),
    lr=0.1,
)
lr_scheduler = LazyCall(optim.lr_scheduler.MultiStepLR)(
    milestones=[100, 150]
)

normalize = LazyCall(T.Normalize)(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010)
)

train_aug = LazyCall(T.Compose)(
    transforms=[
        LazyCall(T.RandomCrop)(size=32, padding=4),
        LazyCall(T.RandomHorizontalFlip)(),
        LazyCall(T.ToTensor)(),
        normalize
    ]
)

test_aug = LazyCall(T.Compose)(
    transforms=[
        LazyCall(T.ToTensor)(),
        normalize,
    ]
)

train_runner = LazyCall
