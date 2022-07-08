from __future__ import annotations
from . import dataset


class CIFAR10(dataset.ArrayDataset):
    data_name: str = "cifar10"
    num_classes: int = 10
    image_size: int = 32
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]


class CIFAR100(dataset.ArrayDataset):
    data_name: str = "cifar100"
    num_classes: int = 100
    image_size: int = 32
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]


class STL10(dataset.ArrayDataset):
    data_name: str = "stl10"
    num_classes: int = 10
    image_size: int = 96
    ulb_split: str | None = "unlabelled"
    mean = [x / 255 for x in [112.4, 109.1, 98.6]]
    std = [x / 255 for x in [68.4, 66.6, 68.5]]
