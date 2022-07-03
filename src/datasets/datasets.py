from __future__ import annotations
from . import dataset


class CIFAR10(dataset.Dataset):
    name: str = "cifar10"

    lb_path: str = "hub://activeloop/cifar10-train"
    test_path: str = "hub://activeloop/cifar10-test"
    data_meta = {
        "image_size": 32,
        "mean": [x / 255 for x in [125.3, 123.0, 113.9]],
        "std": [x / 255 for x in [63.0, 62.1, 66.7]],
    }


class CIFAR100(dataset.Dataset):
    name: str = "cifar100"

    lb_path: str = "hub://activeloop/cifar100-train"
    test_path: str = "hub://activeloop/cifar100-test"
    data_meta = {
        "image_size": 32,
        "mean": [x / 255 for x in [129.3, 124.1, 112.4]],
        "std": [x / 255 for x in [68.2, 65.4, 70.4]],
    }


class STL10(dataset.Dataset):
    name: str = "stl10"

    lb_path: str = "hub://activeloop/stl10-train"
    ulb_path: str = "hub://activeloop/stl10-unlabeled"
    test_path: str = "hub://activeloop/stl10-test"
    data_meta = {
        "image_size": 96,
        "mean": [x / 255 for x in [112.4, 109.1, 98.6]],
        "std": [x / 255 for x in [68.4, 66.6, 68.5]],
    }
