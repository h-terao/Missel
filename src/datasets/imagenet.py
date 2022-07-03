from __future__ import annotations
import tensorflow as tf

from . import dataset


def imagenet_preprocess(x):
    x = tf.image.resize([x], size=[224, 224])[0]
    x = tf.repeat(x, 3 // x.shape[-1], axis=-1)
    return x


class ImageNet(dataset.Dataset):
    name: str = "imagenet"

    lb_path: str = "hub://activeloop/imagenet-train"
    test_path: str = "hub://activeloop/imagenet-val"
    data_meta = {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    preprocess = imagenet_preprocess
