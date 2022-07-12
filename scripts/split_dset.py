"""split dataset into labeled and unlabeled."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", None, "Dataset name.", required=True)
flags.DEFINE_string("train_path", None, "Path to train data.", required=True)
flags.DEFINE_string("test_path", None, "Path to test data.", required=True)
flags.DEFINE_string("out_dir", "data/", "Output directory.")
flags.DEFINE_multi_integer("seeds", [0], "Seed values to split.")


registered_datasets = {
    "cifar10": (
        "hub://activeloop/cifar10-train",
        "hub://activeloop/cifar10-test",
    ),
    "cifar100": (
        "hub://activeloop/cifar100-train",
        "hub://activeloop/cifar100-test",
    ),
}


def split_into_xy(data):
    pass


def main(argv):
    del argv

    out_dir_path = Path(FLAGS.out_dir, FLAGS.dataset)
