<p align="center">
    <br>
    <img src="./figures/logo.png" height="128" width="128"/>
    <br>
</p>

<h1 align="center">Missel</h1>
<h3 align="center">A semi-supervised learning collection built on JAX/Flax</h3>

## Overview

Missel provides popular semi-supervised learning (SemiSL) methods implemented by JAX/Flax.
You can easily try and customize them.

### Methods
- Supervised
- PiModel
- PseudoLabel
- MeanTeacher
- VAT
- MixMatch
- UDA
- FixMatch

### Datasets
- cifar10
- cifar100
- STL10

### Results
Currently, SemiSL methods are only tested on cifar10.

| cifar10 | 40 labels | 250 labels | 4000 labels |
| ---- | ---- | ---- | ---- |
| Supervised | N/A | N/A | N/A |
| PiModel | N/A | N/A | 87.7 |
| PseudoLabel | N/A | N/A | 84.8 |
| MeanTeacher | N/A | N/A | 91.7 |
| VAT | N/A | N/A | 89.4 |
| MixMatch | N/A | 84.5 | N/A |
| UDA | 84.9 | N/A | N/A |
| FixMatch | 93.1 | 95.3 | N/A |

## Getting started
### Setup

Missel recommends to use [Apptainer](https://apptainer.org/docs/admin/main/installation.html#installation-on-linux) (Singularity) to setup the environment.
After install Apptainer, you can easily build the container for Missel.
```bash
sudo apptainer build env.sif apptainer/env.def
apptainer shell --nv env.sif
```

To install dependencies locally, follow the steps written in [apptainer/env.def](./apptainer/env.def).


### Running

All SemiSL methods have the common interface like:
```bash
python train.py experiment=<learner name> <other configs>
```

For example, the below command trains ResNet50 by VAT on 4000 labels of cifar10.
```bash
python train.py experiment=vat model=resnet50 dataset=cifar10 dataset.num_labels=4000
```

You can check other arguments via:
```bash
python train.py --help
```

### Tab completion

This project supports the tab completion powered by [Hydra](https://hydra.cc/)). <br>
To enable the tab completion, type the following command:
```bash
eval "$(python train.py -sc install=SHELL_NAME)"
```
where SHELL_NAME should be replaced with "bash", "zsh" or "fish".


## Customization

To add your own SemiSL methods or datasets, follow the following steps.

### Add new SemiSL methods
To add your own SemiSL method into Missel, follow the below steps:
1. Write learner class in src/learners.
2. Write config file in config/learner.
3. (Optional) Write experiment config file in config/experiment.

### Add new datasets
To add your own datasets into Missel, follow the below steps:
1. Write dataset class in src/datasets.
2. Write config file in config/dataset.

## TODO
- Rewrite experiments like `mixmatch_cifar10_4000.yaml`.

## Acknowledgement
This project referenced [TorchSSL](https://github.com/TorchSSL/TorchSSL) and [official fixmatch implementation](https://github.com/google-research/fixmatch).<br>
The Missel's logo is generated by OpenAI DALL-E.