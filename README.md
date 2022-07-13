<p align="center">
    <br>
    <img src="./figures/logo.png" height="100" width="100"/>
    <br>
</p>

<h1 align="center">Missel</h1>
<h3 align="center">A collection of semi-supervised learning implemented by JAX/Flax</h3>

## Overview

Missel provides popular semi-supervised learning (SemiSL) methods implemented by JAX/Flax. You can easily try and customize them.

### Methods
- PiModel (now in progress)
- PseudoLabel
- MeanTeacher
- VAT (now in progress)
- MixMatch
- UDA (now in progress)

### Datasets
- cifar10
- cifar100
- STL10

### Results
Currently, SemiSL methods are only tested by cifar10.

| cifar10 | 40 labels | 250 labels | 4000 labels |
| ---- | ---- | ---- | ---- |
| PiModel | N/A | N/A | N/A |
| PseudoLabel | N/A | N/A | 84.8 |
| MeanTeacher | N/A | N/A | 91.7 |
| VAT | N/A | N/A | N/A |
| MixMatch | N/A | 84.5 | N/A |

## Usage

### Installation

Missel is tested on the container.
You can build the container using `apptainer/env.def`.

```sh
git clone git@github.com:h-terao/Missel.git && cd Missel
sudo singularity build env.sif apptainer/env.def
apptainer shell --nv env.sif
```

### Training

```bash
python train.py learner=<Method> dataset=<Dataset> dataset.num_labels=<Number of labeled data> <Other configuration>
```

<br>

<footer>
Logo is generated by <a target="_blank" href="https://huggingface.co/spaces/dalle-mini/dalle-mini">DALL-E mini</a>
</footer>