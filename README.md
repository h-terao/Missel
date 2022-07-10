<div align="center">

# Missel

</div>

A collection of semi-supervised learning algorithms implemented by JAX/Flax.
"Missel" is an anagram of SemiSL (SEMI-Supervised Learning).

# Introduction

Currecntly, missel supports the following algorithms:

- Virtual Adversarial Training

Datasets: cifar10, cifar100, ...

## Results

# Getting Started

## Requirements
- Python >= 3.7

## Examples

Training VAT with default parameters.
```bash
python train.py learner=vat
```

See help to check other parameters:
```bash
python train.py --help
```
