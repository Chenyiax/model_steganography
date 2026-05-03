# Steganography With Constructing Neural Networks

This repository contains the source code for the paper: "Steganography With Constructing Neural Networks".

## Overview

This project implements a neural network-based steganography method that embeds secret information directly into the parameters of a host neural network. It utilizes an encoder-decoder framework to conceal data within the layer weights and biases, while maintaining the host model's primary classification or task performance.

## Prerequisites

- Python 3.x
- PyTorch
- Additional dependencies (check requirements in the `utils` scripts or typical data science environment)

## Project Structure

- `model_steganography.py`: Core logic for embedding (`encode`) and extracting (`decode`) secret information.
- `models/`: Contains pre-trained encoder and decoder models (e.g., `encoder128.pth`, `decoder128.pth`).
- `utils/`: Utility functions for data handling, initialization, training, and testing.
- `dataset/`: Directory for datasets (MNIST, FashionMNIST, CIFAR-10).
- `stego_models/`: Storage for models with embedded information.
- `plt/`: Visualization and plotting scripts used for analysis.
- `test/`: Scripts for testing functionalities like fine-tuning and parameter pruning.

## Core Usage

The `ModelSteganography` class in `model_steganography.py` provides the main interface:

```python
from model_steganography import ModelSteganography

# Initialize with desired parameters
stego = ModelSteganography(init_function=your_init_func)

# Embed information
secret_raw, secret_bch = stego.encode(model)

# Extract information
ext_raw, ext_bch = stego.decode(model)
```
## Experiment Data Download

https://pan.quark.cn/s/8226327ae2eb?pwd=DPMT

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{xu2025steganography,
  title={Steganography with constructing neural networks},
  author={Xu, Chenyi and Huang, Lin and Qin, Chuan and Feng, Guorui and Zhang, Xinpeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

## Disclaimer

This software is for academic and research purposes only. The authors take no responsibility for any misuse of this technology.
