# SoloLinear

This repository is the official implementation for the paper: **SoloLinear: Time Series Forecasting with One Single Endogenous Point**

## Installation

### Environment Setup

Create and activate a conda environment:

```bash
conda create -n SoloLinear python=3.9
conda activate SoloLinear
```

### Install Dependencies

Install required packages using pip:

```bash
pip install -r requirements.txt
```

Alternatively, for users in China, you can use Tsinghua mirror for faster installation:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Quick Start

### Running Training Scripts

The `scripts/` directory contains training scripts for all datasets used in the paper, making it easy to reproduce results across different benchmarks.

To reproduce the training results on ETTh1 dataset:

```bash
bash ./scripts/ETTh1/SoloLinear.sh
```

For other datasets, navigate to the corresponding directory under `scripts/` and run the respective training script:

```bash
# Example for other datasets
bash ./scripts/[DATASET_NAME]/SoloLinear.sh
```

Each script will run the training process on the specified dataset and reproduce the results presented in the paper.


