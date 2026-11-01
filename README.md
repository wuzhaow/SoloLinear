# SoloLinear

This repository contains the official implementation of the paper:

**[Less is more: Challenging the necessity of full endogenous history context in time series forecasting](https://www.sciencedirect.com/science/article/pii/S0950705126016060)**

**Journal:** Knowledge-Based Systems (2026)

**Paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126016060)

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


