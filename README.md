# AI-IO: An Aerodynamics-Inspired Real-Time Inertial Odometry for Quadrotors

[![Watch the video](https://img.youtube.com/vi/lKRBhg7UFSg/maxresdefault.jpg)](https://youtu.be/lKRBhg7UFSg)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Installation

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/SJTU-ViSYS-team/AI-IO.git
cd AI-IO
```

2. Create a conda environment (recommended):
```bash
conda create -n AI-IO python=3.10
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

See [requirements.txt](requirements.txt) for complete dependencies and their version specifications.

## Quick Start

### 1. Data Preparation

#### Option A: Use Pre-processed AI-IO Dataset

Download the official AI-IO dataset:
```bash
# Download and extract
wget https://github.com/SJTU-ViSYS-team/AI-IO/releases/download/v1.0/AI-IO_dataset.tar.gz
tar -xzf AI-IO_dataset.tar.gz
```

#### Option B: Use DIDO Dataset

Alternatively, use the DIDO dataset from [here](https://github.com/zhangkunyi/DIDO).

### 2. Training a New Model

Train the network with default configurations:

```bash
python src/main_learning.py \
    --data_config=config/our2.conf \
    --out_dir=results \
    --dataset=our2 \
    --mode=train \
    --imu_freq=100 \
    --sampling_freq=100 \
    --window_time=1
```

**Common training arguments:**
- `--data_config`: Path to dataset configuration file
- `--out_dir`: Output directory for checkpoints and logs
- `--dataset`: Dataset name (our2, DIDO, etc.)
- `--window_time`: Time window for sequence sampling (seconds)
- `--continue_from`: Resume training from checkpoint


### 3. Evaluation

#### 3a. Test the Learned Model

Evaluate a trained model on test data:

```bash
python src/main_learning.py \
    --data_config=config/our2.conf \
    --out_dir=results \
    --dataset=our2 \
    --mode=test \
    --imu_freq=100 \
    --sampling_freq=100 \
    --window_time=1 \
    --model_fn=checkpoint_yours.pt \
    --show_plots
```

**Note:** Pre-trained weights are available [here](https://github.com/SJTU-ViSYS-team/AI-IO/releases/download/dataset-v1.0/checkpoint_open.pt)

#### 3b. Run Extended Kalman Filter

Estimate full state trajectories using the learned model with EKF:

```bash
python src/main_filter.py \
    --data_config=config/our2.conf \
    --out_dir=results \
    --dataset=our2 \
    --checkpoint_fn=checkpoint_yours.pt \
    --model_param_fn=model_net_parameters.json
```

**EKF Configuration Options:**
- `--checkpoint_fn`: Path to trained network weights
- `--model_param_fn`: Model architecture parameters (JSON format)

#### 3c. Visualize Filter Results

Generate plots and performance metrics:

```bash
python src/filter/python/plot_filter_output.py \
    --data_config=config/our2.conf \
    --result_dir=results \
    --dataset=our2
```

## Citation

If you use AI-IO in your research, please cite our paper:

```bibtex

```

## Acknowledgments

This project builds upon excellent prior work:

- [Learned Inertial Model Odometry (IMO)](https://github.com/uzh-rpg/learned_inertial_model_odometry)
- [TLIO](https://github.com/CathIAS/TLIO)
