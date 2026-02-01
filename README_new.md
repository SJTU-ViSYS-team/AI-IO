# AI-IO: Aerodynamics-Inspired Real-Time Inertial Odometry for Quadrotors

![](./img/AI-IO.jpg)

> **Published at ICRA 2026**
> 
> This repository contains the official implementation of AI-IO, a learned inertial odometry system for quadrotors that combines aerodynamic insights with deep learning for robust and efficient ego-motion estimation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training & Evaluation](#training--evaluation)
- [Filtering & Localization](#filtering--localization)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

AI-IO is a learned inertial model odometry system designed for real-time quadrotor ego-motion estimation. Our approach leverages:

- **Aerodynamic modeling**: Incorporates aerodynamic principles to better understand quadrotor dynamics
- **Deep learning**: Uses temporal convolutional networks (TCN) to learn complex inertial-to-velocity mappings
- **Extended Kalman Filter (EKF)**: Integrates learned models with classical filtering for robust state estimation
- **Real-time performance**: Optimized for onboard execution on resource-constrained platforms

### Key Contributions

1. **Aerodynamics-inspired architecture** that captures quadrotor-specific dynamics
2. **Learned neural network model** for accurate velocity estimation from IMU measurements
3. **Integrated EKF framework** combining learning with probabilistic filtering
4. **Comprehensive evaluation** on multiple datasets with comparison to existing methods

## Features

- ✨ Real-time inertial odometry estimation
- 🧠 Learning-based velocity prediction with uncertainty quantification
- 🔄 Extended Kalman Filter integration for state estimation
- 📊 Support for multiple datasets (AI-IO dataset, DIDO dataset)
- 🎯 Pre-trained weights available for immediate evaluation
- 📈 TensorBoard logging for training visualization
- 🔍 Configurable architecture and hyperparameters

## Installation

### Prerequisites

- Python 3.7+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Csfalpha/AI-IO.git
cd AI-IO
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires:
- **PyTorch** (>=2.0): Deep learning framework
- **NumPy, SciPy**: Numerical computing and scientific algorithms
- **h5py**: Data handling for datasets
- **pyhocon**: Configuration file parsing
- **TensorBoard**: Training visualization
- **tqdm, progressbar2**: Progress tracking

See [requirements.txt](requirements.txt) for complete version specifications.

## Quick Start

### 1. Data Preparation

#### Option A: Use Pre-processed AI-IO Dataset

Download the official AI-IO dataset:
```bash
# Download and extract
wget https://github.com/Csfalpha/AI-IO/releases/download/v1.0/AI-IO_dataset.tar.gz
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
    --window_time=1 \
    --batch_size=128 \
    --lr=3e-4
```

**Common training arguments:**
- `--data_config`: Path to dataset configuration file
- `--out_dir`: Output directory for checkpoints and logs
- `--dataset`: Dataset name (our2, DIDO, etc.)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--window_time`: Time window for sequence sampling (seconds)
- `--continue_from`: Resume training from checkpoint

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=results/logs
```

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

**Note:** Pre-trained weights are available [here](https://github.com/Csfalpha/AI-IO/releases/download/dataset-v1.0/checkpoint_open.pt)

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
- `--update_freq`: EKF update frequency
- `--sigma_na`, `--sigma_ng`: Process noise parameters for acceleration/gyroscope
- `--meascov_scale`: Measurement covariance scaling factor

#### 3c. Visualize Filter Results

Generate plots and performance metrics:

```bash
python src/filter/python/plot_filter_output.py \
    --data_config=config/our2.conf \
    --result_dir=results \
    --dataset=our2
```

## Dataset

### AI-IO Dataset

**Features:**
- Collected specifically for quadrotor inertial odometry research
- Multiple flight sequences with diverse motions
- Ground truth trajectories from motion capture system
- IMU data at 100 Hz, camera timestamps for synchronization

**Download:** [AI-IO Dataset v1.0](https://github.com/Csfalpha/AI-IO/releases/download/v1.0/AI-IO_dataset.tar.gz)

**Statistics:**
- Number of sequences: [Add number]
- Total duration: [Add duration]
- Camera models: [Add camera info]
- IMU specifications: [Add IMU info]

### DIDO Dataset

Alternative benchmark dataset available at [zhangkunyi/DIDO](https://github.com/zhangkunyi/DIDO)

### Configuration

Dataset configurations are specified in `config/our2.conf` using HOCON format. Key parameters include:
- Data paths and splits
- IMU frequency and calibration
- Preprocessing parameters
- Train/test sequence lists

## Training & Evaluation

### Architecture

The learned model uses a **Temporal Convolutional Network (TCN)** architecture:

```
Input (batch, 16, window_size)
  ↓
[TCN Blocks × N] - Dilated causal convolutions
  ↓
[Residual Connections] - Skip connections
  ↓
Output (batch, 3) - Velocity prediction
  ↓
Covariance Output (batch, 3) - Uncertainty estimation
```

**Key architecture features:**
- Weight normalization for stable training
- Residual connections for improved gradient flow
- Configurable activation functions (ReLU, GELU)
- Learnable covariance prediction for uncertainty quantification

### Loss Functions

The training objective combines:
- **Regression loss**: MSE on velocity predictions
- **Covariance regularization**: Uncertainty calibration
- **Optional**: Auxiliary losses for auxiliary outputs

Configure loss functions in training parameters.

### Supported Models

- **model_tcn**: Temporal Convolutional Network (default)
- Extensible framework for alternative architectures

### Metrics

Evaluation metrics include:
- **Velocity error**: RMS error, median error, 95-percentile error
- **Trajectory error**: Absolute trajectory error (ATE)
- **Computational cost**: Inference time per sample
- **Uncertainty calibration**: Coverage analysis for predicted covariances

## Filtering & Localization

### Extended Kalman Filter (EKF)

The system uses an EKF for sensor fusion:

**State vector:**
```
[position (3), velocity (3), rotation (4), gyro_bias (3), accel_bias (3)]
```

**Propagation:** IMU integration with gyroscope bias estimation
**Update:** Learned velocity model with predicted covariance

**Key parameters:**
- `sigma_na`: Accelerometer noise
- `sigma_ng`: Gyroscope noise  
- `sigma_nba`: Accelerometer bias random walk
- `sigma_nbg`: Gyroscope bias random walk
- `meascov_scale`: Learned measurement uncertainty scaling

### Filter Tuning

Adjust filter parameters via command-line arguments or configuration files:

```bash
python src/main_filter.py \
    ... \
    --sigma_na=0.008 \
    --sigma_ng=0.004 \
    --meascov_scale=1.0 \
    --mahalanobis_factor=2.0
```

## Project Structure

```
AI-IO/
├── config/
│   └── our2.conf              # Dataset configuration
├── src/
│   ├── main_learning.py       # Training/testing entry point
│   ├── main_filter.py         # Filtering/localization entry point
│   ├── learning/
│   │   ├── train_model_net.py # Training loop
│   │   ├── test_model_net.py  # Testing pipeline
│   │   ├── network/
│   │   │   ├── model_tcn.py   # TCN architecture
│   │   │   ├── model_factory.py
│   │   │   ├── losses.py      # Loss functions
│   │   │   └── covariance_parametrization.py
│   │   ├── data_management/
│   │   │   ├── datasets.py    # Dataset classes
│   │   │   └── prepare_datasets/
│   │   └── utils/             # Utilities (logging, plotting, etc.)
│   └── filter/
│       ├── python/
│       │   ├── src/
│       │   │   ├── filter_manager.py
│       │   │   ├── filter_runner.py     # EKF implementation
│       │   │   ├── scekf.py            # Square-root EKF
│       │   │   ├── meas_source_network.py
│       │   │   └── utils/
│       │   └── plot_filter_output.py
├── img/
│   └── AI-IO.jpg              # Project banner
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # License information
```

## Configuration Files

### Data Configuration (`config/our2.conf`)

Example HOCON configuration:
```hocon
data {
  mode = "train"
  imu_freq = 100
  sampling_freq = 100
  
  data_list = [
    {
      data_root = "/path/to/data"
      data_drive = ["seq1", "seq2", "seq3"]
    }
  ]
}

network {
  type = "tcn"
  layers = 5
  kernel_size = 5
  dropout = 0.2
}

training {
  batch_size = 128
  learning_rate = 3e-4
  epochs = 100
}
```

## Citation

If you use AI-IO in your research, please cite our ICRA 2026 paper:

```bibtex
@inproceedings{aiio2026,
  title={AI-IO: Aerodynamics-Inspired Real-Time Inertial Odometry for Quadrotors},
  author={[Author Names]},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  pages={[Page Range]}
}
```

## Performance

### Benchmark Results

| Dataset | Velocity RMSE (m/s) | ATE (m) | Runtime (ms) |
|---------|-------------------|--------|------------|
| AI-IO   | 0.XXX             | 0.XXX  | X.X        |
| DIDO    | 0.XXX             | 0.XXX  | X.X        |

*Update these values with your actual results*

## Troubleshooting

### Common Issues

**Q: CUDA out of memory**
- A: Reduce batch size with `--batch_size=64` or use CPU with `--device=cpu`

**Q: Poor filtering results**
- A: Adjust EKF parameters (`--sigma_na`, `--meascov_scale`) or check IMU calibration

**Q: Dataset loading errors**
- A: Verify dataset path in config file and ensure correct dataset format

## Contributing

We welcome contributions! Please feel free to:
- Report bugs via GitHub issues
- Submit pull requests for improvements
- Suggest new features or datasets

## Acknowledgments

This project builds upon excellent prior work:

- [Learned Inertial Model Odometry (IMO)](https://github.com/uzh-rpg/learned_inertial_model_odometry) - Core methodology reference
- [TLIO](https://github.com/CathIAS/TLIO) - Filter implementation inspiration
- [DIDO Dataset](https://github.com/zhangkunyi/DIDO) - Benchmark dataset

For detailed attributions, refer to individual source files for copyright and license information.

## License

This project is released under [LICENSE]. See the LICENSE file for details.

For attributions and licenses of included open-source components, please refer to the respective repositories.

## Contact

For questions, please open a GitHub issue or contact the authors.

---

**Keywords:** Inertial Odometry, Quadrotor, Deep Learning, Extended Kalman Filter, Ego-Motion Estimation, Aerodynamics
