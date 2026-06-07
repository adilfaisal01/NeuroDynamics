# NeuroDynamics

Using transformers for chaotic system parameter identification.

## Overview

NeuroDynamics is a deep learning research project that applies **transformer** and **LSTM** neural networks to the problem of **parameter identification in chaotic physical systems**. Given noisy time-series observations of a chaotic system's state, the model infers the underlying physical parameters that generated those dynamics.

The primary testbed is the **double pendulum** -- a classic chaotic system with 4 states (θ₁, ω₁, θ₂, ω₂) and 4 physical parameters (m₁, m₂, l₁, l₂).

**Two-stage training pipeline:**

1. **Stage 1 (Supervised)** -- Train a Transformer (or LSTM) from scratch on noisy trajectory data to predict physical parameters using MSE loss.
2. **Stage 2 (Physics-Informed Fine-Tuning)** -- Fine-tune with a Hamiltonian energy conservation penalty. Since ideal double pendulums conserve total energy, this physics-based regularization significantly improves parameter estimation accuracy.

## Features

- **Chaotic system simulation** -- ODE-based physics engine using `scipy.integrate.solve_ivp` (RK45). Supports double pendulum and Chua circuit systems with configurable noise.
- **Transformer model** -- Configurable Transformer Encoder architecture with positional encoding, adaptive pooling, and MLP regression head.
- **LSTM model** -- Stacked LSTM with final hidden state mapped to parameter predictions.
- **Physics-informed loss** -- Hamiltonian energy conservation penalty (variance of total energy H = T + V) combined with MSE.
- **C++ Hamiltonian verification** -- Fast pybind11 C++ module for energy conservation checking during inference.
- **Docker & RunPod support** -- Containerized with GPU acceleration, automated hyperparameter sweeps.

## Installation

### Local

```bash
# Python dependencies
pip install -r requirements.txt

# PyTorch (install separately per your CUDA version)
pip install torch torchvision torchaudio

# Optional: C++ test suite
cd test_suites && mkdir build && cd build
cmake .. && make
```

### Docker

```bash
docker compose build
```

## Usage

### 1. Generate dataset

```bash
python dataset_generation.py
```

Generates double pendulum trajectory datasets (including noisy observations) as `.parquet` files in `datasets/`.

### 2. Stage 1 -- Train from scratch

```bash
python model_training.py \
  --model_name "model.pth" \
  --model_type "transformer" \
  --dataset_dir "datasets" \
  --output_dir "outputs"
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | `1e-3` | Learning rate |
| `DS_LEN` | `5000` | Downsampled sequence length |
| `NE` | `200` | Number of epochs |
| `BATCH` | `32` | Batch size |
| `ED` | `256` | Embedding dimension |
| `HEAD` | `16` | Attention heads |
| `HD` | `2048` | Hidden/FFN dimension |
| `NAME` | — | Model output filename |
| `TYPE` | `transformer` | Model type (`transformer` or `lstm`) |

### 3. Stage 2 -- Physics-informed fine-tuning

```bash
python physics_tuning.py
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | `3e-5` | Learning rate |
| `LAMBDA` | `0.4` | Hamiltonian loss weight |
| `BATCH` | `64` | Batch size |

Loads a pre-trained model from `outputs/model_dlen5000_big.pth` and fine-tunes it on `datasets/dataset_doublependulumpts_finetune.parquet`. The loss function is:

```
Loss = MSE(params_pred, params_true) + λ * Var(H)
```

where `Var(H)` is the variance of the computed Hamiltonian (normalized by mean magnitude), penalizing energy non-conservation.

### 4. Evaluation

```bash
python test.py
```

Evaluates trained models on Set C data, tests ensemble predictions with blending ratios α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} between the base and fine-tuned models.

## Docker / RunPod

### Hyperparameter sweeps

```bash
# Transformer sweep (36 configurations)
bash transformer_execution.sh

# LSTM sweep (12 configurations)
bash LSTM_execution.sh
```

### Cloud training (RunPod)

```bash
# Stage 1: Sweep over sequence lengths 2500 and 5000 (200 epochs each)
bash runpod_sweep.sh

# Stage 2: Fine-tune (20 epochs, lr=3e-5, λ=0.4, batch=64)
bash runpod_finetune.sh
```

## Project Structure

```
.
├── physics_system.py           # ODE simulation: DoublePendulum, ChuaCircuit
├── dataset_generation.py       # Synthetic dataset generation (parquet)
├── transformer_model.py        # Transformer encoder architecture
├── LSTMmodel.py                # LSTM architecture
├── model_training.py           # Stage 1: supervised training
├── physics_tuning.py           # Stage 2: physics-informed fine-tuning
├── test.py                     # Inference & evaluation
├── requirements.txt
├── DockerFile.dockerfile
├── docker-compose.yaml
├── entrypoint.sh               # Docker entrypoint
├── datasets/                   # Parquet datasets
├── outputs/                    # Trained models, loss plots, error analysis
├── test_suites/                # C++ pybind11 Hamiltonian verification
│   ├── CMakeLists.txt
│   ├── hamiltonian_pybind.cpp
│   └── hamiltonian_tests.cpp
├── transformer_execution.sh    # Docker sweep script (transformer)
├── LSTM_execution.sh           # Docker sweep script (LSTM)
├── runpod_sweep.sh             # RunPod Stage 1 sweep
└── runpod_finetune.sh          # RunPod Stage 2 fine-tune
```

## License

MIT License. Copyright (c) 2025 Adil Faisal.
