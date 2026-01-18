# Contrastive Trajectory Encoders for Zero-Shot Adaptation in CMDPs

A research framework for learning context-aware RL policies through self-supervised contrastive learning on trajectory dynamics.

## Overview

This project implements a two-phase training pipeline:
- **Phase 1 (The "Eye")**: Learn trajectory embeddings via contrastive learning (InfoNCE)
- **Phase 2 (The "Brain")**: Train context-conditional policies using the learned embeddings

## Project Structure

```
rl_encoder/
├── src/
│   ├── data/              # Data collection and trajectory generation
│   ├── models/            # Encoder and policy architectures
│   ├── training/          # Training loops for both phases
│   ├── evaluation/        # Metrics and visualization
│   └── utils/             # Helper functions
├── configs/               # Experiment configurations
├── experiments/           # Saved models, logs, embeddings
├── scripts/               # Entry points for training/evaluation
└── requirements.txt       # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Phase 1: Train Trajectory Encoder

```bash
python scripts/train_encoder.py --config configs/encoder_config.yaml
```

### Phase 2: Train Context-Conditional Policy

```bash
python scripts/train_policy.py --config configs/policy_config.yaml --encoder-path experiments/encoder_best.pt
```

### Evaluate Zero-Shot Performance

```bash
python scripts/evaluate.py --policy-path experiments/policy_best.pt --encoder-path experiments/encoder_best.pt
```

## Baselines

- **Standard PPO**: Context-blind policy trained across all environments
- **Oracle PPO**: Policy with ground-truth context labels (upper bound)

## Key Features

- Support for CARL environments (Ant, Pendulum, CartPole)
- Flexible encoder architectures (LSTM, Transformer)
- InfoNCE contrastive loss with hard negative mining
- t-SNE visualization of learned embeddings
- Integration with Stable-Baselines3
- Comprehensive evaluation metrics

## Research Context

This implementation is based on the principle that trajectories from the same environment (same physics parameters) should produce similar embeddings, while trajectories from different environments should be distinguishable. This enables zero-shot adaptation to novel contexts without explicit labels so the anchor and postive are closer together and negative is further apart

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{contrastive_trajectory_encoders,
  title={Contrastive Trajectory Encoders for Zero-Shot Adaptation in CMDPs},
  author={Your Name},
  year={2026}
}
```
