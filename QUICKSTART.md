# Quick Start Guide

This guide will walk you through the complete pipeline for training contrastive trajectory encoders and context-conditional policies.

## Installation

```bash
# Clone or navigate to the repository
cd rl_encoder

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Collect Trajectory Data

First, collect trajectory segments from CARL environments:

```bash
# Collect training data
python scripts/collect_data.py \
    --config configs/encoder_config.yaml \
    --split train

# Collect test data
python scripts/collect_data.py \
    --config configs/encoder_config.yaml \
    --split test
```

This will create pickle files in `experiments/data/` containing trajectory segments.

**Expected output:**
- `pendulum_train_segments.pkl`: ~2000 segments from 20 contexts
- `pendulum_test_segments.pkl`: ~500 segments from 5 contexts

## Step 2: Train Trajectory Encoder (Phase 1)

Train the encoder using contrastive learning:

```bash
python scripts/train_encoder.py \
    --config configs/encoder_config.yaml \
    --data-path experiments/data/pendulum_train_segments.pkl
```

**What happens:**
- The encoder learns to map trajectories to latent embeddings
- Trajectories from the same context cluster together
- Training takes ~30 minutes on GPU for 100 epochs

**Outputs:**
- Best model: `experiments/checkpoints/encoder/best_encoder.pt`
- Logs: `experiments/logs/encoder/`
- TensorBoard logs for monitoring

**Monitor training:**
```bash
tensorboard --logdir experiments/logs/encoder
```

## Step 3: Visualize Embeddings

Verify that the encoder learned meaningful representations:

```bash
python scripts/visualize_embeddings.py \
    --encoder-path experiments/checkpoints/encoder/best_encoder.pt \
    --data-path experiments/data/pendulum_train_segments.pkl \
    --method tsne
```

**Expected results:**
- t-SNE plot showing clear clusters for different contexts
- High intra-context similarity (>0.8)
- Low inter-context similarity (<0.3)
- High separation metric (>0.5)

## Step 4: Train Context-Conditional Policy (Phase 2)

Train a PPO policy that uses the learned context embeddings:

```bash
python scripts/train_policy.py \
    --config configs/policy_config.yaml \
    --encoder-path experiments/checkpoints/encoder/best_encoder.pt
```

**What happens:**
- The encoder is frozen and used to provide context embeddings
- PPO learns a policy conditioned on [observation, context]
- Training takes ~2-3 hours for 1M timesteps

**Outputs:**
- Best model: `experiments/checkpoints/policy/best_model.zip`
- Logs: `experiments/logs/policy/`

## Step 5: Evaluate Zero-Shot Performance

Test the policy on unseen contexts:

```bash
python scripts/evaluate.py \
    --policy-path experiments/checkpoints/policy/best_model.zip \
    --encoder-path experiments/checkpoints/encoder/best_encoder.pt \
    --env-name pendulum \
    --n-episodes 20
```

**Expected results:**
- The policy should adapt to new contexts without additional training
- Performance should be close to oracle (policy with ground-truth labels)

## Configuration Options

### Encoder Configuration ([encoder_config.yaml](configs/encoder_config.yaml))

**Key parameters:**
- `encoder.architecture`: Choose "lstm" or "transformer"
- `encoder.latent_dim`: Size of context embedding (default: 64)
- `data.segment_length`: Trajectory length for encoding (default: 32)
- `loss.temperature`: Temperature for InfoNCE loss (default: 0.07)

### Policy Configuration ([policy_config.yaml](configs/policy_config.yaml))

**Key parameters:**
- `policy.algorithm`: Choose "ppo" or "sac"
- `encoder.buffer_length`: How many steps to buffer before inferring context
- `training.total_timesteps`: Total training steps (default: 1M)

## Experiment Tracking

All experiments are logged to:
- TensorBoard: `experiments/logs/`
- Checkpoints: `experiments/checkpoints/`
- Results: `experiments/results/`

## Tips for Best Results

1. **Data Collection:**
   - Use at least 50-100 segments per context
   - Longer segments (32-64 steps) work better
   - Random policy is sufficient for data collection

2. **Encoder Training:**
   - Train for at least 50 epochs
   - Monitor the separation metric (should increase)
   - Try both LSTM and Transformer architectures

3. **Policy Training:**
   - Freeze the encoder (recommended)
   - Use a buffer length that matches training segment length
   - Start with PPO (more stable than SAC)

4. **Debugging:**
   - If embeddings don't cluster, increase training epochs
   - If policy doesn't adapt, check buffer length and encoder quality
   - Visualize embeddings frequently to verify learning

## Common Issues

**Issue: CUDA out of memory**
- Reduce batch size in config
- Use smaller encoder (fewer layers/hidden dim)
- Switch to CPU if necessary

**Issue: Poor embedding quality**
- Collect more data per context
- Increase number of training epochs
- Try different temperature values (0.05-0.1)

**Issue: Policy doesn't adapt**
- Verify encoder produces distinct embeddings for different contexts
- Increase buffer length (needs more observations)
- Check that encoder is properly frozen during policy training

## Next Steps

1. Try different environments (CartPole, Ant)
2. Experiment with harder contexts (larger parameter ranges)
3. Compare with baselines (standard PPO, oracle PPO)
4. Implement curriculum learning for policy training
5. Try ensemble encoders for robustness

## Directory Structure After Training

```
rl_encoder/
├── experiments/
│   ├── data/
│   │   ├── pendulum_train_segments.pkl
│   │   └── pendulum_test_segments.pkl
│   ├── checkpoints/
│   │   ├── encoder/
│   │   │   ├── best_encoder.pt
│   │   │   └── config.yaml
│   │   └── policy/
│   │       ├── best_model.zip
│   │       └── config.yaml
│   ├── logs/
│   │   ├── encoder/
│   │   └── policy/
│   ├── visualizations/
│   │   ├── embeddings_tsne_by_context.png
│   │   └── context_separation.png
│   └── results/
│       └── zero_shot_results.json
```

## Citation

If you use this code for research, please cite:

```bibtex
@misc{contrastive_trajectory_encoders,
  title={Contrastive Trajectory Encoders for Zero-Shot Adaptation in CMDPs},
  author={Your Name},
  year={2026}
}
```
