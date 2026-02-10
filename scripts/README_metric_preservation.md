# Metric Preservation Analysis

This script analyzes whether the trained contrastive encoder preserves the metric structure of the context parameter space.

## What is Metric Preservation?

In the context of contrastive trajectory encoders, **metric preservation** means that:

1. Contexts with similar parameters (e.g., gravity values) produce similar embeddings
2. The distance between embeddings is proportional to the distance between parameters
3. The learned embedding space maintains the neighborhood structure of the parameter space

This is crucial for zero-shot adaptation because it ensures that the encoder can generalize smoothly across the parameter space.

## Usage

### Basic Usage

```bash
conda activate rl_encoder
python scripts/analyze_metric_preservation.py \
    --encoder-path experiments/encoder/best_encoder.pt \
    --data-path experiments/data/pendulum_train_segments.pkl
```

### Full Options

```bash
python scripts/analyze_metric_preservation.py \
    --encoder-path experiments/encoder/best_encoder.pt \
    --data-path experiments/data/pendulum_train_segments.pkl \
    --results-dir results \
    --param-key g \
    --param-name Gravity \
    --device auto
```

### Arguments

- `--encoder-path` (required): Path to trained encoder checkpoint (.pt file)
- `--data-path` (required): Path to trajectory data (.pkl file with segments)
- `--results-dir`: Directory to save results (default: `results/`)
- `--param-key`: Context parameter key to analyze (default: `g` for gravity)
- `--param-name`: Human-readable parameter name for plots (default: `Gravity`)
- `--device`: Device to use: `cuda`, `cpu`, or `auto` (default: `auto`)

## Output Files

The script generates two files in the results directory:

### 1. `metric_preservation_correlation.png`
A scatter plot showing:
- X-axis: Parameter distance (e.g., Δgravity)
- Y-axis: Embedding distance (L2 norm)
- Red line: Linear regression fit
- Text box: Correlation statistics

### 2. `metric_preservation_stats.txt`
A text file containing:
- Pearson correlation coefficient and p-value
- Spearman correlation coefficient and p-value
- R² score from linear regression
- Linear regression slope and intercept
- Interpretation of results

## Interpreting Results

### Correlation Coefficients

- **Pearson r > 0.8**: Excellent metric preservation (strong linear relationship)
- **Pearson r = 0.6-0.8**: Good metric preservation (moderate linear relationship)
- **Pearson r < 0.6**: Weak metric preservation (limited relationship)

### What Good Results Mean

High correlation (>0.8) indicates:
- ✓ The encoder successfully learned to map similar contexts to similar embeddings
- ✓ The embedding space preserves the structure of the parameter space
- ✓ Zero-shot adaptation should work well across the parameter range
- ✓ Interpolation between contexts is meaningful

### What Poor Results Mean

Low correlation (<0.6) suggests:
- ✗ The encoder may not have learned meaningful context representations
- ✗ Similar contexts might produce very different embeddings
- ✗ Zero-shot adaptation performance may be unpredictable
- ✗ May need to adjust training hyperparameters or loss function

## Example Workflow

### 1. Train an encoder (Phase 1)

```bash
# Collect data
python scripts/collect_data.py \
    --env pendulum \
    --split train \
    --output experiments/data/pendulum_train_segments.pkl

# Train encoder
python scripts/train_encoder.py \
    --config configs/encoder_config.yaml \
    --data-path experiments/data/pendulum_train_segments.pkl
```

### 2. Analyze metric preservation

```bash
python scripts/analyze_metric_preservation.py \
    --encoder-path experiments/encoder/best_encoder.pt \
    --data-path experiments/data/pendulum_train_segments.pkl
```

### 3. Review results

Check the generated files in `results/`:
- View the scatter plot to visualize the relationship
- Read the statistics file for numerical metrics
- Use the interpretation to decide if the encoder is ready for Phase 2

### 4. If results are poor

Consider:
- Training for more epochs
- Collecting more trajectory data
- Adjusting the contrastive loss temperature
- Trying a different encoder architecture
- Increasing the latent dimension

## For Different Environments

### CartPole (multiple parameters)

```bash
python scripts/analyze_metric_preservation.py \
    --encoder-path experiments/encoder/cartpole_encoder.pt \
    --data-path experiments/data/cartpole_train_segments.pkl \
    --param-key length \
    --param-name "Pole Length"
```

### Ant (MuJoCo environment)

```bash
python scripts/analyze_metric_preservation.py \
    --encoder-path experiments/encoder/ant_encoder.pt \
    --data-path experiments/data/ant_train_segments.pkl \
    --param-key mass \
    --param-name "Body Mass"
```

## Technical Details

### How it Works

1. **Load encoder and data**: Loads the trained encoder checkpoint and trajectory segments
2. **Compute mean embeddings**: For each context, computes the mean embedding across all trajectories
3. **Pairwise distances**: Computes all pairwise distances in both parameter space and embedding space
4. **Correlation analysis**: Calculates Pearson and Spearman correlations
5. **Linear regression**: Fits a linear model and computes R² score
6. **Visualization**: Creates scatter plot with regression line

### Distance Metrics

- **Parameter distance**: Absolute difference |param_i - param_j|
- **Embedding distance**: L2 norm ||emb_i - emb_j||₂

### Statistical Tests

- **Pearson correlation**: Measures linear relationship strength
- **Spearman correlation**: Measures monotonic relationship (rank-based)
- **R² score**: Proportion of variance explained by linear model
- **p-values**: Statistical significance of correlations (p < 0.05 is significant)

## Troubleshooting

### Error: "No module named 'src'"

Make sure you're running from the project root directory:
```bash
cd /path/to/contrastive-trajectory-encoders-
python scripts/analyze_metric_preservation.py ...
```

### Error: "Parameter key 'g' not found"

Check what parameters are available in your data:
```python
import pickle
with open('path/to/segments.pkl', 'rb') as f:
    segments = pickle.load(f)
print(segments[0].context_params)  # Shows available parameters
```

Then use the correct `--param-key`.

### Error: "No such file or directory: experiments/encoder/best_encoder.pt"

Make sure you've trained an encoder first:
```bash
python scripts/train_encoder.py --config configs/encoder_config.yaml
```

## Citation

If you use this analysis in your research, please cite:

```bibtex
@misc{contrastive_trajectory_encoders,
  title={Contrastive Trajectory Encoders for Zero-Shot Adaptation in CMDPs},
  author={Your Name},
  year={2026}
}
```
