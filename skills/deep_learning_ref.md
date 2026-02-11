# Deep Learning & Prediction Models - Skill Reference

> Practical reference for building time series and financial prediction models in Python/PyTorch.
> Ranked by real-world effectiveness, not hype. Updated for 2025+ best practices.

---

## 1. Time Series Architectures (Ranked by Effectiveness)

### 1.1 Temporal Fusion Transformer (TFT)
**State of the art for interpretable multi-horizon forecasting.**
- Combines LSTM encoder with multi-head attention and variable selection networks.
- Handles static covariates, known future inputs, and observed past inputs natively.
- Produces quantile forecasts out of the box.
- **When to use:** You need interpretability AND accuracy; you have mixed feature types.
- **Key hyperparams:** `hidden_size` (64-256), `num_attention_heads` (4-8), `dropout` (0.1-0.3), `num_lstm_layers` (1-2), `quantiles` ([0.1, 0.5, 0.9]).
- **Pitfall:** Slow to train. Overkill for univariate series. Needs decent dataset size (>1000 samples).

```python
# Using pytorch-forecasting (recommended wrapper)
from pytorch_forecasting import TemporalFusionTransformer
model = TemporalFusionTransformer.from_dataset(
    training_dataset,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.2,
    hidden_continuous_size=64,
    learning_rate=1e-3,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
)
```

### 1.2 N-HiTS / N-BEATS
**Neural basis expansion — pure time series, no covariates needed.**
- N-BEATS: Stacks of fully connected blocks with basis expansion (trend + seasonality).
- N-HiTS: Adds hierarchical interpolation, drastically cuts compute for long horizons.
- **When to use:** Univariate or few-variate forecasting; you want a strong baseline fast.
- **Key hyperparams:** `num_stacks` (2-4), `num_blocks` (3-5 per stack), `layer_widths` (256-512), `expansion_coefficient_dim` (5-10), `pooling_sizes` (N-HiTS only: [2,4,8]).
- **Pitfall:** No native covariate support in vanilla N-BEATS. Use N-HiTS for long horizons.

### 1.3 PatchTST
**Patched Time Series Transformer — channel-independent design.**
- Splits each time series into subseries-level patches, then applies vanilla transformer.
- Channel independence means each variate gets its own transformer — reduces overfitting.
- **When to use:** Multivariate forecasting; long context windows (512+ steps).
- **Key hyperparams:** `patch_len` (12-24), `stride` (8-12), `d_model` (128-256), `n_heads` (4-8), `n_layers` (2-4), `dropout` (0.2-0.3).
- **Pitfall:** Patch length must divide well into your lookback window. Poor with very short sequences.

### 1.4 TimesNet
**Temporal 2D-variation modeling — reshapes 1D series into 2D tensors.**
- Discovers multiple periodicities via FFT, reshapes time series into 2D based on periods.
- Applies 2D convolutions (Inception blocks) to capture intra- and inter-period patterns.
- **When to use:** Data with strong multi-scale periodicities (daily + weekly + monthly cycles).
- **Key hyperparams:** `d_model` (64-128), `d_ff` (128-256), `num_kernels` (6), `top_k` (3-5 periods), `e_layers` (2-3).

### 1.5 iTransformer
**Inverted Transformer — treats each variate as a token, not each timestep.**
- Transposes the standard approach: attention across variates, FFN along time.
- Captures multivariate correlations explicitly while preserving temporal patterns via FFN.
- **When to use:** Highly correlated multivariate systems (e.g., multiple commodity spreads).
- **Key hyperparams:** `d_model` (128-512), `n_heads` (4-8), `d_ff` (256-1024), `e_layers` (2-4).
- **Pitfall:** Needs enough variates to make cross-variate attention worthwhile (>5).

### 1.6 Temporal Convolutional Networks (TCN)
**Dilated causal convolutions — simple, fast, parallelizable.**
- Stack of 1D causal convolutions with exponentially increasing dilation factors.
- Receptive field grows exponentially with depth: `receptive_field = 2^(num_layers) * kernel_size`.
- **When to use:** Need fast inference; moderate sequence lengths; want a reliable baseline.
- **Key hyperparams:** `num_channels` ([64,64,64,64]), `kernel_size` (3-7), `dropout` (0.1-0.2), `dilation_base` (2).

```python
import torch.nn as nn
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.chomp = lambda x: x[:, :, :-padding] if padding > 0 else x
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.relu(self.chomp(self.conv(x))))
```

### 1.7 BiLSTM with Multi-Head Attention
**Classic but effective — strong baseline, easy to debug.**
- Bidirectional LSTM captures forward and backward temporal dependencies.
- Multi-head attention layer on top selects which timesteps matter for prediction.
- **When to use:** Small-to-medium datasets; need something reliable and interpretable.
- **Key hyperparams:** `hidden_size` (64-256), `num_layers` (1-3), `num_heads` (4-8), `dropout` (0.2-0.4).
- **Pitfall:** Bidirectional = uses future info. For strict causal prediction, use unidirectional LSTM + attention.

```python
class LSTMAttn(nn.Module):
    def __init__(self, input_dim, hidden=128, heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.attn = nn.MultiheadAttention(hidden*2, heads, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden*2, 1)
    def forward(self, x):
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        return self.fc(a[:, -1, :])  # last timestep
```

### 1.8 DeepAR
**Autoregressive probabilistic forecasting with LSTMs.**
- Outputs parameters of a probability distribution (Gaussian, negative binomial, etc.) at each step.
- Generates samples via Monte Carlo for prediction intervals.
- **When to use:** Need calibrated uncertainty estimates; count data or intermittent demand.
- **Key hyperparams:** `hidden_size` (40-160), `num_layers` (2-3), `cell_type` ('LSTM'), `distribution` ('NegativeBinomial' or 'Normal').

### 1.9 Informer
**Efficient transformer for long sequences — ProbSparse attention.**
- Replaces full O(n^2) attention with ProbSparse self-attention: O(n log n).
- Distilling layers progressively halve the sequence length.
- **When to use:** Very long sequences (1000+ timesteps) where vanilla transformers OOM.
- **Key hyperparams:** `d_model` (256-512), `n_heads` (8), `e_layers` (2-3), `d_layers` (1), `factor` (5, controls sparsity).
- **Pitfall:** Superseded by PatchTST and iTransformer for most tasks. Use only if sequence length is extreme.

---

## 2. Anti-Overfitting Arsenal (Critical for Small Datasets)

> **Rule of thumb for financial data:** If you have fewer than 5,000 training samples, overfitting is your primary enemy, not model expressiveness.

### 2.1 Temporal Cross-Validation (Blocked / Purged)
**Never use standard k-fold on time series.** Temporal leakage will give you false confidence.

- **Expanding window:** Train on [0:t], validate on [t:t+h], increment t.
- **Sliding window:** Train on [t-w:t], validate on [t:t+h], slide by step s.
- **Purged CV:** Add a gap between train and validation to prevent label leakage from overlapping targets.
- **Embargo period:** After purging, add extra buffer equal to forecast horizon.

```python
# Purged walk-forward split
def purged_walk_forward(n, train_size, val_size, gap, step):
    splits = []
    start = 0
    while start + train_size + gap + val_size <= n:
        train_idx = list(range(start, start + train_size))
        val_idx = list(range(start + train_size + gap, start + train_size + gap + val_size))
        splits.append((train_idx, val_idx))
        start += step
    return splits
```

### 2.2 Dropout Strategies
- **Standard dropout (0.1-0.4):** Apply after dense and attention layers.
- **Spatial dropout:** Drops entire feature channels in Conv1D — better for TCN.
- **Variational dropout:** Same mask across timesteps in RNNs — preserves temporal structure.
- **Attention dropout (0.05-0.15):** Applied inside attention computation, separate from layer dropout.

### 2.3 Weight Decay / L2 Regularization
- Use AdamW (decoupled weight decay), not Adam + L2. They are NOT equivalent.
- Typical values: `weight_decay=1e-4` to `1e-2`. Higher for smaller datasets.
- **Pitfall:** Do NOT apply weight decay to bias terms or LayerNorm parameters.

```python
no_decay = ['bias', 'LayerNorm.weight', 'layernorm']
params = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-3},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = torch.optim.AdamW(params, lr=1e-3)
```

### 2.4 Early Stopping with Restore-Best-Weights
- Monitor validation loss, not training loss.
- Patience: 10-30 epochs for financial data. Too low = underfitting.
- **Always restore best weights.** The last epoch is almost never the best.

### 2.5 Data Augmentation for Time Series
- **Jittering:** Add Gaussian noise (sigma=0.01-0.05 * std).
- **Window slicing:** Random sub-windows of the full lookback.
- **Magnitude warping:** Multiply by smooth random curves (cubic spline with sigma=0.2).
- **Time warping:** Smooth random distortion of time axis.
- **Window warping:** Speed up or slow down random segments.
- **Pitfall:** Aggressive augmentation can destroy financial signal. Use conservatively.

### 2.6 Mixup / CutMix for Time Series
- **Mixup:** `x_mix = lambda * x_i + (1-lambda) * x_j`, same for labels.
- **CutMix:** Replace a random time segment of one sample with another.
- Lambda from Beta(alpha, alpha) with alpha=0.2-0.4.
- **Pitfall:** Only mix within the same regime or class. Mixing bull and bear market samples creates nonsense.

### 2.7 Label Smoothing
- Replace hard targets with `y_smooth = y * (1 - epsilon) + epsilon / num_classes`.
- Typical `epsilon`: 0.05-0.1 for classification tasks.
- For regression: clip extreme target values instead.

### 2.8 Gradient Clipping
- **Max norm clipping** (preferred): `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
- Essential for RNNs/LSTMs to prevent exploding gradients.
- For transformers, max_norm of 0.5-1.0 is typical.

### 2.9 Batch Normalization vs Layer Normalization
- **BatchNorm:** Normalizes across batch dim. Bad for time series with varying distributions over time. Avoid for non-stationary financial data.
- **LayerNorm:** Normalizes across feature dim per sample. Preferred for transformers and RNNs.
- **RMSNorm:** Lighter than LayerNorm, works nearly as well. Use if speed matters.
- **Pitfall:** BatchNorm leaks info between samples in a batch. In time series this can leak temporal info.

### 2.10 Ensemble Methods
- **Snapshot ensembles:** Save checkpoints at each cosine annealing cycle minimum, average predictions.
- **Stochastic Weight Averaging (SWA):** Average weights over last K epochs. Nearly free ensemble effect.
- **Multi-seed ensembles:** Train same architecture with 3-5 different seeds, average outputs.
- **Deep ensembles:** Train N independent models, average predictions and use disagreement as uncertainty.

```python
# SWA in PyTorch
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
# After warmup epochs: swa_model.update_parameters(model) each epoch
# At end: torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

---

## 3. Feature Engineering for Financial Time Series

### 3.1 Returns and Log-Returns
- **Simple returns:** `r_t = (P_t - P_{t-1}) / P_{t-1}` — interpretable, additive over assets.
- **Log-returns:** `r_t = ln(P_t / P_{t-1})` — additive over time, approximately normal, preferred for modeling.
- **Pitfall:** Never feed raw prices into models. They are non-stationary and will not generalize.

### 3.2 Rolling Statistics
- `rolling_mean(window)`, `rolling_std(window)`, `rolling_skew(window)`, `rolling_kurtosis(window)`.
- Use multiple windows: [5, 10, 21, 63] for daily data (1wk, 2wk, 1mo, 3mo).
- Rolling z-score: `(x - rolling_mean) / rolling_std` — naturally stationary.

### 3.3 Volatility Measures
- **Parkinson (HL):** `vol = sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))` — uses high/low, more efficient than close-close.
- **Garman-Klass:** Adds open/close info. `GK = 0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2`.
- **Yang-Zhang:** Best estimator when there are overnight jumps. Combines overnight and intraday.
- **Realized vol from intraday data** if available — gold standard but requires tick data.

### 3.4 Volume / Open Interest Ratios
- Volume relative to rolling average: `vol_ratio = volume / rolling_mean(volume, 20)`.
- OI changes: `delta_oi = OI_t - OI_{t-1}` (commitment / unwinding signal).
- Volume-price divergence: Rising price + falling volume = weak trend.
- Put/call volume ratios (if options data available).

### 3.5 Calendar Features
- **Cyclical encoding** (preferred): `sin(2*pi*day/7)`, `cos(2*pi*day/7)` — avoids discontinuity.
- Day of week, day of month, month of year, week of year.
- Holiday indicators, USDA report dates, contract expiry flags, FOMC meeting dates.
- Pre/post-holiday effects (encode as separate binary features).
- **LNY-specific:** Encode days-to-LNY and days-since-LNY as features.

### 3.6 Lagged Features and Autocorrelation
- Include lags at [1, 2, 3, 5, 10, 21] for daily data.
- Partial autocorrelation function (PACF) to select significant lags.
- Cross-correlation lags between related instruments (CBOT vs DCE).
- **Pitfall:** Too many lagged features on small datasets = multicollinearity + overfitting.

### 3.7 Technical Indicators as Features
- RSI(14), MACD(12,26,9), Bollinger Band %B, ATR(14), ADX(14).
- Use z-scored versions rather than raw indicator values.
- **Pitfall:** Most technical indicators are highly correlated with each other. Use PCA or feature selection to reduce redundancy.

---

## 4. Training Best Practices

### 4.1 Learning Rate Finding (Leslie Smith's Method)
- Sweep LR from 1e-7 to 1e-1 over one epoch, plot loss vs LR.
- Pick LR where loss is still decreasing steeply (typically 1/10th of the minimum loss LR).
- Rerun for each new architecture or dataset.

```python
from torch.optim.lr_scheduler import ExponentialLR
# Quick LR range test
lr_min, lr_max, steps = 1e-7, 1e-1, 200
optimizer = torch.optim.Adam(model.parameters(), lr=lr_min)
gamma = (lr_max / lr_min) ** (1 / steps)
scheduler = ExponentialLR(optimizer, gamma=gamma)
# Train one batch per step, record loss. Plot. Pick LR at steepest descent.
```

### 4.2 OneCycleLR vs CosineAnnealingWarmRestarts
- **OneCycleLR:** Single cycle from low -> high -> low LR. Best for fixed epoch budgets. Preferred default.
- **CosineAnnealingWarmRestarts:** Multiple warm restarts. Good for snapshot ensembles.
- **Warmup:** 5-10% of total steps for transformers. LSTMs usually don't need warmup.

```python
# OneCycleLR — the default choice
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader),
    epochs=num_epochs, pct_start=0.1, anneal_strategy='cos'
)
# Step EVERY BATCH, not every epoch
```

### 4.3 Temporal Train/Val/Test Splitting Rules
- **Order matters:** Train = oldest, Val = middle, Test = newest. Never shuffle.
- **Ratios:** 70/15/15 or 80/10/10. For very small data: 80/20 with walk-forward.
- **Gap:** Insert gap >= forecast horizon between splits to prevent leakage.
- **Regime awareness:** Ensure each split contains at least one full market cycle if possible.
- **Pitfall:** If test period is a completely different regime (e.g., COVID crash), results will mislead. Use multiple test windows.

### 4.4 Batch Size Selection
- Smaller batches (16-64) act as implicit regularization — good for small datasets.
- Larger batches (128-512) train faster but may need linear LR scaling.
- **Linear scaling rule:** If you double batch size, double the LR.
- For financial time series with <5K samples: batch_size=16-32 is often optimal.

### 4.5 Mixed Precision Training
- Use `torch.amp` for 1.5-2x speedup with no accuracy loss.
- Essential for transformers on GPU; marginal benefit for small LSTMs.

```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    output = model(x)
    loss = criterion(output, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4.6 Gradient Accumulation
- Simulate larger batches without more GPU memory.
- Accumulate over N steps, then update. Effective batch = actual_batch * N.
- Essential when sequence length eats all GPU RAM.

```python
accumulation_steps = 4
for i, (x, y) in enumerate(train_loader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. Statistical Validation Methods

> **Golden rule:** A model that "works" without statistical validation is just a lucky backtest.

### 5.1 Paired t-test and Welch's t-test
- Compare model predictions vs baseline on same test set.
- **Paired t-test:** When comparing same samples (model A error vs model B error per sample).
- **Welch's t-test:** When comparing two independent sets of results with potentially unequal variances.
- Null hypothesis: no difference in mean error. Reject at p < 0.05.

```python
from scipy.stats import ttest_rel, ttest_ind
# Paired: same test samples
t_stat, p_val = ttest_rel(errors_model_a, errors_model_b)
# Independent: different runs
t_stat, p_val = ttest_ind(errors_model_a, errors_model_b, equal_var=False)
```

### 5.2 Mann-Whitney U Test
- Non-parametric alternative to t-test. No normality assumption.
- Use when error distributions are skewed (common in financial prediction).
- `scipy.stats.mannwhitneyu(errors_a, errors_b, alternative='two-sided')`

### 5.3 Kolmogorov-Smirnov Test
- Tests whether two distributions differ. Sensitive to shape, not just location.
- Useful for checking if prediction distribution matches actual return distribution.
- `scipy.stats.ks_2samp(predicted_returns, actual_returns)`

### 5.4 Bootstrap Confidence Intervals
- Resample test predictions with replacement, compute metric each time.
- 1000-10000 bootstrap iterations for stable CIs.
- Report 95% CI: [2.5th percentile, 97.5th percentile].

```python
import numpy as np
def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=5000, ci=0.95):
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    return np.percentile(scores, [100*alpha, 100*(1-alpha)])
```

### 5.5 Permutation Tests
- Shuffle labels, recompute metric many times to build null distribution.
- p-value = fraction of permuted scores >= observed score.
- Gold standard for "is this model actually predictive?" testing.

### 5.6 Effect Size Measures
- **Cohen's d:** `d = (mean_a - mean_b) / pooled_std`. Small=0.2, medium=0.5, large=0.8.
- **Cliff's delta:** Non-parametric. Proportion of pairs where A > B minus proportion B > A.
- **Always report effect size alongside p-values.** Statistical significance != practical significance.

---

## 6. Model Interpretability

### 6.1 SHAP Values for Time Series
- Use KernelSHAP or DeepSHAP to explain individual predictions.
- Aggregate SHAP values over time to see which features matter at which lags.
- **Pitfall:** SHAP is slow for large models. Use a subsample of test data (100-500 samples).

```python
import shap
explainer = shap.DeepExplainer(model, background_data[:100])
shap_values = explainer.shap_values(test_data[:50])
shap.summary_plot(shap_values, feature_names=feature_names)
```

### 6.2 Attention Weight Visualization
- Extract and plot attention maps from transformer layers.
- Shows which timesteps the model attends to for each prediction.
- **Pitfall:** Attention weights != feature importance. They show what the model looks at, not causal contribution. Use SHAP for true importance.

### 6.3 Feature Ablation Studies
- Systematically remove one feature (or group) at a time, retrain, measure degradation.
- Expensive but most reliable method for feature importance.
- **Shortcut:** Zero out features at test time (permutation importance) — cheaper, less reliable.

### 6.4 Partial Dependence Plots
- Fix all features except one, vary it across its range, plot predicted output.
- Shows marginal effect of each feature on predictions.
- Use ICE (Individual Conditional Expectation) plots for per-sample view.

---

## 7. Production-Ready Patterns

### 7.1 Model Checkpointing and Versioning
- Save full state: model weights, optimizer state, epoch, best metric, hyperparameters.
- Use deterministic naming: `model_{arch}_{date}_{metric:.4f}.pt`.
- Keep top 3 checkpoints, delete the rest to save disk.

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
    'config': config_dict,
    'feature_names': feature_names,
    'scaler_params': {'mean': scaler.mean_, 'scale': scaler.scale_},
}
torch.save(checkpoint, f'checkpoints/model_{epoch:03d}_{val_loss:.4f}.pt')
```

### 7.2 Inference Pipeline Design
- **Always** store preprocessing parameters (scaler mean/std, feature order) with the model.
- Input validation: check for NaNs, correct shape, expected value ranges.
- Deterministic inference: `model.eval()`, `torch.no_grad()`, fixed seeds for dropout if using MC dropout.

```python
@torch.no_grad()
def predict(model, raw_data, scaler, device='cpu'):
    model.eval()
    x = scaler.transform(raw_data)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    pred = model(x).cpu().numpy()
    return scaler.inverse_transform_target(pred)
```

### 7.3 Incremental Updates Without Full Retraining
- **Fine-tuning:** Freeze all layers except final 1-2. Train on new data with low LR (1/10th original).
- **Online learning:** Update with each new batch. Use very small LR and gradient clipping.
- **Expanding window retrain:** Periodically retrain from scratch on all data. Most reliable but expensive.
- **Trigger-based retraining:** Monitor prediction error; retrain when rolling error exceeds threshold.
- **Pitfall:** Fine-tuning on too little new data causes catastrophic forgetting. Always validate on holdout from BOTH old and new periods.

### 7.4 Prediction Confidence Intervals
- **MC Dropout:** Enable dropout at inference time, run N forward passes (30-100), compute mean and std.
- **Quantile regression:** Train model to output [10th, 50th, 90th] percentiles directly.
- **Conformal prediction:** Post-hoc calibration. Split calibration set, compute residuals, use quantile of residuals as interval width. Distribution-free coverage guarantees.
- **Deep ensembles:** Variance across ensemble members = epistemic uncertainty.

```python
# MC Dropout for uncertainty
def mc_predict(model, x, n_samples=50):
    model.train()  # keep dropout active
    preds = torch.stack([model(x) for _ in range(n_samples)])
    return preds.mean(dim=0), preds.std(dim=0)  # mean prediction, uncertainty
```

---

## 8. Common Pitfalls Checklist

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Feeding raw prices (not returns) | Model memorizes levels, fails on new data | Use returns or log-returns |
| Standard k-fold CV on time series | Massive lookahead bias, inflated metrics | Use temporal/purged CV |
| No gap between train/val splits | Label leakage from overlapping windows | Add gap >= forecast horizon |
| Using BatchNorm on financial data | Leaks batch-level info, fails on regime changes | Use LayerNorm or RMSNorm |
| Too many features on small dataset | Overfitting, spurious correlations | Feature selection, PCA, regularization |
| Optimizing only MSE for directional prediction | Low MSE != profitable trading signal | Add directional accuracy or Sharpe-based loss |
| Ignoring transaction costs in evaluation | Apparent alpha disappears after costs | Include spread + commission in backtest |
| Reporting only point estimates | No way to assess reliability | Always report confidence intervals |
| Same random seed for train and eval | Subtle data contamination possible | Use separate RNG streams |
| Not checking for NaN propagation | Silent model corruption | Add NaN checks in data pipeline and loss |

---

## 9. Quick Architecture Selection Guide

```
Dataset size < 500 samples:
  -> Linear model or Ridge regression. DL will overfit.

Dataset size 500-2000:
  -> BiLSTM + Attention OR TCN with heavy regularization.
  -> Ensemble 3-5 models with different seeds.

Dataset size 2000-10000:
  -> PatchTST, N-HiTS, or TFT depending on need.
  -> PatchTST for pure forecasting accuracy.
  -> TFT if interpretability matters.

Dataset size > 10000:
  -> Full TFT or iTransformer.
  -> TimesNet if strong periodicities exist.
  -> Can afford deeper models with less regularization.

Univariate only:
  -> N-HiTS > N-BEATS > TCN.

Multivariate with known future covariates:
  -> TFT (designed for this exact scenario).

Need uncertainty estimates:
  -> DeepAR or MC Dropout on any architecture.

Need fastest inference:
  -> TCN or N-HiTS (no recurrence, fully parallel).
```

---

*End of reference. When in doubt: simpler model + good features + proper validation > complex model + raw data + random split.*
