# ADR-017 — Google Colab A100 for Training

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-017 |
| **Title** | Google Colab A100 GPU for Phase 2 Training |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Phase 2 of this project involves training deep CNNs on a dataset of 10,000+ spectrograms. Deep learning training has significant computational requirements:

- **Memory**: Large batch sizes and model weights require substantial GPU VRAM
- **Compute**: Matrix operations benefit enormously from GPU parallelization  
- **Time**: Training without GPU can take days vs. hours with GPU

Local compute options are limited:
- Laptop CPU: Too slow for practical deep learning
- Consumer GPU (if available): Limited VRAM (typically 4-12 GB)
- No access to institutional HPC clusters

**Google Colab** offers cloud-based Jupyter notebooks with GPU access. The premium tier provides access to **NVIDIA A100 GPUs**, the current flagship data center GPU with:
- 40-80 GB VRAM
- Tensor cores for accelerated training
- High memory bandwidth

---

## 2. Problem Statement

We need GPU compute for Phase 2 CNN training. The solution must:
1. Provide sufficient GPU memory for batch training
2. Be accessible without institutional resources
3. Support PyTorch deep learning workflows
4. Allow training runs of several hours
5. Enable reproducible experiments

**Key Question**: What compute platform should we use for Phase 2 deep learning?

---

## 3. Decision Drivers

1. **GPU Access**: Must have modern GPU for practical training times

2. **Memory Requirements**: 10K spectrograms (224×224×3) = ~2 GB raw; need headroom for batches, model, gradients

3. **Cost**: Student/science fair budget is limited

4. **Accessibility**: Should be accessible from any computer with browser

5. **PyTorch Support**: Must run PyTorch 2.0+ with CUDA

6. **Session Length**: Training may require multiple hours

7. **Notebook Integration**: Jupyter notebook format preferred for documentation

---

## 4. Considered Options

### Option A: Google Colab Pro/Pro+ with A100

**Description**: Use Google Colab with paid subscription ($10-50/month) for access to A100 GPUs.

**Specs**:
- A100 40GB or 80GB
- Up to 24 hour sessions
- Priority GPU access
- High RAM instances available

**Pros**:
- State-of-the-art GPU (A100)
- 40-80 GB VRAM enables large batches
- Tensor cores + bf16 support
- Jupyter notebook interface
- Easy data upload/download via Drive
- No setup required

**Cons**:
- Subscription cost (~$10-50/month)
- Session time limits (must reconnect)
- GPU availability not guaranteed
- Data must be uploaded each session

### Option B: Free Google Colab (T4/P100)

**Description**: Use free Colab tier with limited GPU access.

**Specs**:
- T4 (16 GB) or P100 (16 GB) or K80 (12 GB)
- ~12 hour sessions
- No priority access

**Pros**:
- Free
- Sufficient for basic training
- Same interface as Pro

**Cons**:
- Limited VRAM (16 GB max)
- Slower GPUs
- Usage limits and disconnections
- May not get GPU at all during peak times

### Option C: Kaggle Notebooks

**Description**: Use Kaggle's free notebook environment with GPU.

**Specs**:
- P100 or T4 (16 GB)
- 30 hours/week GPU quota

**Pros**:
- Free
- Generous weekly quota
- Good for data science

**Cons**:
- Limited VRAM
- Session restrictions
- Less flexible than Colab

### Option D: Local GPU (If Available)

**Description**: Use personal/school GPU if available.

**Pros**:
- No session limits
- No upload required
- Full control

**Cons**:
- May not be available
- Consumer GPUs have less VRAM
- Setup complexity
- Noise/heat/power concerns

---

## 5. Decision Outcome

**Chosen Option**: Option A — Google Colab Pro/Pro+ with A100

**Rationale**:

The A100 access transforms what's possible in the project timeline:

1. **Batch Size Impact**:
   - T4 (16 GB): batch_size ≈ 32 for ResNet50
   - A100 (40 GB): batch_size ≈ 128-256 for ResNet50
   - Larger batches → faster convergence, more stable gradients

2. **Model Capacity**:
   - T4: Limited to ResNet18 or EfficientNet-B0
   - A100: Can run ResNet50, ResNet101, EfficientNet-B4+
   - Bigger models may capture more nuanced features

3. **Mixed Precision Training**:
   - A100 has excellent bfloat16 support
   - 2x speedup with automatic mixed precision
   - T4 has limited bf16 support

4. **Training Time**:
   - 10K samples, 50 epochs, A100: ~1-2 hours
   - Same on T4: ~4-8 hours
   - Faster iteration enables more experiments

5. **Memory Headroom**:
   - 40 GB allows loading full dataset in VRAM
   - No disk I/O bottleneck during training
   - Can experiment with larger architectures

**Cost Justification**: $10-50/month for 2-3 months = $30-150 total. This is reasonable for a competitive science fair project and dramatically improves capabilities.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Training speed**: Hours instead of days
- **Larger batches**: 128-256 vs 32 → better gradient estimates
- **Bigger models**: Can try ResNet50/101, EfficientNet variants
- **Mixed precision**: Free 2x speedup with bf16
- **Full dataset in memory**: No I/O bottleneck
- **More experiments**: Fast training enables hyperparameter search

### 6.2 Negative Consequences

- **Cost**: $10-50/month subscription
- **Session management**: Must save checkpoints, handle disconnections
- **Data upload**: Must transfer data to Colab environment
- **Availability risk**: A100 not always available (rare)
- **Dependency**: Project completion depends on external service

### 6.3 Neutral Consequences

- Notebooks must be Colab-compatible
- Need Google Drive for data persistence
- Must design for session interruption resilience

---

## 7. Validation

**Success Criteria**:
- A100 GPU accessible via Colab
- Training completes within session time limit
- Checkpoints saved and loadable
- Results reproducible across sessions

**Review Date**: Week 5 (Phase 2 start)

**Reversal Trigger**:
- A100 unavailable for extended period (>1 week)
- Cost becomes prohibitive
- Colab service issues prevent training

---

## 8. Implementation Notes

### 8.1 A100-Optimized Training Configuration

```python
# configs/model_params.yaml (A100 section)

a100_training:
  # Larger batches possible with A100
  batch_size: 128  # vs 32 on consumer GPU
  
  # Mixed precision for 2x speedup
  mixed_precision: true
  precision: "bf16"  # bfloat16 preferred on A100
  
  # Model architecture
  backbone: "resnet50"  # Can go bigger than ResNet18
  
  # Data loading
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  
  # Optimizer
  optimizer: "adamw"
  learning_rate: 3e-4
  weight_decay: 0.01
  
  # Training
  epochs: 50
  gradient_accumulation_steps: 1
```

### 8.2 Colab Notebook Setup Cell

```python
# === COLAB SETUP ===
# Run this cell first to configure environment

# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q gwpy pycbc timm wandb

# Clone/copy project
PROJECT_DIR = '/content/drive/MyDrive/gw_project'
%cd {PROJECT_DIR}
```

### 8.3 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize
scaler = GradScaler()
model = model.cuda()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(dtype=torch.bfloat16):
            outputs = model(batch['input'].cuda())
            loss = criterion(outputs, batch['label'].cuda())
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 8.4 Checkpoint Management

```python
import os
from pathlib import Path

CHECKPOINT_DIR = Path('/content/drive/MyDrive/gw_project/checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

def save_checkpoint(model, optimizer, epoch, metrics, name='checkpoint'):
    """Save training checkpoint to Drive."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    path = CHECKPOINT_DIR / f'{name}_epoch{epoch}.pt'
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(model, optimizer, path):
    """Load checkpoint and resume training."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

# Auto-save every N epochs
SAVE_EVERY = 5
for epoch in range(start_epoch, num_epochs):
    train_one_epoch(...)
    if (epoch + 1) % SAVE_EVERY == 0:
        save_checkpoint(model, optimizer, epoch + 1, current_metrics)
```

### 8.5 Data Loading Strategy

```python
# Option 1: Load from Drive
data_path = '/content/drive/MyDrive/gw_project/data/spectrograms/'

# Option 2: Copy to local (faster I/O)
!cp -r /content/drive/MyDrive/gw_project/data/spectrograms /content/data/
data_path = '/content/data/spectrograms/'

# Optimized DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### 8.6 A100 vs Other GPU Comparison

| Capability | K80 | T4 | P100 | A100 |
|------------|-----|-----|------|------|
| VRAM | 12 GB | 16 GB | 16 GB | 40-80 GB |
| Max Batch (ResNet50) | 16 | 32 | 32 | 128-256 |
| bf16 Support | No | Limited | No | **Yes** |
| Training Time (10K) | ~12h | ~6h | ~4h | **~1.5h** |
| Recommended Model | ResNet18 | ResNet18/34 | ResNet34/50 | **ResNet50/101** |

### 8.7 Session Resilience

```python
# Auto-reconnect handling
import signal
import sys

def save_on_interrupt(signum, frame):
    """Save checkpoint if session interrupted."""
    print("\n⚠ Interrupt detected, saving checkpoint...")
    save_checkpoint(model, optimizer, current_epoch, current_metrics, 'interrupt')
    sys.exit(0)

signal.signal(signal.SIGTERM, save_on_interrupt)
signal.signal(signal.SIGINT, save_on_interrupt)
```

---

## 9. References

- [Google Colab Pro](https://colab.research.google.com/signup): Subscription options
- [A100 Specifications](https://www.nvidia.com/en-us/data-center/a100/): Hardware details
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html): AMP documentation
- [Colab Best Practices](https://colab.research.google.com/notebooks/pro.ipynb): Tips for Pro users
- [Training Efficiency](https://arxiv.org/abs/2001.04063): Deep learning optimization

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
