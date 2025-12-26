# ADR-009 — Reproducibility Requirements

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-009 |
| **Title** | Reproducibility: Immutable Manifests + Deterministic Pipelines |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Scientific research demands reproducibility—the ability for others to repeat experiments and obtain consistent results. In machine learning projects, reproducibility is particularly challenging due to:

1. **Data variability**: Data sources may change over time
2. **Randomness**: Training involves random initialization, shuffling, sampling
3. **Software versions**: Library updates can change behavior
4. **Hardware differences**: GPU vs CPU, different GPU architectures
5. **Implicit parameters**: Undocumented settings affecting results

For a science fair project, reproducibility serves multiple purposes:
- **Judging credibility**: Judges must trust that results are genuine
- **Scientific integrity**: Work must be verifiable by others
- **Self-protection**: Ability to recreate results if issues arise
- **Collaboration**: Others can build on reproducible work

Gravitational-wave research has particularly high reproducibility standards, as GWOSC and LVK emphasize open science and data sharing.

---

## 2. Problem Statement

We need to establish practices ensuring that:
1. Experiments can be exactly reproduced weeks or months later
2. Another researcher could replicate our results from documentation alone
3. Results don't depend on undocumented "magic" settings
4. Data provenance is traceable to original sources

**Key Question**: What specific mechanisms will ensure full reproducibility of all experiments and results in this project?

---

## 3. Decision Drivers

1. **Science Fair Requirements**: Judges expect documented methodology and reproducible claims

2. **Scientific Standards**: GW community has high expectations for open science

3. **Development Robustness**: Ability to return to known-good states during debugging

4. **Documentation Overhead**: Reproducibility mechanisms should not impede rapid iteration

5. **Verification**: Must be able to verify results haven't changed unexpectedly

6. **Collaboration Potential**: Future users/collaborators need clear starting point

---

## 4. Considered Options

### Option A: Full Reproducibility Infrastructure

**Description**: Implement comprehensive reproducibility measures including manifests, version pinning, random seed control, configuration files, and automated documentation.

**Components**:
- Event manifests with exact GPS times and data versions
- requirements.txt with pinned versions
- Random seeds in all stochastic operations
- YAML configuration files for all parameters
- Automated logging of all experiment runs
- Version control for code and configs

**Pros**:
- Maximum reproducibility
- Professional-grade infrastructure
- Enables confident comparison of experiments
- Protects against "it worked yesterday" problems
- Impressive to judges

**Cons**:
- Setup overhead
- Must maintain manifest consistency
- Some additional complexity

### Option B: Minimal Reproducibility (Seeds Only)

**Description**: Just set random seeds, rely on informal documentation.

**Pros**:
- Quick to implement
- Minimal overhead

**Cons**:
- Data versions may drift
- Parameters scattered across code
- No systematic logging
- Hard to reproduce exactly later

### Option C: Container-Based Reproducibility

**Description**: Use Docker containers to capture entire environment.

**Pros**:
- Complete environment capture
- Guaranteed identical execution

**Cons**:
- Overkill for science fair scope
- Complex setup
- Large containers
- Learning curve

---

## 5. Decision Outcome

**Chosen Option**: Option A — Full Reproducibility Infrastructure

**Rationale**:

Reproducibility is a core scientific value. The overhead of implementing proper reproducibility mechanisms is modest compared to the benefits:

1. **Manifest System**: Create immutable records of exactly which data was used
2. **Configuration Files**: Centralize all parameters in version-controlled YAML
3. **Random Seed Control**: Deterministic results for any given seed
4. **Version Pinning**: Exact library versions in requirements.txt
5. **Experiment Logging**: Record every training run's parameters and results

This infrastructure pays dividends throughout the project:
- Debug issues by comparing to known-good runs
- Confidently report results knowing they're reproducible
- Enable judges/reviewers to verify claims
- Support future work building on this project

---

## 6. Consequences

### 6.1 Positive Consequences

- **Verifiable results**: Any result can be independently reproduced
- **Debugging support**: Can isolate changes when things break
- **Professional quality**: Demonstrates scientific rigor
- **Collaboration ready**: Others can easily contribute or extend
- **Future-proofing**: Can return to project months later
- **Comparison validity**: Fair comparison between experiments

### 6.2 Negative Consequences

- **Initial setup time**: Must create infrastructure before experiments
- **Maintenance burden**: Must update manifests when data changes
- **Discipline required**: Must actually use the systems consistently

### 6.3 Neutral Consequences

- Project structure includes configs/, manifests/, logs/ directories
- All scripts must accept configuration parameters
- Must document any manual steps

---

## 7. Validation

**Success Criteria**:
- Can reproduce any reported result from manifest + config + code
- Results match within floating-point tolerance across runs
- Another person can set up and run experiments from documentation
- All experiments logged with full parameters

**Review Date**: Ongoing throughout project

**Reversal Trigger**:
- Reproducibility overhead significantly impedes progress
- Find simpler approach achieving same goals

---

## 8. Implementation Notes

### 8.1 Manifest System

#### Event Manifest (`manifests/event_manifest.csv`)
```csv
event_name,gps_time,run,detector,class_label,source,download_date,data_version
GW150914,1126259462.4,O1,L1,BBH,GWTC-1,2024-01-15,v1
GW170817,1187008882.4,O2,L1,BNS,GWTC-1,2024-01-15,v1
...
```

#### Synthetic Manifest (`manifests/synthetic_manifest.csv`)
```csv
injection_id,m1,m2,spin1z,spin2z,snr,noise_segment,class_label,waveform,seed
syn_0001,25.3,18.7,0.12,-0.05,15.3,O2_noise_001,BBH,IMRPhenomD,42
syn_0002,1.4,1.3,0.01,0.02,12.1,O3a_noise_042,BNS,TaylorF2,43
...
```

#### Train/Test Split (`manifests/splits.json`)
```json
{
  "split_version": "v1.0",
  "created": "2024-01-20",
  "random_seed": 42,
  "train": ["GW150914", "GW151012", "..."],
  "val": ["GW170104", "..."],
  "test_o4a": ["GW230529_181500", "..."]
}
```

### 8.2 Configuration System

#### Main Config (`configs/config.yaml`)
```yaml
# Data acquisition
data:
  sample_rate: 4096
  window_seconds: 32
  detector: "L1"
  cache_dir: "data/raw"

# Preprocessing
preprocessing:
  bandpass_low: 20
  bandpass_high: 500
  whiten: true
  
# Spectrogram
spectrogram:
  fft_length: 0.5  # seconds
  overlap: 0.25    # seconds
  freq_min: 20
  freq_max: 500
  
# Features
features:
  ridge_method: "peak_tracking"
  normalize: "per_event"

# Random seeds
seeds:
  global: 42
  numpy: 42
  sklearn: 42
  torch: 42

# Outputs
output:
  models_dir: "outputs/models"
  figures_dir: "outputs/figures"
  results_dir: "outputs/results"
  logs_dir: "outputs/logs"
```

### 8.3 Random Seed Control

```python
import random
import numpy as np
import torch

def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Call at start of every script
set_all_seeds(config['seeds']['global'])
```

### 8.4 Version Pinning

#### requirements.txt
```
# Core dependencies - pinned versions
numpy==1.24.3
scipy==1.11.3
pandas==2.0.3
matplotlib==3.8.0
seaborn==0.13.0

# GW-specific
gwpy==3.0.4
pycbc==2.2.0

# ML
scikit-learn==1.3.2
xgboost==2.0.2
torch==2.1.0
torchvision==0.16.0
timm==0.9.12

# Utilities
pyyaml==6.0.1
h5py==3.10.0
tqdm==4.66.1
```

### 8.5 Experiment Logging

```python
import json
import datetime
from pathlib import Path

def log_experiment(config, metrics, model_path, log_dir="outputs/logs"):
    """Log experiment with full parameters and results."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "model_path": str(model_path),
        "git_hash": get_git_hash(),  # if using git
        "python_version": sys.version,
    }
    
    log_path = Path(log_dir) / f"experiment_{log_entry['timestamp']}.json"
    with open(log_path, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    return log_path
```

### 8.6 Directory Structure for Reproducibility

```
project/
├── configs/
│   ├── config.yaml              # Main configuration
│   ├── injection_params.yaml    # Synthetic generation params
│   └── model_params.yaml        # Model hyperparameters
├── manifests/
│   ├── event_manifest.csv       # Real events
│   ├── synthetic_manifest.csv   # Synthetic events
│   └── splits.json              # Train/val/test splits
├── outputs/
│   ├── logs/                    # Experiment logs
│   │   └── experiment_2024-01-20T14:30:00.json
│   ├── models/                  # Saved models
│   └── results/                 # Metrics, predictions
└── requirements.txt             # Pinned dependencies
```

### 8.7 Verification Script

```python
def verify_reproducibility(config_path, expected_metrics):
    """Verify that experiment produces expected results."""
    config = load_config(config_path)
    set_all_seeds(config['seeds']['global'])
    
    # Run experiment
    metrics = run_experiment(config)
    
    # Compare
    for key, expected in expected_metrics.items():
        actual = metrics[key]
        if abs(actual - expected) > 1e-6:
            raise ValueError(f"{key}: expected {expected}, got {actual}")
    
    print("✓ Reproducibility verified")
```

---

## 9. References

- [Reproducible ML Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf): Best practices
- [GWOSC Reproducibility](https://gwosc.org/about/): Open data philosophy
- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html): Framework-specific guidance
- [Papers With Code](https://paperswithcode.com/): Reproducibility in ML research
- [FAIR Principles](https://www.go-fair.org/fair-principles/): Findable, Accessible, Interoperable, Reusable

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
