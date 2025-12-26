# Rapid Spectrogram-Based Characterization of Gravitational-Wave Events

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project develops a **rapid classification system** for gravitational-wave compact binary coalescence (CBC) events, distinguishing between **Binary Black Hole (BBH)** and **Neutron Star-present (BNS/NSBH)** events using spectrogram-based machine learning.

### Key Features

- **Two-phase approach**: Interpretable baseline (Random Forest) + Deep Learning (CNN + Physics-Informed NN)
- **10,000+ synthetic samples** via PyCBC injection pipeline
- **Physics-informed loss function** incorporating chirp mass constraints
- **Domain shift evaluation** from O1-O3 → O4a
- **A100 GPU optimized** training pipeline

## Project Structure

```
project/
├── configs/              # Configuration files (YAML)
├── data/                 # Raw, synthetic, and processed data
├── manifests/            # Event manifests and data splits
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Reusable Python modules
├── scripts/              # CLI automation scripts
├── outputs/              # Models, figures, results
└── tests/                # Unit tests
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- ~50GB disk space for data

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gw-classification.git
cd gw-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Google Colab (A100)

For Phase 2 training, use Google Colab with A100 runtime:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo to Colab
!git clone https://github.com/yourusername/gw-classification.git
%cd gw-classification
!pip install -r requirements.txt
```

## Quick Start

### 1. Download Real Events

```bash
python scripts/download_events.py --config configs/config.yaml
```

### 2. Generate Synthetic Data (Phase 2)

```bash
python scripts/generate_synthetic.py --config configs/config.yaml --n-samples 10000
```

### 3. Train Baseline Model (Phase 1)

```bash
python scripts/train_baseline.py --config configs/config.yaml
```

### 4. Train CNN (Phase 2)

```bash
python scripts/train_cnn.py --config configs/config.yaml --gpu
```

### 5. Evaluate

```bash
python scripts/evaluate.py --config configs/config.yaml --model outputs/models/phase2/best.pt
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Explore GWTC events and strain data |
| `02_data_download.ipynb` | Download real events from GWOSC |
| `03_synthetic_injection.ipynb` | Generate synthetic CBC signals |
| `04_spectrogram_generation.ipynb` | Create spectrograms from strain |
| `05_feature_extraction.ipynb` | Extract quantitative features |
| `06_phase1_baseline.ipynb` | Train Random Forest baseline |
| `07_phase2_cnn.ipynb` | Train CNN on A100 |
| `08_physics_informed.ipynb` | Physics-informed loss experiments |
| `09_evaluation.ipynb` | Full evaluation and analysis |

## Results

### Phase 1 (Baseline)

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| AUC-ROC | TBD |
| NS-present Recall | TBD |

### Phase 2 (CNN + PINN)

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >90% | TBD |
| False Positive Rate | <10% | TBD |
| NS-present Recall | >85% | TBD |
| Inference Latency | <1 sec | TBD |

## Citation

If you use this code, please cite:

```bibtex
@misc{gw-classification-2025,
  author = {Your Name},
  title = {Rapid Spectrogram-Based Characterization of Gravitational-Wave Events},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/gw-classification}
}
```

## Acknowledgments

- LIGO Scientific Collaboration for GWOSC data
- PyCBC team for waveform generation tools
- GWpy developers for data access tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
