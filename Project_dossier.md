# Project Dossier

## Project Title

**Rapid Spectrogram-Based Characterization of Gravitational-Wave Events Using Quantitative Feature Extraction and Machine Learning**

## 1) Executive Summary

This project tests whether **quantitative features extracted from time–frequency spectrograms** can rapidly classify compact binary coalescences into **BBH vs NS-present (BNS/NSBH)** using only data products that can be produced quickly after a trigger. The project follows a **two-phase approach**:

* **Phase 1 (Baseline):** Feature-based ML (Random Forest/XGBoost) on ~90 real GWTC events
* **Phase 2 (Deep):** CNN + Physics-Informed Neural Network on 10,000+ synthetic + real samples

Training uses **O1–O3** events (real) plus **PyCBC-generated synthetic injections**; generalization is evaluated on **O4a** (first public portion of O4), which provides a strong "new-era detector conditions" test. GWOSC provides calibrated strain data, event lists, and documentation needed for reproducible workflows. ([GW Open Science Center][1])

### Novelty Claims
* First physics-informed rapid CBC classifier using spectrogram features
* Synthetic-to-real transfer learning validation for GW classification
* Quantified speedup vs traditional parameter estimation (<1 sec vs hours)

---

## 2) Research Questions and Hypotheses

### Primary Question

Can **spectrogram-derived quantitative features** rapidly distinguish **BBH vs NS-present (BNS/NSBH)** with statistically meaningful accuracy?

### Hypotheses

* **H1:** Spectrogram morphology (chirp rate, duration, peak frequency, band-energy evolution) contains enough information to classify event type above baseline.
* **H2:** A model trained on **O1–O3** (real + synthetic) retains performance on **O4a**, despite changes in detector sensitivity and noise characteristics (domain shift).
* **H3:** A physics-informed model that incorporates chirp mass constraints ($f(t) \propto M_c^{-5/8}$) will outperform pure data-driven approaches.
* **H4:** Synthetic-to-real transfer learning is viable for GW classification tasks.

### Classification Strategy

**2-class problem: BBH vs NS-present**
* Rationale: O1-O3 contains only ~5-6 NS-involved events (2 BNS, 2-4 NSBH), making 3-class statistically fragile
* "NS-present" is operationally meaningful for multi-messenger alerts
* 3-class (BBH/BNS/NSBH) noted as future work pending more events

---

## 3) Data Plan

### 3.1 Data Source (Public, Reproducible)

**GWOSC (Gravitational-Wave Open Science Center)** is the official open-data platform for LVK strain releases, event catalogs, and documentation. ([GW Open Science Center][1])

### 3.2 Observing Runs and Split

* **Training:** O1 + O2 + O3 (O3a + O3b)

  * O3a / O3b dates and data availability are documented by GWOSC. ([GW Open Science Center][2])
  * Use GWTC lists (GWTC-1, GWTC-2.1, GWTC-3) via the GWOSC event portal. ([GW Open Science Center][3])
* **Test:** **O4a** (public O4 segment)

  * GWOSC lists **O4a time range** and detectors (H1, L1). ([GW Open Science Center][1])

**Important clarity:** Public “O4 test set” should be defined as **O4a**, because that’s the open-data release currently available through GWOSC (H1 and L1). ([GW Open Science Center][1])

### 3.3 What You Download

**Real Events: Event-centered strain segments**

* For each event, download a fixed window around merger time (e.g., ±32s) for **L1** and optionally **H1**.
* This is much smaller than full-run strain, and it matches your task: "rapid classification right after trigger."
* Expected: ~90 real events from GWTC (O1-O3)

GWOSC provides bulk release pages and run docs (e.g., O3a/O3b, O4a). ([GW Open Science Center][1])

### 3.4 Synthetic Data Generation (Phase 2)

**PyCBC Injection Pipeline:**

* Generate synthetic CBC waveforms using `pycbc.waveform`
* Inject into real LIGO noise segments from GWOSC
* Target: **10,000+ total samples** (5,000 BBH + 5,000 NS-present)

**Waveform Parameters:**

| Class | Mass Range (M☉) | Spin | Waveform Approximant |
|-------|-----------------|------|----------------------|
| BBH | m1, m2: 5-100 | χ: 0-0.99 | IMRPhenomD |
| BNS | m1, m2: 1-2.5 | χ: 0-0.05 | TaylorF2 |
| NSBH | m1: 5-50, m2: 1-2.5 | χ1: 0-0.99, χ2: 0-0.05 | IMRPhenomNSBH |

**Injection Protocol:**
1. Sample parameters from astrophysically motivated priors
2. Generate waveform at various SNRs (8-30)
3. Inject into random noise segments from O1-O3
4. Apply same preprocessing as real events

**Benefits:**
* Solves class imbalance (real data has ~85 BBH vs ~5 NS-present)
* Enables statistically robust training
* Tests synthetic-to-real generalization

### 3.5 Detectors

* Primary: **Livingston (L1)**
* Optional improvement: add **Hanford (H1)** and fuse features (early fusion = concatenation; late fusion = average model probabilities).

O4a open release currently lists H1 and L1. ([GW Open Science Center][1])

### 3.6 Download Method (Scalable)

Use **GWpy** to fetch open data segments directly (no manual clicking).

* `TimeSeries.fetch_open_data()` is GWpy’s interface to GWOSC. ([gwpy.github.io][4])

---

## 4) End-to-End Pipeline (What the project “looks like”)

### Stage 0 — Build the Event Table (Ground Truth + Split)

Create a table with columns like:

* `event_name`, `run` (O1/O2/O3a/O3b/O4a), `gps_time`
* `source_class_label`: BBH / BNS / NSBH
* `detectors_available`: H1, L1 (and V1 if you choose)
  Use GWOSC event list / GWTC pages as the source of event names and metadata. ([GW Open Science Center][5])

**Label policy (important):**
Use the catalog’s classification (or masses thresholds if classification isn’t explicit). The cleanest is to use confident event sets (GWTC-x confident) to reduce label noise. ([GW Open Science Center][3])

---

### Stage 1 — Data Acquisition (Large-scale but manageable)

For each event and detector:

1. Fetch strain in `[t0 - W, t0 + W]` (W = 16–64 seconds).
2. Save raw segment to disk for reproducibility (HDF5 or GWpy format).

Why this matches your “rapid” goal: spectrograms can be made from these short windows immediately.

---

### Stage 2 — Preprocessing (Consistency is everything)

Apply the same preprocessing to every segment:

* Detrend / remove mean
* Bandpass (typical: 20–512 Hz for BBH-heavy; if including BNS, you may extend higher)
* Whitening (recommended to reduce noise differences between runs)
* Optional notch filters (60 Hz and harmonics)

*Goal:* reduce domain shift across O1–O4a while keeping chirp structure.

---

### Stage 3 — Spectrogram Generation (Your “data product”)

Compute time–frequency representation with fixed parameters:

* FFT length (e.g., 0.5–2 s)
* Overlap (e.g., 50%)
* Frequency range cropping (keep relevant band)

Outputs:

* **Spectrogram image** (for CNN route) and/or
* **Spectrogram matrix** (for numeric feature extraction)

---

### Stage 4 — Feature Extraction (Core novelty)

Turn each spectrogram into a compact, quantitative feature vector. Examples:

**Chirp geometry**

* Dominant ridge slope statistics (mean/median slope)
* Chirp duration above threshold
* Time to peak frequency

**Frequency-energy distribution**

* Peak frequency
* Band power ratios (e.g., 30–80 vs 80–200 vs 200–500 Hz)
* Spectral centroid / bandwidth vs time

**Texture/shape descriptors (optional, advanced)**

* Spectral entropy
* GLCM texture stats (contrast, homogeneity)
* Gradient orientation histogram features

Output: `features[event, detector] -> vector`

---

### Stage 5 — Models (Two-Phase Plan)

---

#### PHASE 1: Interpretable Baseline (Weeks 1-4)

**Track A: Feature-based ML**

* Random Forest / XGBoost with class weights
* Input: Hand-crafted spectrogram features (~10-20 features)
* Pros: interpretable, works with ~90 real events, easy ablations
* Output: class probabilities + feature importance
* Target: Establish baseline performance, identify important features

---

#### PHASE 2: Deep Learning + Physics-Informed (Weeks 5-8)

**Track B: CNN Classification**

* Architecture: ResNet18 or custom CNN (pretrained on ImageNet)
* Input: Spectrogram images (224×224 PNG)
* Training: 10,000+ synthetic + real samples
* Augmentation: time jitter, noise injection, frequency masking
* Pros: learns subtle morphology automatically
* Target: >90% accuracy, <10% FPR

**Track C: Physics-Informed Chirp Mass Estimator**

* Input: Same spectrogram
* Output: Estimated chirp mass $M_c$
* Physics constraint in loss function:
  $$\mathcal{L}_{physics} = \left| f_{peak} - k \cdot M_c^{-5/8} \right|^2$$
* Use $M_c$ estimate to validate/inform classification
* Novelty: First physics-informed rapid CBC classifier

**Combined Pipeline:**
```
Spectrogram → CNN → P(BBH), P(NS-present)
            ↘ PINN → M_c estimate → Physics validation
```

---

### Stage 6 — Evaluation (What you report)

**Phase 1 Evaluation (Real events only):**

* 5-fold cross-validation on O1–O3 real events
* Report with confidence intervals (given small N)
* Feature importance analysis

**Phase 2 Evaluation (Synthetic + Real):**

* Train on synthetic (10K+), validate on O1-O3 real
* Final test on O4a (held out). ([GW Open Science Center][1])

**Metrics:**

| Metric | Target | Why |
|--------|--------|-----|
| Accuracy | >90% | Match reference project |
| False Positive Rate | <10% | Operational reliability |
| NS-present Recall | >85% | Multi-messenger critical |
| Inference Latency | <1 sec | "Rapid" claim validation |
| AUC-ROC | >0.95 | Class separation quality |

**Domain Shift Analysis:**

* Performance drop from synthetic→real
* Performance drop from O3→O4a
* Analysis of failure modes

**Comparison Baselines:**

* Random baseline (50%)
* Phase 1 RF vs Phase 2 CNN
* With vs without physics-informed loss

---

### Stage 7 — Deliverables (What you physically produce)

**Data Artifacts:**
1. Real event manifest (CSV/JSON) with labels
2. Synthetic injection manifest with parameters
3. Feature dataset (CSV/Parquet)
4. Spectrogram image dataset

**Code Artifacts:**
5. Data download pipeline (GWpy)
6. Synthetic injection pipeline (PyCBC)
7. Feature extraction module
8. CNN training notebook (Colab-compatible)
9. Physics-informed loss implementation
10. Inference script with latency benchmark

**Documentation:**
11. Paper-style write-up: background, methods, results, limitations
12. Figures: example spectrograms, feature distributions, confusion matrices, ROC curves
13. Comparison table: Phase 1 vs Phase 2 performance
14. Domain shift analysis visualization

---

## 5) Compute / Storage Plan (for “huge data”)

### If you do event-centered segments (recommended)

* Data size: typically manageable (GBs to tens of GBs depending on windows + detectors + sample rate)

### If you do bulk O1–O4a continuous strain (true “huge”)

* You’ll likely be in **hundreds of GB to TB-scale** depending on sample rate and how much you mirror locally.
* You’ll need:

  * strict chunking strategy
  * file manifest
  * caching
  * careful IO and parallel processing

GWOSC offers 4 kHz and 16 kHz options for runs like O3 and O4a. ([GW Open Science Center][1])

---

# Architecture Decision Records (ADRs)

All architectural decisions are documented in comprehensive ADR files in the `docs/ADR/` directory.

## ADR Index

| ID | Title | Status | Category |
|----|-------|--------|----------|
| [ADR-001](docs/ADR/ADR-001-data-source-gwosc.md) | Data Source: GWOSC | Accepted | Data |
| [ADR-002](docs/ADR/ADR-002-train-test-split-by-run.md) | Train/Test Split by Observing Run | Accepted | Data |
| [ADR-003](docs/ADR/ADR-003-event-centered-segments.md) | Event-Centered Data Segments | Accepted | Data |
| [ADR-004](docs/ADR/ADR-004-detector-selection.md) | Detector Selection Strategy | Accepted | Data |
| [ADR-005](docs/ADR/ADR-005-sample-rate.md) | Sample Rate Selection | Accepted | Data |
| [ADR-006](docs/ADR/ADR-006-spectrogram-features.md) | Spectrogram + Features as Primary Input | Accepted | Features |
| [ADR-007](docs/ADR/ADR-007-gwpy-data-access.md) | GWpy for Data Access | Accepted | Infrastructure |
| [ADR-008](docs/ADR/ADR-008-ns-present-recall-priority.md) | NS-Present Recall Priority | Accepted | Evaluation |
| [ADR-009](docs/ADR/ADR-009-reproducibility.md) | Reproducibility Requirements | Accepted | Infrastructure |
| [ADR-010](docs/ADR/ADR-010-binary-classification.md) | Binary Classification (BBH vs NS-present) | Accepted | Model |
| [ADR-011](docs/ADR/ADR-011-synthetic-data-pycbc.md) | Synthetic Data via PyCBC | Accepted | Data |
| [ADR-012](docs/ADR/ADR-012-two-phase-strategy.md) | Two-Phase Implementation Strategy | Accepted | Architecture |
| [ADR-013](docs/ADR/ADR-013-physics-informed-loss.md) | Physics-Informed Loss Function | Accepted | Model |
| [ADR-014](docs/ADR/ADR-014-data-augmentation.md) | Light Data Augmentation | Accepted | Data |
| [ADR-015](docs/ADR/ADR-015-per-event-normalization.md) | Per-Event Feature Normalization | Accepted | Features |
| [ADR-016](docs/ADR/ADR-016-confident-events-only.md) | Confident Events Only | Accepted | Data |
| [ADR-017](docs/ADR/ADR-017-colab-a100-training.md) | Google Colab A100 for Training | Accepted | Infrastructure |
| [ADR-018](docs/ADR/ADR-018-ridge-extraction.md) | Ridge Extraction via Peak Tracking | Accepted | Features |

## ADR Summary

Each ADR follows a comprehensive format (see `docs/ADR/_template.md`) including:
- **Metadata**: ID, status, dates, authors, category
- **Context**: Background and current situation
- **Problem Statement**: What decision needs to be made
- **Decision Drivers**: Key factors influencing the decision
- **Considered Options**: Alternatives evaluated with detailed pros/cons
- **Decision Outcome**: Chosen option with detailed rationale
- **Consequences**: Positive, negative, and neutral impacts
- **Validation**: Success criteria and review triggers
- **Implementation Notes**: Practical guidance and code examples
- **References**: Supporting documentation and links

For the complete ADR documentation, see [docs/ADR/README.md](docs/ADR/README.md).

---

# Project Timeline

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | Setup | Environment setup, GWpy install, download real events, build manifest |
| 2 | Phase 1 | PyCBC setup, begin synthetic injection pipeline |
| 3 | Phase 1 | Spectrogram generation, feature extraction implementation |
| 4 | Phase 1 | Train Random Forest baseline, cross-validation, feature importance |
| 5 | Phase 2 | Generate 10K+ synthetic samples, prepare image dataset |
| 6 | Phase 2 | CNN training on Colab, hyperparameter tuning |
| 7 | Phase 2 | Physics-informed loss implementation, combined evaluation |
| 8 | Eval | O4a testing, domain shift analysis, latency benchmarks |
| 9 | Write-up | Figures, paper draft, polish |
| 10 | Buffer | Revisions, presentation prep |

---

# Tech Stack

| Component | Tool | Phase |
|-----------|------|-------|
| Language | Python 3.10+ | Both |
| Data fetch | GWpy | Both |
| Synthetic waveforms | PyCBC | Phase 2 |
| Spectrograms | scipy.signal / gwpy.spectrogram | Both |
| Features | numpy, scipy | Phase 1 |
| ML (baseline) | scikit-learn, XGBoost | Phase 1 |
| Deep learning | PyTorch 2.0+ | Phase 2 |
| Physics-informed loss | Custom PyTorch | Phase 2 |
| Visualization | matplotlib, seaborn | Both |
| Notebooks | Jupyter / Google Colab | Both |
| GPU | **Google Colab A100** (40-80GB VRAM) | Phase 2 |
| Mixed Precision | torch.cuda.amp | Phase 2 |
| Logging | TensorBoard / Weights & Biases | Phase 2 |

---

# A100 GPU Capabilities

| Capability | Configuration | Benefit |
|------------|---------------|---------|
| Batch Size | 128-256 (vs 32 on consumer GPU) | Faster convergence, stable gradients |
| Model Size | ResNet50/101 (vs ResNet18) | Higher capacity, better features |
| Mixed Precision | `torch.cuda.amp` with bfloat16 | 2x speedup, same accuracy |
| Data Loading | 8+ workers, pin_memory=True | No GPU idle time |
| Full Dataset | 10K+ spectrograms in VRAM | No disk I/O bottleneck |

---

# Folder Structure

```
project/
├── Project_dossier.md              # Planning document (this file)
├── README.md                       # Project overview + setup instructions
├── requirements.txt                # Python dependencies
├── setup.py                        # Make project installable (optional)
│
├── configs/
│   ├── config.yaml                 # Main config (paths, params, seeds)
│   ├── injection_params.yaml       # PyCBC waveform parameters
│   └── model_params.yaml           # Model hyperparameters
│
├── data/
│   ├── raw/                        # Downloaded strain files (.gwf, .hdf5)
│   │   ├── O1/
│   │   ├── O2/
│   │   ├── O3/
│   │   └── O4a/
│   ├── noise/                      # Background noise segments for injection
│   ├── synthetic/                  # PyCBC-generated injections
│   │   ├── bbh/
│   │   └── ns_present/
│   ├── spectrograms/
│   │   ├── images/                 # PNG/NPY for CNN (224x224)
│   │   └── matrices/               # Full resolution for features
│   └── features/                   # Extracted feature CSVs
│
├── manifests/
│   ├── event_manifest.csv          # Real events (name, gps, class, run)
│   ├── synthetic_manifest.csv      # Synthetic events (params, file path)
│   └── splits.json                 # Train/val/test splits
│
├── notebooks/                      # Jupyter notebooks (exploration + Colab)
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_download.ipynb
│   ├── 03_synthetic_injection.ipynb
│   ├── 04_spectrogram_generation.ipynb
│   ├── 05_feature_extraction.ipynb
│   ├── 06_phase1_baseline.ipynb
│   ├── 07_phase2_cnn.ipynb         # Colab A100 compatible
│   ├── 08_physics_informed.ipynb   # Colab A100 compatible
│   └── 09_evaluation.ipynb
│
├── src/                            # Reusable Python modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py              # GWpy data download
│   │   ├── injection.py            # PyCBC synthetic generation
│   │   ├── preprocessing.py        # Bandpass, whitening, etc.
│   │   └── dataset.py              # PyTorch Dataset classes
│   ├── features/
│   │   ├── __init__.py
│   │   ├── spectrogram.py          # Spectrogram generation
│   │   ├── ridge.py                # Chirp ridge extraction
│   │   └── extractor.py            # Feature vector extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py             # Random Forest / XGBoost
│   │   ├── cnn.py                  # ResNet / custom CNN
│   │   └── physics_loss.py         # Physics-informed loss
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop (A100 optimized)
│   │   └── callbacks.py            # Early stopping, checkpoints
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # Accuracy, F1, AUC, etc.
│       └── visualization.py        # Confusion matrix, ROC plots
│
├── scripts/                        # CLI scripts for automation
│   ├── download_events.py
│   ├── generate_synthetic.py
│   ├── train_baseline.py
│   ├── train_cnn.py
│   └── evaluate.py
│
├── outputs/
│   ├── models/                     # Saved model weights
│   │   ├── phase1/
│   │   └── phase2/
│   ├── figures/                    # Generated plots
│   ├── results/                    # Metrics, predictions CSVs
│   └── logs/                       # Training logs, TensorBoard
│
└── tests/                          # Unit tests (optional)
    ├── test_preprocessing.py
    └── test_features.py
```

---
