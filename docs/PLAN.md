# Project Plan: GW Event Classification

## Overview

**Goal:** Classify gravitational-wave events (BBH vs NS-present) using spectrogram features + ML/DL  
**Duration:** 10 weeks  
**Phases:** Setup → Phase 1 (Baseline) → Phase 2 (Deep Learning) → Evaluation → Write-up

---

# Milestones

| ID | Milestone | Target Week | Description |
|----|-----------|-------------|-------------|
| M1 | Environment & Data Ready | Week 1 | Dev environment, real events downloaded, manifest built |
| M2 | Phase 1 Complete | Week 4 | Feature extraction + RF baseline trained and validated |
| M3 | Synthetic Pipeline Ready | Week 5 | 10K+ synthetic samples generated via PyCBC |
| M4 | Phase 2 Complete | Week 7 | CNN + physics-informed model trained on Colab A100 |
| M5 | Final Evaluation | Week 8 | O4a test, domain shift analysis, latency benchmarks |
| M6 | Paper Ready | Week 10 | Figures, write-up, presentation materials |

---

# Epics

## Epic 1: Project Setup & Infrastructure
**Milestone:** M1  
**Goal:** Development environment ready, data pipeline functional

| Issue | Title | Description |
|-------|-------|-------------|
| E1-1 | Environment setup | Create conda env, install GWpy/PyCBC/PyTorch, verify imports |
| E1-2 | Build event manifest | Scrape GWTC for O1-O3 events, create CSV with labels |
| E1-3 | Download real events | Fetch ±32s strain windows for ~90 events via GWpy |
| E1-4 | Data validation | Verify downloads, check for gaps, document missing events |
| E1-5 | Config system | Finalize config.yaml, injection_params.yaml, model_params.yaml |

---

## Epic 2: Preprocessing Pipeline
**Milestone:** M2  
**Goal:** Raw strain → clean, normalized spectrograms

| Issue | Title | Description |
|-------|-------|-------------|
| E2-1 | Preprocessing module | Implement bandpass (20-512Hz), whitening, detrend |
| E2-2 | Spectrogram generation | Q-transform or STFT, save matrices + images |
| E2-3 | Per-event normalization | Z-score normalization per spectrogram (ADR-015) |
| E2-4 | Preprocessing notebook | Interactive notebook for visual validation |
| E2-5 | Batch processing script | Process all events, save to data/spectrograms/ |

---

## Epic 3: Feature Extraction (Phase 1)
**Milestone:** M2  
**Goal:** Extract interpretable features from spectrograms

| Issue | Title | Description |
|-------|-------|-------------|
| E3-1 | Ridge extraction | Implement peak tracking for chirp ridge (ADR-018) |
| E3-2 | Chirp geometry features | Ridge slope, duration, time-to-peak |
| E3-3 | Frequency features | Peak freq, band power ratios, spectral centroid |
| E3-4 | Statistical features | Spectral entropy, kurtosis, skewness |
| E3-5 | Feature dataset | Generate features CSV for all events |
| E3-6 | Feature exploration | Notebook with distributions, correlations, class separation |

---

## Epic 4: Baseline Model (Phase 1)
**Milestone:** M2  
**Goal:** Train interpretable ML model on real events

| Issue | Title | Description |
|-------|-------|-------------|
| E4-1 | Train/val split | Stratified split preserving class balance |
| E4-2 | Random Forest baseline | Train RF with class weights, 5-fold CV |
| E4-3 | XGBoost comparison | Train XGBoost, compare to RF |
| E4-4 | Feature importance | SHAP values, permutation importance |
| E4-5 | Baseline evaluation | Accuracy, F1, AUC, confusion matrix |
| E4-6 | Phase 1 report | Document baseline results, identify best features |

---

## Epic 5: Synthetic Data Pipeline (Phase 2)
**Milestone:** M3  
**Goal:** Generate 10K+ synthetic injections via PyCBC

| Issue | Title | Description |
|-------|-------|-------------|
| E5-1 | PyCBC waveform setup | Implement BBH/BNS/NSBH waveform generation |
| E5-2 | Noise segment collection | Download background noise from O1-O3 |
| E5-3 | Injection pipeline | Inject waveforms into noise at various SNRs |
| E5-4 | Synthetic manifest | Track all injections with parameters |
| E5-5 | Generate BBH samples | 5,000 BBH injections |
| E5-6 | Generate NS-present samples | 5,000 BNS + NSBH injections |
| E5-7 | Synthetic spectrograms | Process all synthetic → spectrograms + images |
| E5-8 | Quality validation | Visual check, SNR distribution analysis |

---

## Epic 6: CNN Model (Phase 2)
**Milestone:** M4  
**Goal:** Train deep learning classifier on synthetic + real data

| Issue | Title | Description |
|-------|-------|-------------|
| E6-1 | PyTorch Dataset | Implement SpectrogramDataset with augmentation |
| E6-2 | CNN architecture | ResNet18/50 with custom head |
| E6-3 | Data augmentation | Time jitter, noise injection, freq masking (ADR-014) |
| E6-4 | Colab notebook | A100-optimized training notebook |
| E6-5 | Training loop | AMP, gradient accumulation, early stopping |
| E6-6 | Hyperparameter tuning | LR, batch size, weight decay sweep |
| E6-7 | Model checkpointing | Save best model by val loss |
| E6-8 | CNN evaluation | Test on real events, compare to baseline |

---

## Epic 7: Physics-Informed Model (Phase 2)
**Milestone:** M4  
**Goal:** Add physics constraints to improve classification

| Issue | Title | Description |
|-------|-------|-------------|
| E7-1 | Physics loss function | Implement chirp mass constraint loss (ADR-013) |
| E7-2 | Multi-task head | Classification + Mc regression |
| E7-3 | Combined loss | α·L_class + β·L_physics |
| E7-4 | PINN training | Train with physics-informed loss |
| E7-5 | Ablation study | With vs without physics loss comparison |
| E7-6 | Mc validation | Compare estimated Mc to catalog values |

---

## Epic 8: Final Evaluation
**Milestone:** M5  
**Goal:** Comprehensive evaluation on held-out O4a data

| Issue | Title | Description |
|-------|-------|-------------|
| E8-1 | O4a data download | Fetch O4a events (test set) |
| E8-2 | O4a preprocessing | Apply same pipeline to O4a |
| E8-3 | Final predictions | Run all models on O4a |
| E8-4 | Domain shift analysis | Compare O3→O4a performance drop |
| E8-5 | Synthetic-to-real analysis | Performance on real vs synthetic |
| E8-6 | Latency benchmark | Measure end-to-end inference time |
| E8-7 | Error analysis | Analyze misclassified events |
| E8-8 | Final metrics table | Compile all results |

---

## Epic 9: Documentation & Write-up
**Milestone:** M6  
**Goal:** Publication-ready documentation

| Issue | Title | Description |
|-------|-------|-------------|
| E9-1 | Example spectrograms | Generate figures for paper |
| E9-2 | Feature distribution plots | Violin/box plots by class |
| E9-3 | ROC/PR curves | Publication-quality evaluation plots |
| E9-4 | Confusion matrices | Phase 1 vs Phase 2 comparison |
| E9-5 | Architecture diagram | Visual pipeline overview |
| E9-6 | Methods section | Write data/methods for paper |
| E9-7 | Results section | Write results with tables/figures |
| E9-8 | Discussion/limitations | Honest assessment of limitations |
| E9-9 | README polish | Final project README with setup instructions |
| E9-10 | Presentation prep | Slides for science fair |

---

# Issue Summary

| Epic | Issues | Priority |
|------|--------|----------|
| E1: Setup | 5 | 🔴 Critical |
| E2: Preprocessing | 5 | 🔴 Critical |
| E3: Features | 6 | 🔴 Critical |
| E4: Baseline | 6 | 🟠 High |
| E5: Synthetic | 8 | 🟠 High |
| E6: CNN | 8 | 🟠 High |
| E7: Physics | 6 | 🟡 Medium |
| E8: Evaluation | 8 | 🔴 Critical |
| E9: Write-up | 10 | 🟠 High |
| **Total** | **62** | |

---

# Dependency Graph

```
E1 (Setup)
    ↓
E2 (Preprocessing) ←─────────────────┐
    ↓                                │
E3 (Features)                    E5 (Synthetic)
    ↓                                │
E4 (Baseline Phase 1)                ↓
    ↓                            E6 (CNN)
    │                                │
    │                            E7 (Physics)
    │                                │
    └────────────→ E8 (Evaluation) ←─┘
                        ↓
                   E9 (Write-up)
```

---

# Quick Reference: Next Actions

**Week 1 Focus:**
1. E1-1: Environment setup
2. E1-2: Build event manifest
3. E1-3: Download real events

**Blockers to Watch:**
- GWOSC API availability
- PyCBC installation (can be tricky on Windows)
- Colab GPU quota limits

---

# GitHub Labels (Suggested)

| Label | Color | Description |
|-------|-------|-------------|
| `epic:setup` | #0052CC | Project setup tasks |
| `epic:preprocessing` | #5319E7 | Data preprocessing |
| `epic:features` | #006B75 | Feature extraction |
| `epic:baseline` | #B60205 | Phase 1 baseline model |
| `epic:synthetic` | #FBCA04 | Synthetic data generation |
| `epic:cnn` | #D93F0B | CNN model |
| `epic:physics` | #0E8A16 | Physics-informed model |
| `epic:evaluation` | #1D76DB | Final evaluation |
| `epic:docs` | #C5DEF5 | Documentation |
| `priority:critical` | #B60205 | Must complete |
| `priority:high` | #D93F0B | Important |
| `priority:medium` | #FBCA04 | Nice to have |
| `phase:1` | #BFD4F2 | Phase 1 work |
| `phase:2` | #D4C5F9 | Phase 2 work |
