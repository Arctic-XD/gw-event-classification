# Architecture Decision Records (ADRs)

## Overview

This directory contains the Architecture Decision Records (ADRs) for the Gravitational-Wave Classification Project. ADRs document significant technical decisions made during the project, providing context, rationale, and consequences for future reference.

## ADR Format

Each ADR follows the standard format:

1. **Title & Metadata**: Unique identifier, date, status, authors
2. **Context**: Background information and current situation
3. **Problem Statement**: The specific problem being addressed
4. **Decision Drivers**: Key factors influencing the decision
5. **Considered Options**: Alternatives that were evaluated
6. **Decision Outcome**: The chosen option and rationale
7. **Consequences**: Positive, negative, and neutral impacts
8. **Validation**: How to verify the decision was correct
9. **References**: Supporting documentation and links

## Status Definitions

| Status | Description |
|--------|-------------|
| **Proposed** | Under discussion, not yet approved |
| **Accepted** | Approved and in effect |
| **Deprecated** | No longer applies, superseded |
| **Superseded** | Replaced by another ADR |

## ADR Index

| ID | Title | Status | Category |
|----|-------|--------|----------|
| [ADR-001](ADR-001-data-source-gwosc.md) | Data Source: GWOSC | Accepted | Data |
| [ADR-002](ADR-002-train-test-split-by-run.md) | Train/Test Split by Observing Run | Accepted | Data |
| [ADR-003](ADR-003-event-centered-segments.md) | Event-Centered Data Segments | Accepted | Data |
| [ADR-004](ADR-004-detector-selection.md) | Detector Selection Strategy | Accepted | Data |
| [ADR-005](ADR-005-sample-rate.md) | Sample Rate Selection | Accepted | Data |
| [ADR-006](ADR-006-spectrogram-features.md) | Spectrogram + Features as Primary Input | Accepted | Features |
| [ADR-007](ADR-007-gwpy-data-access.md) | GWpy for Data Access | Accepted | Infrastructure |
| [ADR-008](ADR-008-ns-present-recall-priority.md) | NS-Present Recall Priority | Accepted | Evaluation |
| [ADR-009](ADR-009-reproducibility.md) | Reproducibility Requirements | Accepted | Infrastructure |
| [ADR-010](ADR-010-binary-classification.md) | Binary Classification (BBH vs NS-present) | Accepted | Model |
| [ADR-011](ADR-011-synthetic-data-pycbc.md) | Synthetic Data via PyCBC | Accepted | Data |
| [ADR-012](ADR-012-two-phase-strategy.md) | Two-Phase Implementation Strategy | Accepted | Architecture |
| [ADR-013](ADR-013-physics-informed-loss.md) | Physics-Informed Loss Function | Accepted | Model |
| [ADR-014](ADR-014-data-augmentation.md) | Light Data Augmentation | Accepted | Data |
| [ADR-015](ADR-015-per-event-normalization.md) | Per-Event Feature Normalization | Accepted | Features |
| [ADR-016](ADR-016-confident-events-only.md) | Confident Events Only | Accepted | Data |
| [ADR-017](ADR-017-colab-a100-training.md) | Google Colab A100 for Training | Accepted | Infrastructure |
| [ADR-018](ADR-018-ridge-extraction.md) | Ridge Extraction via Peak Tracking | Accepted | Features |

## Category Legend

- **Data**: Decisions about data acquisition, storage, and processing
- **Features**: Decisions about feature engineering and representation
- **Model**: Decisions about model architecture and training
- **Evaluation**: Decisions about metrics and validation
- **Infrastructure**: Decisions about tools, environment, and workflow
- **Architecture**: High-level system design decisions

## Creating New ADRs

1. Copy `_template.md` to `ADR-XXX-short-title.md`
2. Fill in all sections completely
3. Submit for review before marking as "Accepted"
4. Update this README index

## References

- [Michael Nygard's ADR Article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
