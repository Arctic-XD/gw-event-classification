# ADR-002 — Train/Test Split by Observing Run

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-002 |
| **Title** | Train/Test Split by Observing Run (O1-O3 Train, O4a Test) |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Machine learning models require separation of data into training and test sets to evaluate generalization performance. In gravitational-wave (GW) astronomy, data comes from distinct "observing runs" (O1, O2, O3, O4, etc.) during which the LIGO/Virgo/KAGRA detectors operate with specific configurations and sensitivity levels.

Each observing run has unique characteristics:
- **O1 (Sept 2015 - Jan 2016)**: First observing run, lowest sensitivity, 3 detections
- **O2 (Nov 2016 - Aug 2017)**: Improved sensitivity, Virgo joined, 8 detections
- **O3a (Apr 2019 - Oct 2019)**: Significantly improved sensitivity, 39 detections
- **O3b (Nov 2019 - Mar 2020)**: Continued O3 improvements, 35 detections
- **O4a (May 2023 - Jan 2024)**: Latest public data, further improvements, ~80+ candidates

Between observing runs, detectors undergo upgrades that change:
- Noise power spectral density (PSD)
- Frequency-dependent sensitivity curves
- Glitch populations and characteristics
- Calibration procedures

This creates a natural "domain shift" between runs—a model trained on O1-O3 data will encounter different noise characteristics when applied to O4 data. Testing on a different observing run is therefore a rigorous test of model generalization.

---

## 2. Problem Statement

We need to establish a principled train/test split strategy that:
1. Maximizes training data for model learning
2. Provides a rigorous test of generalization to new data
3. Simulates real-world deployment conditions (model trained on past, applied to future)
4. Uses only publicly available data
5. Avoids data leakage between training and testing

**Key Question**: How should we partition gravitational-wave events across training and test sets to ensure valid evaluation of model performance?

---

## 3. Decision Drivers

1. **Domain Shift Evaluation**: A key scientific question is whether spectrogram-based features generalize across detector upgrades. The split must enable testing this.

2. **Temporal Realism**: In real deployment, a model would be trained on past observations and applied to future events. The split should reflect this.

3. **Test Set Independence**: The test set must be completely independent of training to avoid optimistic bias in performance estimates.

4. **Statistical Power**: Both training and test sets need sufficient events for meaningful conclusions, given the limited total number of detections (~90 in O1-O3, ~80 in O4a).

5. **Data Availability**: Only GWOSC public releases can be used, which currently includes O1-O3 and O4a.

6. **Reproducibility**: The split must be clearly defined and reproducible by others.

7. **Scientific Relevance**: The test demonstrates "future readiness"—can a model work on the next generation of data?

---

## 4. Considered Options

### Option A: Chronological Split by Observing Run (O1-O3 Train / O4a Test)

**Description**: Use all events from O1, O2, O3a, and O3b for training and validation. Reserve all O4a events exclusively for final testing.

**Pros**:
- Strong domain shift test (different detector configurations)
- Mimics real-world deployment (past→future)
- Clear, unambiguous split definition
- Maximum training data from historical runs
- O4a is genuinely "unseen" during development
- Tests generalization to improved detector sensitivity

**Cons**:
- Cannot use O4a data for any hyperparameter tuning
- O4a class distribution may differ from O1-O3
- If O4a performance is poor, limited ability to diagnose
- O4a event count depends on what GWOSC releases

### Option B: Random Split Across All Runs

**Description**: Pool all events from O1-O4a and randomly split into 80% train / 20% test, stratified by class.

**Pros**:
- Maximizes data utilization
- Ensures similar class distributions in train/test
- More events in test set for statistical power

**Cons**:
- Data leakage: Test includes events from same runs as training
- Does not test generalization to new detector conditions
- Does not reflect real deployment scenario
- Overly optimistic performance estimates
- Events from same run may share correlated noise

### Option C: Leave-One-Run-Out Cross-Validation

**Description**: Train on N-1 runs, test on 1 run, rotate through all runs. Report average performance.

**Pros**:
- Tests generalization to each run
- Uses all data for both training and testing
- Provides uncertainty estimates

**Cons**:
- Computationally expensive (4+ model trainings)
- Different runs have very different event counts (O1: 3, O3a: 39)
- Doesn't provide single "final" model
- Complicates comparison with other methods

### Option D: O3 Train / O4a Test (Discard O1-O2)

**Description**: Train only on O3 data (74 events), test on O4a.

**Pros**:
- O3 is closest to O4 in detector configuration
- Reduces domain shift between train and test
- Simpler noise characteristics

**Cons**:
- Discards valuable training data (11 events from O1-O2)
- May not learn diverse enough representations
- Includes GW170817 (only confident BNS) in O2—critical for NS class

---

## 5. Decision Outcome

**Chosen Option**: Option A — Chronological Split by Observing Run (O1-O3 Train / O4a Test)

**Rationale**:

This decision directly addresses our core scientific question: *"Can a model trained on O1-O3 retain performance on O4a despite changes in detector sensitivity and noise characteristics?"*

Key reasons for this choice:

1. **Domain Shift as Feature, Not Bug**: The whole point of this project is rapid classification that works on *future* events. Testing on O4a—which has genuinely different noise characteristics—validates this capability. A model that only works on data similar to its training set is not useful for real deployment.

2. **Temporal Validity**: This split respects time. In production, we would train on all available past data and deploy on new observations. This split exactly mimics that scenario.

3. **No Data Leakage**: O4a events are completely independent of O1-O3. There's no possibility of the model "memorizing" test events or learning run-specific noise patterns that appear in both sets.

4. **Clear Narrative**: "Trained on O1-O3, tested on O4a" is immediately understandable and compelling for science fair judges or paper reviewers.

5. **Maximum Training Data**: We use all ~90 O1-O3 events for training/validation rather than holding some back randomly.

While Option B might give higher reported accuracy (by avoiding domain shift), it would be misleading about real-world performance. We prefer honest evaluation over flattering metrics.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Rigorous generalization test**: Performance on O4a demonstrates true generalization, not memorization
- **Real-world relevance**: Results directly predict how well the model would work on future events
- **Clear scientific narrative**: "Domain shift" analysis becomes a key contribution of the project
- **Benchmark quality**: This split could be used by others for comparison
- **Intellectual honesty**: We report performance in the hardest, most realistic scenario
- **Interesting failure modes**: If O4a performance differs from O1-O3, we can analyze why (detector changes, population shifts)

### 6.2 Negative Consequences

- **Expected performance drop**: O4a accuracy will likely be lower than O1-O3 cross-validation (this is informative, not a problem)
- **No O4a hyperparameter tuning**: Cannot use O4a to select model hyperparameters or architecture
- **Test set size uncertainty**: O4a event count depends on GWOSC release schedule
- **Class distribution shift**: O4a may have different BBH/NS-present ratio than O1-O3
- **One-shot evaluation**: Can only evaluate on O4a once to avoid implicit overfitting

### 6.3 Neutral Consequences

- Must clearly document that "O4" means "O4a public release" in all communications
- Cross-validation within O1-O3 is still needed for hyperparameter selection
- May need to address reviewers who expect random splits

---

## 7. Validation

**Success Criteria**:
- Model achieves >85% accuracy on O1-O3 cross-validation
- Model achieves >75% accuracy on O4a (with domain shift)
- Performance drop O3→O4a is <15 percentage points
- Failure cases on O4a can be explained (e.g., low SNR, edge cases)

**Review Date**: Week 8 (after O4a evaluation)

**Reversal Trigger**:
- O4a public release is delayed beyond project timeline
- O4a contains <20 events (insufficient statistical power)
- O4a classifications are unreliable or disputed

---

## 8. Implementation Notes

### Training Data (O1-O3)
| Run | Events | Period | Usage |
|-----|--------|--------|-------|
| O1 | ~3 | 2015-2016 | Train/Val |
| O2 | ~8 | 2016-2017 | Train/Val |
| O3a | ~39 | 2019 | Train/Val |
| O3b | ~35 | 2019-2020 | Train/Val |
| **Total** | ~85-90 | - | 5-fold CV |

### Test Data (O4a)
| Run | Events | Period | Usage |
|-----|--------|--------|-------|
| O4a | ~80+ | 2023-2024 | Final Test |

### Cross-Validation Strategy for O1-O3
- Use stratified 5-fold cross-validation within O1-O3 for model selection
- Ensure each fold has proportional representation of BBH and NS-present classes
- Report mean ± std across folds for all metrics

### O4a Evaluation Protocol
1. Train final model on entire O1-O3 dataset
2. Evaluate on O4a exactly once (no iterative tuning)
3. Report all metrics with confidence intervals
4. Document any O4a events that fail and analyze causes

### Manifest Structure
```csv
event_name,gps_time,run,split,class_label
GW150914,1126259462.4,O1,train,BBH
GW170817,1187008882.4,O2,train,BNS
...
GW230529_181500,1369504518.0,O4a,test,BBH
```

---

## 9. References

- [GWOSC O4 Data Release](https://gwosc.org/O4/): O4a data availability timeline
- [GWTC-3 Paper](https://arxiv.org/abs/2111.03606): O1-O3 event catalog
- [Domain Adaptation in Deep Learning](https://arxiv.org/abs/2004.10826): Theory of train/test domain shift
- [LIGO Detector Upgrades](https://www.ligo.caltech.edu/page/ligo-technology): Documentation of inter-run improvements
- [Temporal Validation in ML](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/): Best practices for time-series splits

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
