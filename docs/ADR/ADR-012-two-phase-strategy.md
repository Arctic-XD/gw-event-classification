# ADR-012 — Two-Phase Implementation Strategy

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-012 |
| **Title** | Two-Phase Implementation Strategy |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

This project has ambitious goals spanning multiple methodologies:
- Traditional ML with hand-crafted features (interpretable)
- Deep learning with CNNs (high-performance)
- Physics-informed neural networks (novel)
- Synthetic data generation and transfer learning

Attempting to implement all components simultaneously risks:
- Scope creep without completing any component well
- No fallback if advanced methods fail
- Difficulty debugging complex interdependent systems
- No clear progress milestones

A phased approach allows incremental development with clear deliverables at each stage.

---

## 2. Problem Statement

How should we structure project development to:
1. Ensure deliverable results even if time runs short
2. Build complexity incrementally on validated foundations
3. Provide clear milestones for progress tracking
4. Enable graceful degradation if advanced methods fail

**Key Question**: What implementation strategy maximizes both project success probability and potential impact?

---

## 3. Decision Drivers

1. **Time Constraints**: 10-week science fair timeline

2. **Risk Management**: Need working results even if CNN/PINN fails

3. **Dependency Management**: CNN requires synthetic data; PINN requires CNN

4. **Validation**: Each component should be validated before building on it

5. **Presentation**: Clear story of "baseline → improvement"

6. **Scientific Rigor**: Comparison between methods is valuable

---

## 4. Considered Options

### Option A: Two-Phase (Interpretable Baseline → Deep Learning)

**Description**: 
- **Phase 1** (Weeks 1-4): Feature extraction + classical ML on real events
- **Phase 2** (Weeks 5-8): CNN + Physics-informed NN on synthetic + real

**Pros**:
- Guaranteed Phase 1 deliverable
- Clear progression narrative
- Validates features before adding complexity
- Enables direct comparison
- Natural checkpoints

**Cons**:
- May not reach Phase 2 if Phase 1 takes too long
- Some parallelization possible but not utilized

### Option B: Parallel Development

**Description**: Work on all components simultaneously from start.

**Pros**:
- Maximum parallelization
- No artificial sequencing

**Cons**:
- Complex debugging
- No clear checkpoints
- Risk of nothing working
- Hard to compare fairly

### Option C: CNN-First (Skip Classical ML)

**Description**: Go directly to CNN on synthetic data.

**Pros**:
- More "cutting-edge"
- Skip "boring" classical ML

**Cons**:
- No interpretable baseline
- Harder to debug
- No comparison benchmark
- Loses "quantitative features" project focus

---

## 5. Decision Outcome

**Chosen Option**: Option A — Two-Phase Implementation

**Phase 1: Interpretable Baseline (Weeks 1-4)**

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Environment setup, data download | Event manifest, raw data |
| 2 | Preprocessing, spectrogram generation | Spectrogram dataset |
| 3 | Feature extraction, validation | Feature CSV, feature analysis |
| 4 | Train RF/XGBoost, cross-validation | Baseline model, metrics |

**Phase 1 Exit Criteria**:
- ✓ All O1-O3 events downloaded and preprocessed
- ✓ Feature extraction pipeline validated
- ✓ Baseline model achieves >80% CV accuracy
- ✓ Feature importance analysis complete

**Phase 2: Deep Learning + Physics-Informed (Weeks 5-8)**

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 5 | Synthetic data generation (PyCBC) | 10K+ synthetic samples |
| 6 | CNN training on A100 | Trained CNN model |
| 7 | Physics-informed loss implementation | PINN model, comparison |
| 8 | O4a evaluation, domain shift analysis | Final metrics, paper |

**Phase 2 Exit Criteria**:
- ✓ 10,000+ synthetic samples generated
- ✓ CNN achieves >90% accuracy on synthetic
- ✓ CNN achieves >80% on real O1-O3
- ✓ Physics-informed model evaluated
- ✓ O4a test results documented

**Rationale**:

1. **Guaranteed Deliverable**: Phase 1 produces a complete, working classifier. Even if time runs out, project has valid results.

2. **Validation Before Extension**: Feature-based model validates that spectrogram features contain class information before investing in CNN.

3. **Comparison Value**: "CNN improves accuracy from 85% to 92%" is more compelling than "CNN achieves 92%."

4. **Risk Mitigation**: If synthetic data generation fails, Phase 1 results stand alone. If CNN fails, feature model is the result.

5. **Clear Narrative**: "We started simple, validated our approach, then demonstrated improvements with deep learning."

---

## 6. Consequences

### 6.1 Positive Consequences

- **Guaranteed results**: Phase 1 alone is a valid science fair project
- **Progressive complexity**: Each phase builds on validated foundation
- **Clear checkpoints**: Know if on track at week 4
- **Comparison story**: Can quantify improvement from Phase 1→2
- **Debugging support**: Simpler Phase 1 helps diagnose Phase 2 issues

### 6.2 Negative Consequences

- **Sequential dependencies**: Phase 2 cannot start until Phase 1 validates features
- **Time pressure**: Must complete Phase 1 by week 4
- **Potential waste**: If Phase 1 works perfectly, still must do Phase 2

### 6.3 Neutral Consequences

- Project documentation follows phase structure
- Results section has two major parts
- May have different "best model" from each phase

---

## 7. Validation

**Phase 1 Success Criteria** (Week 4 checkpoint):
- [ ] All O1-O3 events processed
- [ ] >10 features extracted per event
- [ ] RF/XGBoost CV accuracy >80%
- [ ] NS-present recall >75%
- [ ] Feature importance identifies physics-meaningful features

**Phase 2 Success Criteria** (Week 8 checkpoint):
- [ ] 10K+ synthetic samples generated
- [ ] CNN trains successfully on A100
- [ ] CNN outperforms Phase 1 baseline (or analysis explains why not)
- [ ] Physics-informed loss evaluated
- [ ] O4a results documented

**Reversal Trigger**:
- Phase 1 accuracy <65% (features insufficient—reconsider approach)
- Phase 1 takes >5 weeks (compress Phase 2)

---

## 8. Implementation Notes

### Phase Transition Checklist

```markdown
## Phase 1 → Phase 2 Transition Checklist

### Required for Phase 2 Start:
- [ ] event_manifest.csv complete and validated
- [ ] All spectrograms generated and saved
- [ ] Feature extraction pipeline tested
- [ ] Baseline model trained and evaluated
- [ ] Phase 1 results documented

### Phase 2 Setup:
- [ ] PyCBC installed and tested
- [ ] Noise segments downloaded
- [ ] Colab A100 access confirmed
- [ ] CNN architecture designed
```

### Risk Mitigation Strategies

| Risk | Mitigation |
|------|------------|
| Phase 1 delays | Parallel data download + feature design in Week 1 |
| Insufficient Phase 1 accuracy | Add features, tune hyperparameters, accept lower threshold |
| Synthetic generation issues | Start Week 5 early, have PyCBC tested in Week 4 |
| A100 unavailable | Fall back to smaller CNN on CPU/T4 |
| Phase 2 incomplete | Document partial results, emphasize Phase 1 |

### Progress Tracking Template

```markdown
# Weekly Progress Report

## Week X Summary

### Phase: [1/2]

### Completed This Week:
- [ ] Task 1
- [ ] Task 2

### Blockers:
- Issue 1: [description] → [resolution plan]

### Next Week Goals:
- Goal 1
- Goal 2

### Phase Checkpoint Status:
- Overall: [On Track / At Risk / Blocked]
```

---

## 9. References

- [Agile Development](https://agilemanifesto.org/): Iterative development principles
- [ML Project Lifecycle](https://ml-ops.org/): Best practices for ML projects
- [Risk Management](https://www.pmi.org/): Project management fundamentals
- [Incremental Development](https://en.wikipedia.org/wiki/Iterative_and_incremental_development): Software engineering approach

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
