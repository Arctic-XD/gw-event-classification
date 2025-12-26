# ADR-010 — Binary Classification (BBH vs NS-present)

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-010 |
| **Title** | Binary Classification: BBH vs NS-present |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Compact binary coalescence (CBC) events detected by LIGO/Virgo fall into three physical categories:

1. **Binary Black Hole (BBH)**: Two black holes merging
   - O1-O3 count: ~85 confident events
   - Characteristic: High masses (10-150 M☉), short duration, lower frequencies

2. **Binary Neutron Star (BNS)**: Two neutron stars merging
   - O1-O3 count: 2 confident events (GW170817, GW190425)
   - Characteristic: Low masses (1-2.5 M☉), long duration, higher frequencies
   - Multi-messenger relevance: Produces EM counterparts (kilonova)

3. **Neutron Star-Black Hole (NSBH)**: One neutron star, one black hole
   - O1-O3 count: 2-4 confident events (GW190814 disputed, GW200105, GW200115)
   - Characteristic: Intermediate properties
   - Multi-messenger relevance: May produce EM counterparts

The severe class imbalance (85 BBH : 2 BNS : 3 NSBH) presents a fundamental challenge for machine learning. A 3-class classifier would have only 2 examples per class for BNS, making statistical inference unreliable.

---

## 2. Problem Statement

We must decide how to frame the classification problem given extreme class imbalance:

1. **3-class (BBH/BNS/NSBH)**: Matches physical categories but has statistically fragile minority classes
2. **2-class (BBH/NS-present)**: Combines BNS+NSBH into single class, improving statistics
3. **Multi-label or other formulations**: Alternative framings

**Key Question**: How should we formulate the classification task to balance physical meaningfulness with statistical validity?

---

## 3. Decision Drivers

1. **Statistical Validity**: Need sufficient samples per class for reliable training and evaluation. Rule of thumb: ≥10 samples per class minimum, preferably ≥30.

2. **Physical Meaningfulness**: Classification should align with astrophysically relevant categories.

3. **Operational Utility**: For multi-messenger astronomy, the key question is "does this event involve a neutron star?" (triggers EM follow-up).

4. **Evaluation Reliability**: Performance metrics need enough test samples to be meaningful.

5. **Future Extensibility**: 3-class becomes viable as more NS events are detected.

6. **Synthetic Data Role**: Synthetic injection can address imbalance, but 3-class still complicates training.

---

## 4. Considered Options

### Option A: 2-Class (BBH vs NS-present)

**Description**: Collapse BNS and NSBH into single "NS-present" class. Binary classification.

**Class Distribution**:
- BBH: ~85 events (94%)
- NS-present: ~5-6 events (6%)

**Pros**:
- Statistically more robust (~6 NS-present vs 2 BNS + 3 NSBH separately)
- Directly answers operational question ("any neutron star?")
- Cleaner decision boundary
- Easier evaluation with limited data
- Aligns with LVK "HasNS" alert parameter

**Cons**:
- Loses distinction between BNS and NSBH
- Less physically granular
- Cannot prioritize BNS (both NS produce kilonovae differently)

### Option B: 3-Class (BBH/BNS/NSBH)

**Description**: Full physical classification into three source types.

**Class Distribution**:
- BBH: ~85 events (94%)
- BNS: 2 events (2%)
- NSBH: 3-4 events (4%)

**Pros**:
- Matches physical reality
- Maximum information in output
- Can distinguish EM emission expectations

**Cons**:
- Only 2 BNS examples—impossible to train/validate reliably
- Confidence intervals on BNS metrics would be huge
- High risk of overfitting to specific events
- Statistically unsound

### Option C: Hierarchical Classification

**Description**: First classify BBH vs NS-present, then sub-classify NS-present into BNS vs NSBH.

**Pros**:
- Gets both coarse and fine-grained classification
- Second stage only runs on small subset

**Cons**:
- Second stage has only 5-6 training examples
- Error propagation from first stage
- Overcomplicated for available data

### Option D: Multi-Label (HasNS, HasBH)

**Description**: Predict two binary labels: "contains neutron star" and "contains black hole."

**Mapping**:
- BBH: HasNS=0, HasBH=1
- BNS: HasNS=1, HasBH=0
- NSBH: HasNS=1, HasBH=1

**Pros**:
- Elegant formulation
- Directly answers component questions

**Cons**:
- HasBH is nearly always 1 (only BNS has HasBH=0)
- Effectively same as NS-present for NSBH detection
- Added complexity without benefit

---

## 5. Decision Outcome

**Chosen Option**: Option A — 2-Class (BBH vs NS-present)

**Rationale**:

With only 2 BNS and 3-4 NSBH events in O1-O3, 3-class classification is statistically unsound:

1. **Minimum Sample Requirements**:
   - Standard ML guidance: ≥10-30 samples per class for reliable metrics
   - 2 BNS samples: Cannot estimate precision, recall, or confidence intervals meaningfully
   - Leave-one-out CV on 2 samples is not validation—it's memorization testing

2. **Operational Alignment**:
   - The key decision for multi-messenger astronomy is: "Should we trigger EM follow-up?"
   - This depends on NS *presence*, not whether it's BNS or NSBH
   - Both BNS and NSBH can produce kilonovae (NSBH if disruption occurs outside horizon)
   - LVK public alerts already use "HasNS" probability

3. **Statistical Improvement**:
   - Combining 2 BNS + 4 NSBH → 6 NS-present events
   - Still severely imbalanced (85:6) but now estimable
   - Cross-validation can produce meaningful bounds

4. **Future Work Note**:
   - O4 is detecting more NS events
   - When O4+O5 provide 20+ NS events total, 3-class becomes viable
   - Document this as explicit future work

**Physics Justification**:
"NS-present" is physically meaningful: these events involve matter (neutron star), unlike BBH which are pure spacetime. The presence of matter enables:
- Tidal disruption signatures
- Post-merger remnant physics  
- Electromagnetic counterparts
- Nucleosynthesis

---

## 6. Consequences

### 6.1 Positive Consequences

- **Statistical validity**: Can report meaningful metrics for both classes
- **Operational relevance**: Directly answers "trigger EM follow-up?" question
- **Cleaner training**: Binary classification is simpler and more robust
- **Evaluation reliability**: 6 NS-present events enable cross-validation
- **Clear narrative**: "Distinguish NS-containing events from pure BBH"

### 6.2 Negative Consequences

- **Information loss**: Cannot distinguish BNS from NSBH
- **Physical granularity**: Less detailed than astrophysical source types
- **Future limitation**: Must revisit when more data available

### 6.3 Neutral Consequences

- Synthetic data still generates BNS and NSBH separately (just labeled NS-present)
- Feature analysis can still examine BNS vs NSBH differences qualitatively
- Model could output 3-class probabilities internally if needed

---

## 7. Validation

**Success Criteria**:
- NS-present class has ≥5 events in training (achieved with real data)
- NS-present class has ≥10 events after synthetic augmentation
- Cross-validation produces bounded (not degenerate) performance estimates
- Model performs above random (50%) on both classes

**Review Date**: Week 4 (Phase 1 evaluation)

**Reversal Trigger**:
- O4a release contains 10+ NS events (would enable 3-class)
- Domain expert strongly recommends 3-class despite statistics
- Synthetic data generation successfully creates statistically equivalent 3-class dataset

---

## 8. Implementation Notes

### Label Mapping

```python
LABEL_MAP_3CLASS = {
    'BBH': 0,
    'BNS': 1,
    'NSBH': 2
}

LABEL_MAP_2CLASS = {
    'BBH': 0,
    'BNS': 1,      # → NS-present
    'NSBH': 1      # → NS-present
}

def map_to_binary(label_3class):
    """Convert 3-class label to binary NS-present."""
    if label_3class in ['BNS', 'NSBH']:
        return 'NS-present'
    return 'BBH'
```

### Class Descriptions

| Class | Physical Meaning | Examples | Spectrogram Signature |
|-------|------------------|----------|----------------------|
| BBH | Two black holes | GW150914, GW190521 | Short chirp, low freq, <1s |
| NS-present | At least one NS | GW170817, GW200115 | Longer chirp, higher freq, >1s |

### Handling Synthetic Data

```python
def generate_synthetic_label(event_type):
    """
    Synthetic events are generated as BBH, BNS, or NSBH
    but labeled as binary for training.
    """
    # Store original type for analysis
    original_type = event_type
    
    # Map to binary
    if event_type in ['BNS', 'NSBH']:
        binary_label = 'NS-present'
    else:
        binary_label = 'BBH'
    
    return binary_label, original_type
```

### Future 3-Class Extension

```python
# When sufficient NS events available:
if len(bns_events) >= 15 and len(nsbh_events) >= 15:
    use_3class = True
    label_map = LABEL_MAP_3CLASS
else:
    use_3class = False
    label_map = LABEL_MAP_2CLASS
```

### Documentation Template

```markdown
## Classification Problem

**Formulation**: Binary classification
- **Class 0 (BBH)**: Binary black hole mergers (N=85)
- **Class 1 (NS-present)**: Events involving ≥1 neutron star (N=6)
  - Includes: BNS (2), NSBH (4)

**Rationale**: 3-class classification was rejected due to insufficient 
BNS samples (N=2) for statistically valid training and evaluation.

**Future Work**: Extend to 3-class when O4+O5 provide ≥15 events per NS subclass.
```

---

## 9. References

- [GWTC-3 Classifications](https://arxiv.org/abs/2111.03606): Source type assignments for O1-O3
- [Class Imbalance in ML](https://www.sciencedirect.com/science/article/pii/S0957417417302424): Statistical challenges
- [LVK Alert Parameters](https://emfollow.docs.ligo.org/): HasNS probability definition
- [Small Sample Learning](https://arxiv.org/abs/1904.05046): Minimum samples for reliable ML
- [BNS Kilonova](https://arxiv.org/abs/1710.05833): GW170817 multi-messenger observations

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
