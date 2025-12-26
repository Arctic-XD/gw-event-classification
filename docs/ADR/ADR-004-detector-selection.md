# ADR-004 — Detector Selection Strategy

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-004 |
| **Title** | Detector Selection: L1-First with Optional H1 Fusion |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

The global gravitational-wave detector network consists of multiple instruments:
- **LIGO Livingston (L1)**: Louisiana, USA — operational since O1
- **LIGO Hanford (H1)**: Washington, USA — operational since O1
- **Virgo (V1)**: Cascina, Italy — joined in O2, operational since
- **KAGRA (K1)**: Japan — joined in O3GK, limited sensitivity

Each detector independently measures gravitational-wave strain. For a given event:
- Not all detectors may have been operational
- Each detector has different noise characteristics
- Signal appears at different times due to light travel time (up to ~10ms between L1 and H1)
- Signal amplitude varies based on source sky location relative to detector antenna pattern

Multi-detector analysis provides benefits:
- Sky localization (triangulation)
- Signal confirmation (coherent detection)
- Improved SNR (coherent combination)
- Glitch rejection (coincidence requirement)

However, multi-detector analysis also adds complexity:
- Must handle missing detectors for some events
- Data fusion strategies needed
- Increased data volume and processing time

---

## 2. Problem Statement

We need to determine which detector(s) to use for spectrogram-based classification:
1. Which detector(s) should be the primary data source?
2. How should we handle events where the primary detector(s) didn't observe?
3. If using multiple detectors, how should we combine information?
4. What is the trade-off between simplicity and performance?

**Key Question**: Should we use a single detector, multiple detectors, or a flexible approach that adapts per-event?

---

## 3. Decision Drivers

1. **Data Availability**: Not all detectors observed all events. L1 has the highest duty cycle across O1-O4.

2. **Consistency**: Using the same detector for all events provides consistent noise characteristics and simplifies analysis.

3. **Complexity Management**: Multi-detector fusion requires choosing fusion strategy, handling missing data, and increases implementation effort.

4. **Performance Potential**: Multiple detectors could improve classification accuracy through redundancy.

5. **Science Fair Scope**: A 10-week project has limited time for extensive multi-detector infrastructure.

6. **O4a Compatibility**: O4a public data currently includes H1 and L1 (not V1 initially).

7. **Extensibility**: The chosen approach should allow future expansion to more detectors.

---

## 4. Considered Options

### Option A: Single Detector (L1 Only)

**Description**: Use only LIGO Livingston data for all events. Skip events where L1 didn't observe.

**Pros**:
- Maximum simplicity—no fusion logic needed
- Consistent noise characteristics across all samples
- L1 has excellent duty cycle (rarely missing events)
- Fastest implementation
- Cleaner experimental setup

**Cons**:
- Loses information from other detectors
- Must skip events where L1 was down
- May miss loud signals better seen in H1
- Not representative of operational multi-detector pipelines

### Option B: Single Detector (Best Available Per-Event)

**Description**: For each event, select the single detector with highest SNR or best data quality. Different events may use different detectors.

**Pros**:
- Uses best available data for each event
- Can include events where L1 was down
- Still no fusion complexity

**Cons**:
- Introduces detector-dependent variation in features
- Must implement detector selection logic
- Model must handle different noise characteristics
- Inconsistent data source complicates analysis

### Option C: Multi-Detector Feature Concatenation (Early Fusion)

**Description**: Compute features independently for each available detector, concatenate into a single feature vector.

**Pros**:
- Uses all available detector information
- Simple fusion strategy
- Model learns to weight detectors
- Handles missing detectors with zero-padding or masking

**Cons**:
- Feature dimensionality doubles or triples
- Must handle variable input sizes (missing detectors)
- Training complexity increases
- Overfitting risk with small dataset

### Option D: Multi-Detector Probability Fusion (Late Fusion)

**Description**: Train separate models per detector, combine output probabilities.

**Pros**:
- Each model sees consistent data
- Simple probability combination (average, max, voting)
- Gracefully handles missing detectors
- Interpretable per-detector contributions

**Cons**:
- Must train multiple models
- More computational cost
- Combination strategy affects results
- Still need to handle single-detector events

### Option E: L1-First with Optional H1 Upgrade Path

**Description**: Start with L1-only baseline. After establishing baseline performance, add H1 features as an optional upgrade. Document both single and multi-detector results.

**Pros**:
- Gets working system quickly (L1 only)
- Provides clear ablation (L1 vs L1+H1)
- Matches incremental development approach
- Can stop at L1 if time runs short
- Scientific value in comparing single vs multi

**Cons**:
- Requires implementing both approaches eventually
- H1 upgrade may not happen if time is limited
- Two experimental conditions to track

---

## 5. Decision Outcome

**Chosen Option**: Option E — L1-First with Optional H1 Upgrade Path

**Rationale**:

This decision follows the project's two-phase philosophy: get something working first, then enhance.

1. **L1 as Primary**: LIGO Livingston (L1) is chosen as the primary detector because:
   - L1 has consistently high duty cycle across O1-O4
   - L1 was operating for nearly all confident GWTC events
   - L1 data quality is generally excellent
   - Single-detector baseline provides clean experimental setup

2. **H1 as Optional Enhancement**: After Phase 1 baseline is complete:
   - Add H1 features using early fusion (concatenation)
   - Compare L1-only vs L1+H1 performance
   - Quantify multi-detector benefit (expected: 5-15% improvement)

3. **Why Not V1/K1**: 
   - V1 has lower duty cycle and different noise characteristics
   - K1 has very limited operational time
   - O4a public release focuses on H1/L1
   - Adding more detectors has diminishing returns vs complexity

4. **Graceful Degradation**: If an event has no L1 data:
   - Primary approach: Use H1 if available (with appropriate flag)
   - Fallback: Exclude event from dataset (rare cases)

This approach ensures we have a working classifier quickly while leaving room for improvement.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Rapid development**: L1-only baseline can be completed in Phase 1
- **Clean comparison**: L1 vs L1+H1 provides clear ablation study
- **Reduced complexity**: No multi-detector fusion needed for initial system
- **Consistent data**: All training samples have same noise characteristics
- **Extensible design**: Architecture supports adding detectors later
- **Publishable result**: Single-detector performance is a valid scientific result

### 6.2 Negative Consequences

- **Information loss**: Not using all available detector data initially
- **Missing events**: ~5-10% of events may have no L1 data (must use H1 or exclude)
- **Suboptimal performance**: Multi-detector would likely improve accuracy
- **Not operational**: Real LVK pipelines use all available detectors

### 6.3 Neutral Consequences

- L1 bias: Model learns L1-specific noise characteristics (may not transfer to H1-only events)
- H1 upgrade becomes Phase 2+ stretch goal
- Documentation must clearly state single-detector limitation

---

## 7. Validation

**Success Criteria**:
- >95% of GWTC events have L1 data available
- L1-only model achieves >85% accuracy on O1-O3 cross-validation
- If H1 fusion implemented, shows measurable improvement over L1-only

**Review Date**: End of Week 4 (Phase 1 completion)

**Reversal Trigger**:
- <80% of events have L1 data (would need multi-detector from start)
- L1-only performance is significantly worse than expected (<70% accuracy)
- Time permits full multi-detector implementation from start

---

## 8. Implementation Notes

### Detector Priority Order
1. **L1** (LIGO Livingston) — primary, always use if available
2. **H1** (LIGO Hanford) — secondary, use if L1 unavailable or for fusion
3. **V1** (Virgo) — tertiary, optional future enhancement
4. **K1** (KAGRA) — unlikely to use, limited data

### Data Download Logic
```python
def get_detector_for_event(event_name, available_detectors):
    """Select detector(s) for an event based on availability."""
    priority = ['L1', 'H1', 'V1', 'K1']
    
    for det in priority:
        if det in available_detectors:
            return det  # L1-first strategy
    
    raise ValueError(f"No data available for {event_name}")
```

### Manifest Tracking
```csv
event_name,available_detectors,selected_detector,has_L1,has_H1
GW150914,"L1,H1",L1,True,True
GW170817,"L1,H1,V1",L1,True,True
```

### H1 Fusion Strategy (If Implemented)

**Early Fusion (Recommended)**:
- Compute features for L1 and H1 separately
- Concatenate: `features = [L1_features, H1_features]`
- For missing H1: zero-pad or use separate model

**Late Fusion (Alternative)**:
- Train two models: `model_L1`, `model_H1`
- Combine probabilities: `P = 0.5 * P_L1 + 0.5 * P_H1`
- For missing detector: use available model only

### Data Quality Considerations
- Use GWOSC data quality flags to identify clean segments
- Some events have "data quality vetoes" for specific detectors
- Prefer detector with better DQ if both available

---

## 9. References

- [GWOSC Detector Status](https://gwosc.org/detector_status/): Real-time and historical detector operational status
- [LIGO Livingston Observatory](https://www.ligo.caltech.edu/LA): L1 facility information
- [LIGO Hanford Observatory](https://www.ligo.caltech.edu/WA): H1 facility information
- [O4a Data Release Notes](https://gwosc.org/O4/): Available detectors for O4a
- [Multi-Detector GW Analysis](https://arxiv.org/abs/1901.08580): Coherent vs incoherent combination methods

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
