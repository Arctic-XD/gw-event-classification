# ADR-016 — Confident Events Only

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-016 |
| **Title** | Use Confident Events Only (Exclude Marginal Detections) |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

GWTC catalogs categorize events by detection confidence:

- **Confident events**: False alarm rate (FAR) < 1 per year, high SNR, clear signals
- **Marginal events**: 1 < FAR < 2 per year, borderline detections, noisier
- **Sub-threshold triggers**: FAR > 2 per year, likely noise fluctuations

The GWTC catalogs explicitly separate these categories:
- GWTC-3 contains ~90 confident events and ~35 marginal candidates
- Marginal events may be real or noise—their labels are uncertain

For supervised ML, label quality is crucial. Training on incorrectly labeled examples degrades model performance. Marginal events have:
- Higher probability of being noise misclassified as signal
- More uncertain source parameters
- Potentially ambiguous source type classifications

---

## 2. Problem Statement

Should we include marginal/low-confidence events in our training dataset?

Considerations:
1. More data is generally better for ML
2. But noisy labels hurt more than they help
3. Marginal events may have uncertain source classifications
4. Science fair scope favors clean, defensible dataset

**Key Question**: What confidence threshold should we apply to event selection?

---

## 3. Decision Drivers

1. **Label Quality**: Supervised learning requires accurate labels

2. **Dataset Size**: Already have limited data; excluding events reduces further

3. **Classification Certainty**: Marginal events may have ambiguous BBH/NSBH labels

4. **Reproducibility**: Using only confident events is a clear, reproducible criterion

5. **Scope Appropriateness**: Science fair project should use cleanest data

6. **Signal Quality**: Confident events have cleaner spectrograms for feature extraction

---

## 4. Considered Options

### Option A: Confident Events Only (FAR < 1/year)

**Description**: Use only events from GWTC "confident" lists. Exclude all marginal candidates.

**Dataset Size**:
- O1-O3 confident: ~90 events
- Marginal excluded: ~35 events

**Pros**:
- Clean labels—high confidence in source type
- Clear spectrograms for feature extraction
- Reproducible criterion
- Aligns with published GWTC analysis

**Cons**:
- Smaller dataset
- Loses some potentially real events
- Marginal events might provide diversity

### Option B: Include Marginal Events

**Description**: Use all GWTC events including marginal candidates.

**Dataset Size**:
- Total: ~125 events

**Pros**:
- More training data
- Some marginal events are likely real

**Cons**:
- Label noise—some marginal events may be noise
- Source type more uncertain for marginal
- Harder to justify to reviewers

### Option C: Confidence-Weighted Training

**Description**: Include all events but weight by confidence in loss function.

**Pros**:
- Uses all data
- Accounts for label uncertainty

**Cons**:
- Complex implementation
- Still includes potentially wrong labels
- Weights hard to determine

---

## 5. Decision Outcome

**Chosen Option**: Option A — Confident Events Only

**Rationale**:

1. **Label Integrity**: A classification model is only as good as its labels. Marginal events have ~10-50% probability of being noise fluctuations. Including them introduces systematic label noise.

2. **Source Type Ambiguity**: Some marginal events have disputed classifications (e.g., "BBH or terrestrial?"). We cannot train a classifier on events where we don't know the true class.

3. **Spectrogram Quality**: Marginal events by definition have lower SNR. Their spectrograms may not show clear chirp structure, making feature extraction unreliable.

4. **Scientific Defensibility**: "We used GWTC confident events" is a clear, citable criterion. "We used confident plus some marginal events" invites questions about selection bias.

5. **90 Events is Adequate**: For Phase 1 feature-based models, 90 events (85 BBH + 5 NS-present) is workable. Phase 2 synthetic data solves the quantity problem properly.

6. **Precedent**: Many published GW ML papers use only confident events for exactly these reasons.

**Future Work**: Note marginal events as potential extension once classifier is validated on confident events.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Clean training signal**: Model learns from reliable examples
- **Defensible methodology**: Clear selection criterion
- **Better feature quality**: High-SNR events → cleaner spectrograms
- **Confident evaluation**: Test metrics reflect true performance

### 6.2 Negative Consequences

- **Reduced dataset**: ~35 fewer events
- **Potential bias**: Confident events may not represent full population
- **Missed diversity**: Some marginal events could add useful variation

### 6.3 Neutral Consequences

- Must explicitly document selection criterion
- Marginal events can be held out for future analysis
- Consistent with published GW classification work

---

## 7. Validation

**Success Criteria**:
- Event manifest contains only events from GWTC confident lists
- All events have unambiguous source type classification
- No events with FAR > 1/year included

**Review Date**: Week 1 (manifest creation)

**Reversal Trigger**:
- Confident event count is too low (<50)
- Marginal events are reclassified as confident in catalog updates
- Need for marginal events demonstrated by poor model diversity

---

## 8. Implementation Notes

### 8.1 GWTC Confident Event Lists

| Catalog | Run | Confident Events | Notes |
|---------|-----|------------------|-------|
| GWTC-1 | O1+O2 | 11 | GW150914 through GW170823 |
| GWTC-2.1 | O3a | 39 | Includes reruns of O1/O2 |
| GWTC-3 | O3b | ~35 | Extends through O3b |
| O4a | O4a | TBD | Public release events |

### 8.2 Event Selection Query

```python
# Using GWOSC API to get confident events
import requests

def get_confident_events(catalog='GWTC-3'):
    """
    Retrieve confident events from GWOSC.
    
    Parameters
    ----------
    catalog : str
        GWTC catalog name
    
    Returns
    -------
    list
        List of confident event names
    """
    url = f"https://gwosc.org/eventapi/json/query/"
    
    # Query for confident events only
    params = {
        'catalog': catalog,
        'mass1-min': 1.0,  # Exclude bad events
        # GWOSC API filters for confident events
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Filter for confident (FAR < 1/year)
    confident = []
    for event in data['events']:
        far = event.get('far', float('inf'))
        if far < 1.0 / (365.25 * 24 * 3600):  # < 1/year in Hz
            confident.append(event['name'])
    
    return confident
```

### 8.3 Manifest Generation

```python
import pandas as pd

def create_confident_manifest():
    """Create manifest of confident events only."""
    
    # Get confident events from each catalog
    gwtc1 = get_confident_events('GWTC-1')
    gwtc2 = get_confident_events('GWTC-2.1')
    gwtc3 = get_confident_events('GWTC-3')
    
    # Combine (later catalogs supersede earlier)
    all_events = list(set(gwtc1 + gwtc2 + gwtc3))
    
    # Build manifest
    manifest = []
    for event in all_events:
        info = get_event_info(event)  # Get GPS time, classification, etc.
        
        manifest.append({
            'event_name': event,
            'gps_time': info['gps_time'],
            'run': info['run'],
            'class_label': info['classification'],
            'far': info['far'],
            'network_snr': info['snr'],
            'catalog': info['catalog'],
            'confident': True
        })
    
    df = pd.DataFrame(manifest)
    df.to_csv('manifests/event_manifest.csv', index=False)
    
    return df
```

### 8.4 Documentation Template

```markdown
## Event Selection Criteria

**Source**: GWTC-1, GWTC-2.1, GWTC-3 catalogs from GWOSC

**Confidence Threshold**: FAR < 1 per year (confident events only)

**Excluded**: 
- Marginal candidates (FAR > 1/year)
- Retracted events
- Events with ambiguous classification

**Result**: N confident events (X BBH, Y BNS, Z NSBH)

**Rationale**: Clean labels essential for supervised classification. 
Marginal events documented as future work.
```

### 8.5 Verification

```python
def verify_manifest(manifest_path):
    """Verify all events meet confident criteria."""
    df = pd.read_csv(manifest_path)
    
    issues = []
    
    for _, row in df.iterrows():
        # Check FAR
        if row['far'] > 1.0 / (365.25 * 24 * 3600):
            issues.append(f"{row['event_name']}: FAR too high")
        
        # Check classification
        if row['class_label'] not in ['BBH', 'BNS', 'NSBH']:
            issues.append(f"{row['event_name']}: Unknown class")
    
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All events verified as confident")
```

---

## 9. References

- [GWTC-3 Paper](https://arxiv.org/abs/2111.03606): Confident vs marginal event classification
- [GWOSC Event Catalogs](https://gwosc.org/eventapi/): Event API documentation
- [Label Noise in ML](https://arxiv.org/abs/1905.04814): Impact of noisy labels on training
- [GW Detection Statistics](https://arxiv.org/abs/1901.08580): FAR and significance calculations

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
