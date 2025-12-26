# ADR-015 — Per-Event Feature Normalization

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-015 |
| **Title** | Per-Event Feature Normalization |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Features extracted from spectrograms have different scales depending on:
- **Event SNR**: Loud events have higher amplitude features
- **Distance**: Closer sources are louder
- **Detector sensitivity**: O3 more sensitive than O1
- **Noise conditions**: Varying PSD across times and runs

Without normalization, features from a loud, nearby BBH might dominate features from a quiet, distant BNS. The ML model might learn "louder = BBH" rather than physical morphology differences.

Normalization strategies:
- **Global normalization**: z-score across entire dataset
- **Per-event normalization**: z-score within each event independently
- **Per-run normalization**: Normalize separately for O1, O2, O3
- **No normalization**: Use raw feature values

---

## 2. Problem Statement

How should we normalize spectrogram features to:
1. Make features comparable across events with different SNRs
2. Reduce domain shift between observing runs
3. Preserve class-discriminative information
4. Not introduce artifacts

**Key Question**: What normalization strategy best supports classification while controlling for SNR and run variations?

---

## 3. Decision Drivers

1. **SNR Independence**: Classification should depend on morphology, not loudness

2. **Run Robustness**: Model should generalize across O1-O3-O4a despite sensitivity changes

3. **Information Preservation**: Normalization shouldn't destroy class-relevant information

4. **Simplicity**: Normalization should be straightforward to apply and understand

5. **Consistency**: Same normalization for training and inference

---

## 4. Considered Options

### Option A: Per-Event Normalization

**Description**: Normalize features independently for each event. Each event's features have mean=0, std=1.

**Pros**:
- Removes SNR dependence completely
- Each event on equal footing regardless of loudness
- Robust to run-to-run sensitivity changes
- Features reflect relative shape, not absolute amplitude

**Cons**:
- Loses absolute amplitude information
- Very loud/quiet events become indistinguishable by loudness
- May lose some discriminative information

### Option B: Global Normalization

**Description**: Compute mean/std across all training events. Apply same transform to all.

**Pros**:
- Preserves relative differences between events
- Standard approach in ML

**Cons**:
- SNR differences dominate feature differences
- O3 events systematically different from O1
- Run-specific biases persist

### Option C: Per-Run Normalization

**Description**: Compute separate mean/std for each run (O1, O2, O3). Normalize within run.

**Pros**:
- Accounts for run-specific sensitivity
- Middle ground between global and per-event

**Cons**:
- Requires knowing which run each event belongs to
- O4a has no training statistics—must use O3 or heuristic
- Still doesn't handle SNR variation within run

### Option D: Min-Max Normalization Per-Event

**Description**: Scale each event's features to [0, 1] range.

**Pros**:
- Bounded features
- Removes absolute amplitude

**Cons**:
- Sensitive to outliers
- Less statistically principled than z-score

---

## 5. Decision Outcome

**Chosen Option**: Option A — Per-Event Normalization (z-score)

**Implementation**:
For each event $i$ with feature vector $\mathbf{x}_i = [x_{i1}, x_{i2}, ..., x_{in}]$:

$$\tilde{x}_{ij} = \frac{x_{ij} - \mu_i}{\sigma_i}$$

Where:
- $\mu_i = \frac{1}{n}\sum_j x_{ij}$ is the mean of event $i$'s features
- $\sigma_i = \sqrt{\frac{1}{n}\sum_j (x_{ij} - \mu_i)^2}$ is the std

**Rationale**:

1. **SNR Removal**: A BBH at 100 Mpc and a BBH at 500 Mpc have the same chirp structure but different amplitudes. Per-event normalization makes their feature vectors similar, which is correct.

2. **Domain Shift Mitigation**: O4a detectors are more sensitive—events will be louder on average. Per-event normalization makes O4a events comparable to O1-O3 despite this.

3. **Morphology Focus**: We want to classify based on chirp shape (fast/slow, high-freq/low-freq), not absolute energy. Per-event normalization emphasizes relative feature relationships.

4. **Validation**: This is equivalent to asking "what's the shape of this signal?" not "how loud is this signal?"

**Note**: Some features (like chirp duration) are inherently normalized. Normalization primarily affects amplitude-related features.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Run-agnostic**: Same model works on O1, O3, O4a without adjustment
- **SNR-robust**: Classification not biased by source distance
- **Reduced domain shift**: Major source of train-test difference controlled
- **Interpretable**: Features represent relative morphology

### 6.2 Negative Consequences

- **Lost absolute information**: Cannot distinguish loud from quiet events
- **Potential information loss**: If absolute amplitude is class-correlated (unlikely)
- **Edge cases**: Single-feature events or constant features → division by zero (handle with epsilon)

### 6.3 Neutral Consequences

- Must apply same normalization at inference time
- Normalization is part of feature pipeline, not separate step

---

## 7. Validation

**Success Criteria**:
- Features have approximately mean=0, std=1 per event (verify)
- No correlation between event SNR and predicted class (after normalization)
- Model performance stable across O1/O2/O3 subsets
- No division-by-zero errors

**Review Date**: Week 3 (feature extraction)

**Reversal Trigger**:
- Per-event normalization hurts accuracy vs. global
- Evidence that absolute amplitude is class-informative

---

## 8. Implementation Notes

### 8.1 Basic Per-Event Normalization

```python
import numpy as np

def normalize_per_event(features, epsilon=1e-8):
    """
    Normalize feature vector for a single event.
    
    Parameters
    ----------
    features : np.ndarray
        Feature vector for one event, shape (n_features,)
    epsilon : float
        Small constant to avoid division by zero
    
    Returns
    -------
    np.ndarray
        Normalized features
    """
    mean = np.mean(features)
    std = np.std(features) + epsilon
    
    return (features - mean) / std
```

### 8.2 Batch Normalization

```python
def normalize_feature_matrix(feature_matrix, axis=1, epsilon=1e-8):
    """
    Normalize features per-event for entire dataset.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape (n_events, n_features)
    axis : int
        Axis along which to normalize (1 = per event)
    
    Returns
    -------
    np.ndarray
        Normalized feature matrix
    """
    mean = np.mean(feature_matrix, axis=axis, keepdims=True)
    std = np.std(feature_matrix, axis=axis, keepdims=True) + epsilon
    
    return (feature_matrix - mean) / std
```

### 8.3 Feature Pipeline Integration

```python
class FeatureExtractor:
    def __init__(self, normalize='per_event'):
        self.normalize = normalize
    
    def extract(self, spectrogram):
        """Extract and normalize features from spectrogram."""
        # Extract raw features
        features = {
            'chirp_slope': self._compute_chirp_slope(spectrogram),
            'peak_frequency': self._compute_peak_freq(spectrogram),
            'duration': self._compute_duration(spectrogram),
            'low_band_energy': self._compute_band_energy(spectrogram, 20, 100),
            'mid_band_energy': self._compute_band_energy(spectrogram, 100, 300),
            'high_band_energy': self._compute_band_energy(spectrogram, 300, 1000),
            # ... more features
        }
        
        # Convert to array
        feature_vector = np.array(list(features.values()))
        
        # Normalize
        if self.normalize == 'per_event':
            feature_vector = normalize_per_event(feature_vector)
        
        return feature_vector, list(features.keys())
```

### 8.4 Spectrogram Normalization (for CNNs)

For CNN input (spectrogram images), per-event normalization is also applied:

```python
def normalize_spectrogram(spectrogram, epsilon=1e-8):
    """
    Normalize spectrogram for CNN input.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        Shape (freq_bins, time_bins)
    
    Returns
    -------
    np.ndarray
        Normalized spectrogram with mean=0, std=1
    """
    mean = np.mean(spectrogram)
    std = np.std(spectrogram) + epsilon
    
    return (spectrogram - mean) / std
```

### 8.5 Handling Edge Cases

```python
def safe_normalize(features, epsilon=1e-8):
    """Normalize with edge case handling."""
    mean = np.mean(features)
    std = np.std(features)
    
    # Handle constant features (std = 0)
    if std < epsilon:
        # Return zeros (centered at mean)
        return np.zeros_like(features)
    
    return (features - mean) / std
```

### 8.6 Verification Code

```python
def verify_normalization(feature_matrix):
    """Verify per-event normalization is correct."""
    for i in range(len(feature_matrix)):
        event_features = feature_matrix[i]
        mean = np.mean(event_features)
        std = np.std(event_features)
        
        assert np.abs(mean) < 1e-6, f"Event {i}: mean={mean}"
        assert np.abs(std - 1.0) < 1e-6, f"Event {i}: std={std}"
    
    print("✓ All events correctly normalized")
```

---

## 9. References

- [Feature Scaling](https://scikit-learn.org/stable/modules/preprocessing.html): sklearn documentation
- [Normalization in Deep Learning](https://arxiv.org/abs/1502.03167): Batch normalization paper
- [Domain Adaptation](https://arxiv.org/abs/1505.07818): Feature normalization for transfer
- [Signal Processing Normalization](https://www.sciencedirect.com/topics/engineering/signal-normalization): General principles

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
