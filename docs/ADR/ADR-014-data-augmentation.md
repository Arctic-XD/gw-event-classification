# ADR-014 — Light Data Augmentation

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-014 |
| **Title** | Light Data Augmentation for Real Events |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

With only ~91 real events in O1-O3, even the feature-based Phase 1 model may benefit from data augmentation. Data augmentation creates modified versions of existing samples to increase effective training set size and improve model robustness.

However, gravitational-wave signals have specific physical properties that constrain valid augmentations:
- The chirp structure is determined by source physics
- Frequency evolution follows precise GR relationships
- Signal timing relative to merger is meaningful

Invalid augmentations could:
- Create physically impossible signals
- Change the class label (e.g., stretching a BNS to look like BBH)
- Introduce artifacts the model learns as features

---

## 2. Problem Statement

Which augmentation techniques are:
1. Physically valid (don't violate GW physics)
2. Effective (actually improve model generalization)
3. Appropriate for spectrogram-based features

**Key Question**: What augmentation strategy balances increased data diversity with physical validity?

---

## 3. Decision Drivers

1. **Physical Validity**: Augmented samples must remain physically plausible

2. **Class Preservation**: Augmentation must not change the true class label

3. **Feature Robustness**: Features should be invariant to augmented variations

4. **Simplicity**: Augmentation should be simple to implement and validate

5. **Noise Reality**: Real detector noise is complex—augmentation should respect this

---

## 4. Considered Options

### Option A: Light Augmentation (Time Jitter + Noise Injection)

**Description**: Apply minimal, physics-preserving augmentations:
- Time jitter: Shift merger time by ±0.5-1s within window
- Noise injection: Add Gaussian noise to simulate varying SNR

**Pros**:
- Physics-preserving (chirp structure unchanged)
- Simple to implement
- Clearly doesn't change class
- Improves robustness to timing/SNR variations

**Cons**:
- Limited diversity increase
- May not help much

### Option B: Aggressive Augmentation (Including Frequency/Time Scaling)

**Description**: Apply more transformations:
- Time stretching
- Frequency shifting
- Amplitude scaling
- Spectrogram rotation

**Pros**:
- More diversity

**Cons**:
- Time/frequency stretching changes chirp physics
- Could create BBH-like signals from BNS data
- May teach model wrong patterns

### Option C: No Augmentation on Real Events

**Description**: Only use real events as-is. Rely on synthetic data for augmentation.

**Pros**:
- No risk of invalid augmentation
- Real data is "pure"

**Cons**:
- Only 91 samples
- May overfit to specific noise realizations
- Doesn't improve Phase 1 (no synthetic yet)

---

## 5. Decision Outcome

**Chosen Option**: Option A — Light Augmentation

**Allowed Augmentations**:

| Augmentation | Range | Rationale |
|--------------|-------|-----------|
| Time jitter | ±0.5-1.0s | Merger time uncertainty exists; features should be robust |
| Gaussian noise | σ = 0.1-0.5 × signal | Simulates varying SNR; real events have range of SNR |
| Amplitude scaling | 0.8-1.2× | Simulates distance variation; class-preserving |

**Forbidden Augmentations**:

| Augmentation | Reason |
|--------------|--------|
| Time stretching | Changes chirp rate → changes apparent chirp mass |
| Frequency shifting | Changes peak frequency → changes apparent source type |
| Spectrogram rotation | Not physically meaningful |
| Cutout/masking | May remove critical chirp features |

**Rationale**:

1. **Time Jitter is Safe**: The exact merger time within our window is arbitrary. Shifting by ≤1s doesn't change any physical property—just where in the spectrogram the signal appears.

2. **Noise Injection is Safe**: Real events have varying SNR (8-100+). Adding noise to high-SNR events simulates lower-SNR detection conditions. This is physics-preserving.

3. **Amplitude Scaling is Safe**: Distance affects amplitude, not spectral shape. A closer/farther source of the same type has same chirp pattern.

4. **Time/Frequency Stretching is Dangerous**: The chirp rate directly encodes chirp mass. Stretching time changes apparent mass → could change class.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Robustness**: Model robust to timing and SNR variations
- **Modest data increase**: ~3-5x effective samples with augmentation
- **Safe augmentation**: No risk of class label corruption
- **Physical validity**: All augmented samples are physically plausible

### 6.2 Negative Consequences

- **Limited diversity**: Light augmentation provides less variety than aggressive
- **Not a replacement for synthetic**: Still need PyCBC injection for Phase 2
- **Implementation overhead**: Must implement augmentation pipeline

### 6.3 Neutral Consequences

- Augmentation applied online during training (not stored)
- Same augmentation strategy for Phase 1 and Phase 2
- Augmentation parameters become hyperparameters to tune

---

## 7. Validation

**Success Criteria**:
- Augmented samples visually maintain chirp structure
- Model with augmentation ≥ model without augmentation (accuracy)
- No systematic bias introduced by augmentation

**Review Date**: Week 4 (Phase 1 ablation)

**Reversal Trigger**:
- Augmentation hurts performance
- Augmented samples look physically wrong

---

## 8. Implementation Notes

### 8.1 Time Jitter Implementation

```python
import numpy as np

def apply_time_jitter(spectrogram, max_shift_bins=10):
    """
    Shift spectrogram in time dimension.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        Shape (freq_bins, time_bins)
    max_shift_bins : int
        Maximum shift in time bins
    
    Returns
    -------
    np.ndarray
        Jittered spectrogram
    """
    shift = np.random.randint(-max_shift_bins, max_shift_bins + 1)
    
    if shift == 0:
        return spectrogram
    
    # Roll with zero-padding
    jittered = np.roll(spectrogram, shift, axis=1)
    
    if shift > 0:
        jittered[:, :shift] = 0
    else:
        jittered[:, shift:] = 0
    
    return jittered
```

### 8.2 Noise Injection Implementation

```python
def add_gaussian_noise(spectrogram, snr_reduction=0.3):
    """
    Add Gaussian noise to spectrogram.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram
    snr_reduction : float
        Noise level as fraction of signal RMS
    
    Returns
    -------
    np.ndarray
        Noisy spectrogram
    """
    signal_rms = np.sqrt(np.mean(spectrogram**2))
    noise_level = snr_reduction * signal_rms
    
    noise = np.random.normal(0, noise_level, spectrogram.shape)
    
    return spectrogram + noise
```

### 8.3 Amplitude Scaling Implementation

```python
def scale_amplitude(spectrogram, scale_range=(0.8, 1.2)):
    """
    Randomly scale spectrogram amplitude.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram
    scale_range : tuple
        (min_scale, max_scale)
    
    Returns
    -------
    np.ndarray
        Scaled spectrogram
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return spectrogram * scale
```

### 8.4 Combined Augmentation Pipeline

```python
class SpectrogramAugmentation:
    def __init__(self, 
                 time_jitter_bins=10,
                 noise_level=0.3,
                 amplitude_range=(0.8, 1.2),
                 p=0.5):
        """
        Parameters
        ----------
        p : float
            Probability of applying each augmentation
        """
        self.time_jitter_bins = time_jitter_bins
        self.noise_level = noise_level
        self.amplitude_range = amplitude_range
        self.p = p
    
    def __call__(self, spectrogram):
        # Time jitter
        if np.random.random() < self.p:
            spectrogram = apply_time_jitter(spectrogram, self.time_jitter_bins)
        
        # Noise injection
        if np.random.random() < self.p:
            spectrogram = add_gaussian_noise(spectrogram, self.noise_level)
        
        # Amplitude scaling
        if np.random.random() < self.p:
            spectrogram = scale_amplitude(spectrogram, self.amplitude_range)
        
        return spectrogram
```

### 8.5 PyTorch Dataset with Augmentation

```python
class AugmentedGWDataset(torch.utils.data.Dataset):
    def __init__(self, spectrograms, labels, augment=True):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augment = augment
        
        if augment:
            self.augmenter = SpectrogramAugmentation()
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx].copy()
        label = self.labels[idx]
        
        if self.augment:
            spec = self.augmenter(spec)
        
        return torch.FloatTensor(spec), label
```

### 8.6 Ablation Study Design

```python
# Compare with and without augmentation
results = {}

# No augmentation
model_no_aug = train_model(train_data, augment=False)
results['no_aug'] = evaluate(model_no_aug, test_data)

# With augmentation
model_aug = train_model(train_data, augment=True)
results['with_aug'] = evaluate(model_aug, test_data)

# Report delta
print(f"Augmentation effect: {results['with_aug'] - results['no_aug']:.1%}")
```

---

## 9. References

- [Data Augmentation Survey](https://arxiv.org/abs/1906.11172): General augmentation techniques
- [GW Signal Augmentation](https://arxiv.org/abs/2012.00493): Augmentation for GW detection
- [Physics-Preserving Augmentation](https://arxiv.org/abs/2106.02430): Domain-specific constraints
- [Spectrogram Augmentation](https://arxiv.org/abs/1904.08779): SpecAugment for audio

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
