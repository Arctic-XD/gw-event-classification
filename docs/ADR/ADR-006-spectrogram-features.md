# ADR-006 — Spectrogram + Quantitative Features as Primary Input

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-006 |
| **Title** | Representation: Spectrogram + Quantitative Features as Primary Input |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Gravitational-wave signals can be represented in multiple ways for machine learning:

1. **Raw strain time series**: The direct detector output h(t)
2. **Frequency domain**: Fourier transform of strain
3. **Time-frequency representations**: Spectrograms, wavelets, Q-transforms
4. **Matched filter outputs**: SNR time series, chi-squared values
5. **Parameter estimates**: Masses, spins, distances from full PE

Each representation has trade-offs between information content, interpretability, computational cost, and suitability for ML algorithms.

**Spectrograms** (time-frequency representations) are particularly attractive for CBC classification because:
- CBC signals have characteristic "chirp" morphology in time-frequency space
- Chirp rate relates directly to source masses via GW physics
- Visual inspection of spectrograms is intuitive
- Image-based representations enable CNN approaches

However, spectrograms can be processed in two ways:
- **Direct image input**: Feed spectrogram image to CNN, let network learn features
- **Quantitative features**: Extract hand-crafted features (slope, duration, peak frequency) from spectrogram

This ADR establishes which approach is primary for this project.

---

## 2. Problem Statement

We need to determine the primary data representation for our classification pipeline:
1. What form should input data take for the ML model?
2. Should we emphasize interpretable features or end-to-end learning?
3. How does this choice align with project goals and constraints?
4. What is the role of spectrograms vs features?

**Key Question**: Should the project prioritize hand-crafted spectrogram features (interpretable) or CNN-on-images (potentially higher performance)?

---

## 3. Decision Drivers

1. **Interpretability**: Science fair judges and reviewers value understanding *why* the model makes decisions. Features like "chirp slope" are intuitive; CNN hidden layers are not.

2. **Sample Size**: Only ~90 real O1-O3 events. Hand-crafted features work better with small datasets; CNNs risk overfitting.

3. **Physics Alignment**: The project title emphasizes "quantitative feature extraction." Features grounded in GW physics are more scientifically interesting than black-box predictions.

4. **Two-Phase Strategy**: Phase 1 (features + classical ML) must work before Phase 2 (CNN).

5. **Novelty**: Many papers use CNNs on spectrograms. Fewer rigorously analyze which spectrogram features matter for classification.

6. **Computational Cost**: Feature extraction once, then train many models quickly. CNNs require GPUs and longer training.

7. **Ablation Studies**: Features enable clear ablation (e.g., "removing chirp slope reduces accuracy by 15%"). CNN ablations are less interpretable.

8. **Failure Analysis**: When a feature-based model fails, we can diagnose *which* features were misleading. CNN failures are harder to explain.

---

## 4. Considered Options

### Option A: Quantitative Features as Primary (CNN as Secondary)

**Description**: Extract interpretable features from spectrograms (chirp slope, duration, peak frequency, band energies). Use these for Phase 1 classification. CNN-on-images is Phase 2 enhancement.

**Pros**:
- Maximum interpretability
- Works with small datasets
- Clear physics connection
- Enables feature importance analysis
- Supports ablation studies
- Aligns with project title
- Fast training iteration

**Cons**:
- Feature engineering requires domain expertise
- May miss patterns humans don't anticipate
- Potentially lower peak accuracy than CNN
- More upfront design work

### Option B: CNN on Spectrogram Images as Primary

**Description**: Generate spectrogram images, train CNN directly. Let network learn optimal features. No manual feature extraction.

**Pros**:
- Potentially highest accuracy
- No feature engineering needed
- Can discover unexpected patterns
- Leverages transfer learning (pretrained ImageNet models)

**Cons**:
- Black box—hard to explain decisions
- Needs large dataset (risky with ~90 events)
- Requires GPU for training
- Overfitting risk
- Less interpretable for science fair

### Option C: Parallel Development (Features and CNN Equal)

**Description**: Develop both approaches simultaneously from the start. Compare and combine.

**Pros**:
- Covers both interpretable and high-performance approaches
- Enables ensemble methods
- Comprehensive comparison

**Cons**:
- Double the development effort
- May not complete either properly in time
- Diffuse focus

### Option D: Raw Time Series Input

**Description**: Input raw strain time series to 1D CNN or RNN. Skip spectrogram step.

**Pros**:
- No information loss from time-frequency transform
- Network learns optimal representation

**Cons**:
- Very long sequences (64s × 4096 Hz = 262,144 samples)
- Loses intuitive time-frequency visualization
- Harder to interpret
- More prone to learning noise artifacts

---

## 5. Decision Outcome

**Chosen Option**: Option A — Quantitative Features as Primary (CNN as Secondary)

**Rationale**:

This decision directly supports the project's core scientific contribution: demonstrating that **interpretable, physics-grounded features** can effectively classify GW events.

1. **Title Alignment**: The project is explicitly about "Quantitative Feature Extraction." Features must be primary to fulfill this promise.

2. **Interpretability Value**: A model that says "BBH because chirp slope = 2.5 Hz/s, duration = 0.3s, peak frequency = 150 Hz" is far more scientifically valuable than "BBH with probability 0.92." We can:
   - Validate predictions against physics expectations
   - Identify which features differ between BBH and NS-present
   - Discover unexpected correlations
   - Explain failures meaningfully

3. **Sample Size Reality**: With ~90 events, CNNs will struggle to generalize without massive augmentation. Feature-based models (Random Forest, XGBoost) are designed for exactly this regime.

4. **Two-Phase Flow**: Phase 1 must establish that features work. Phase 2 CNN can then be compared against this interpretable baseline. If CNN beats features, that's interesting. If features match CNN, that's scientifically important (interpretability without accuracy loss).

5. **Contribution Clarity**: Many papers train CNNs on GW spectrograms. Fewer rigorously answer: "Which spectrogram features actually matter for classification?" This project provides that answer.

**Phase 2 CNN Role**: The CNN becomes a *comparison model*:
- Does end-to-end learning beat hand-crafted features?
- Can physics-informed loss improve CNN interpretability?
- Do CNN attention maps highlight same regions as our features?

---

## 6. Consequences

### 6.1 Positive Consequences

- **Publishable insights**: Feature importance analysis is directly publishable
- **Judge-friendly**: Science fair presentations can explain "why" not just "what"
- **Physics validation**: Features can be validated against GW theory
- **Quick iteration**: Train new model in seconds once features extracted
- **Diagnostic capability**: Clear failure analysis pathway
- **Robust to small data**: Features don't overfit 90 samples

### 6.2 Negative Consequences

- **Ceiling on performance**: May not achieve absolute best accuracy
- **Feature engineering effort**: Must design and validate features carefully
- **Potential blind spots**: Human-designed features may miss relevant patterns
- **Less "AI-impressive"**: CNNs sound more cutting-edge than Random Forest

### 6.3 Neutral Consequences

- Spectrograms are still computed (as intermediate representation for features)
- CNN remains as Phase 2 comparison
- Project spans both "traditional ML" and "deep learning"

---

## 7. Validation

**Success Criteria**:
- At least 10 distinct spectrogram features extracted
- Feature-based model achieves >80% accuracy on O1-O3 CV
- Feature importance ranking identifies 3+ physics-meaningful features in top 5
- Phase 2 CNN comparison shows reasonable delta (<10%) or features win

**Review Date**: Week 4 (Phase 1 completion)

**Reversal Trigger**:
- Feature-based model accuracy <65% (clearly inadequate)
- All feature importances are similar (no clear signals)
- CNN dramatically outperforms features (>25% accuracy gap)

---

## 8. Implementation Notes

### Feature Categories

**1. Chirp Geometry Features**
- `chirp_slope_mean`: Mean frequency change rate (Hz/s)
- `chirp_slope_std`: Variation in chirp slope
- `chirp_duration`: Time from first detection to merger
- `time_to_peak`: Time from window start to maximum amplitude

**2. Frequency Distribution Features**
- `peak_frequency`: Frequency at maximum amplitude
- `frequency_centroid`: Energy-weighted average frequency
- `bandwidth`: Spectral width of signal
- `low_band_energy`: Power in 20-100 Hz
- `mid_band_energy`: Power in 100-300 Hz
- `high_band_energy`: Power in 300-1000 Hz

**3. Amplitude Features**
- `snr_estimate`: Approximate signal-to-noise ratio
- `peak_amplitude`: Maximum spectrogram value
- `amplitude_asymmetry`: Pre vs post merger energy ratio

**4. Texture Features (Optional)**
- `spectral_entropy`: Uniformity of frequency distribution
- `ridge_coherence`: How clean/clear is the chirp track

### Feature Extraction Pipeline
```python
def extract_features(spectrogram, time_axis, freq_axis):
    """Extract all features from a spectrogram."""
    features = {}
    
    # Find ridge (dominant frequency track)
    ridge_freq, ridge_time = find_ridge(spectrogram, freq_axis, time_axis)
    
    # Chirp geometry
    features['chirp_slope_mean'] = compute_chirp_slope(ridge_freq, ridge_time)
    features['chirp_duration'] = ridge_time[-1] - ridge_time[0]
    features['peak_frequency'] = ridge_freq[-1]
    
    # Band energies
    features['low_band_energy'] = band_power(spectrogram, freq_axis, 20, 100)
    features['mid_band_energy'] = band_power(spectrogram, freq_axis, 100, 300)
    features['high_band_energy'] = band_power(spectrogram, freq_axis, 300, 1000)
    
    # ... additional features
    
    return features
```

### CNN Comparison (Phase 2)
```python
# Phase 2: Compare CNN vs features
cnn_accuracy = evaluate_cnn(test_spectrograms, test_labels)
feature_accuracy = evaluate_rf(test_features, test_labels)

print(f"CNN: {cnn_accuracy:.1%}, Features: {feature_accuracy:.1%}")
print(f"Delta: {cnn_accuracy - feature_accuracy:.1%}")
```

---

## 9. References

- [Feature Engineering for GW](https://arxiv.org/abs/2003.11880): Prior work on GW spectrogram features
- [Interpretable ML](https://christophm.github.io/interpretable-ml-book/): Importance of interpretability
- [CNN vs Features](https://arxiv.org/abs/1702.08138): When manual features beat deep learning
- [Random Forest Small Samples](https://scikit-learn.org/stable/modules/ensemble.html#random-forests): RF robustness to small N
- [GW Chirp Physics](https://arxiv.org/abs/gr-qc/9402014): Chirp mass and frequency relationship

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
