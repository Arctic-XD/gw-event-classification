# ADR-018 — Ridge Extraction via Peak Tracking

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-018 |
| **Title** | Ridge Extraction via Peak Tracking |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Gravitational-wave chirp signals trace a characteristic path through time-frequency space—the "chirp ridge." This ridge follows:

$$f(t) \propto (t_c - t)^{-3/8}$$

Where the chirp accelerates toward higher frequencies as the binary inspirals. The shape of this ridge (slope, duration, endpoint frequency) encodes source properties like chirp mass.

**Ridge extraction** is the process of identifying and tracing this dominant frequency track through the spectrogram. Once extracted, the ridge can be characterized by features like:
- Mean/median slope (chirp rate)
- Duration (time in-band)
- Peak frequency (frequency at merger)
- Slope variability

Multiple algorithms exist for ridge extraction, ranging from simple to sophisticated.

---

## 2. Problem Statement

We need to extract the chirp ridge from spectrograms to compute physics-motivated features. The method must:
1. Reliably identify the chirp track in noisy spectrograms
2. Handle varying SNR (loud to faint events)
3. Be implementable within project timeline
4. Produce interpretable features

**Key Question**: What ridge extraction algorithm provides the best trade-off between reliability and implementation complexity?

---

## 3. Decision Drivers

1. **Simplicity**: Must be implementable in an afternoon, not a week

2. **Reliability**: Should work for most events without manual tuning

3. **Interpretability**: Ridge should visually correspond to the chirp

4. **Feature Quality**: Extracted features should discriminate BBH from NS-present

5. **Noise Robustness**: Should handle varying SNR levels

6. **Computational Cost**: Should be fast enough for 10,000+ samples

---

## 4. Considered Options

### Option A: Peak Tracking (Per-Time-Bin Maximum)

**Description**: For each time bin, find the frequency with maximum power. Connect these peaks to form the ridge. Apply smoothing and slope fitting.

**Algorithm**:
1. For each time column in spectrogram, find argmax(frequency)
2. Filter outliers (likely noise spikes)
3. Smooth with moving average or polynomial fit
4. Compute slope via linear regression

**Pros**:
- Very simple to implement
- Fast (single pass through data)
- Interpretable
- Works well for high-SNR events
- No hyperparameters besides smoothing window

**Cons**:
- Noise can corrupt peaks at low SNR
- Assumes single dominant track
- May miss merger where signal is brief

### Option B: Hough Transform

**Description**: Use Hough transform to find lines/curves in time-frequency space that match chirp templates.

**Pros**:
- Robust to noise
- Can find chirp even with gaps
- Principled mathematical framework

**Cons**:
- Complex implementation
- Requires chirp shape templates
- Computationally expensive
- Multiple hyperparameters

### Option C: Dynamic Programming / Viterbi

**Description**: Frame ridge extraction as path-finding problem. Find path through spectrogram that maximizes signal while penalizing frequency jumps.

**Pros**:
- Principled optimization
- Can enforce smoothness constraints
- Good for structured signals

**Cons**:
- Moderately complex implementation
- Requires defining transition costs
- Slower than peak tracking

### Option D: Wavelet Ridge Extraction

**Description**: Use continuous wavelet transform and trace ridges through scale-time space.

**Pros**:
- Native multi-resolution analysis
- Good for chirps

**Cons**:
- Requires wavelet machinery
- More complex than spectrogram approach
- Additional dependencies

---

## 5. Decision Outcome

**Chosen Option**: Option A — Peak Tracking

**Rationale**:

For a science fair project with limited time, peak tracking provides the best simplicity-reliability trade-off:

1. **Implementation Time**: Peak tracking can be implemented in <50 lines of Python. Hough transform or Viterbi would take significantly longer.

2. **Good Enough Performance**: For our purposes (feature extraction for classification), peak tracking is sufficient. We don't need perfect ridge extraction—we need features that discriminate BBH from NS-present.

3. **Interpretability**: Peak tracking produces a ridge that visually corresponds to the spectrogram maximum. Easy to visualize and validate.

4. **Robustness**: With median filtering and outlier rejection, peak tracking handles moderate noise. Our events are confident detections (SNR > 8), so extreme noise is rare.

5. **Speed**: O(time_bins) complexity enables processing thousands of spectrograms quickly.

**Enhancement Path**: If peak tracking proves inadequate, Viterbi is the logical next step. But implement simple first.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Fast implementation**: Working code in hours
- **Simple debugging**: Easy to visualize and fix issues
- **Quick iteration**: Can experiment with smoothing parameters easily
- **Good features**: Sufficient for classification task
- **Transparent**: Clear what algorithm is doing

### 6.2 Negative Consequences

- **Noise sensitivity**: May fail on very low SNR events (rare in confident set)
- **Merger region**: May miss very short merger phase
- **Complex ridges**: Won't handle multimodal tracks (very rare)
- **Not optimal**: More sophisticated methods could do better

### 6.3 Neutral Consequences

- Ridge quality depends on spectrogram quality
- Some manual inspection needed for validation
- Note limitations in documentation

---

## 7. Validation

**Success Criteria**:
- Ridge visually traces chirp for >90% of events
- Extracted slope correlates with expected physics (BBH = steeper)
- Features from ridge extraction discriminate BBH from NS-present
- Processing time <1 second per event

**Review Date**: Week 3 (feature extraction)

**Reversal Trigger**:
- Ridge extraction fails for >20% of events
- Features show no class discrimination
- Need identified for complex ridge structure analysis

---

## 8. Implementation Notes

### 8.1 Basic Peak Tracking

```python
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import linregress

def extract_ridge_peak_tracking(spectrogram, freq_axis, time_axis, 
                                  freq_range=(30, 500), smooth_window=5):
    """
    Extract chirp ridge via peak tracking.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        2D spectrogram, shape (n_freq, n_time)
    freq_axis : np.ndarray
        Frequency values for each row
    time_axis : np.ndarray
        Time values for each column
    freq_range : tuple
        (min_freq, max_freq) to consider
    smooth_window : int
        Median filter window size
    
    Returns
    -------
    dict
        Ridge properties and features
    """
    # Mask to relevant frequency range
    freq_mask = (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])
    spec_masked = spectrogram[freq_mask, :]
    freq_masked = freq_axis[freq_mask]
    
    # Find peak frequency at each time bin
    peak_indices = np.argmax(spec_masked, axis=0)
    peak_freqs = freq_masked[peak_indices]
    
    # Smooth with median filter (removes noise spikes)
    peak_freqs_smooth = median_filter(peak_freqs, size=smooth_window)
    
    # Remove outliers (frequencies that jump too much)
    peak_freqs_clean = reject_outliers(peak_freqs_smooth, m=2)
    
    # Fit linear slope (crude but effective)
    valid_mask = ~np.isnan(peak_freqs_clean)
    if np.sum(valid_mask) < 10:
        return None  # Not enough valid points
    
    slope, intercept, r_value, _, _ = linregress(
        time_axis[valid_mask], 
        peak_freqs_clean[valid_mask]
    )
    
    return {
        'ridge_freqs': peak_freqs_clean,
        'ridge_times': time_axis,
        'slope': slope,  # Hz/s
        'intercept': intercept,
        'r_squared': r_value**2,
        'peak_frequency': np.nanmax(peak_freqs_clean),
        'start_frequency': peak_freqs_clean[valid_mask][0],
        'duration': time_axis[valid_mask][-1] - time_axis[valid_mask][0]
    }

def reject_outliers(data, m=2):
    """Reject outliers more than m std from median."""
    data = data.copy()
    median = np.nanmedian(data)
    std = np.nanstd(data)
    mask = np.abs(data - median) > m * std
    data[mask] = np.nan
    return data
```

### 8.2 Ridge Features

```python
def compute_ridge_features(ridge_info):
    """Compute classification features from ridge."""
    if ridge_info is None:
        return None
    
    features = {
        # Chirp rate (key discriminator)
        'chirp_slope': ridge_info['slope'],
        'chirp_slope_abs': np.abs(ridge_info['slope']),
        
        # Frequency extent
        'peak_frequency': ridge_info['peak_frequency'],
        'start_frequency': ridge_info['start_frequency'],
        'frequency_span': ridge_info['peak_frequency'] - ridge_info['start_frequency'],
        
        # Duration
        'chirp_duration': ridge_info['duration'],
        
        # Fit quality (indicates chirp clarity)
        'ridge_r_squared': ridge_info['r_squared'],
    }
    
    return features
```

### 8.3 Visualization

```python
import matplotlib.pyplot as plt

def plot_ridge_overlay(spectrogram, freq_axis, time_axis, ridge_info,
                       title='Chirp Ridge Extraction'):
    """Plot spectrogram with extracted ridge overlaid."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Spectrogram
    ax.pcolormesh(time_axis, freq_axis, spectrogram, 
                  shading='auto', cmap='viridis')
    
    # Ridge overlay
    valid = ~np.isnan(ridge_info['ridge_freqs'])
    ax.plot(ridge_info['ridge_times'][valid], 
            ridge_info['ridge_freqs'][valid],
            'r-', linewidth=2, label='Extracted Ridge')
    
    # Linear fit
    fit_freqs = ridge_info['slope'] * time_axis + ridge_info['intercept']
    ax.plot(time_axis, fit_freqs, 'w--', linewidth=1, 
            label=f"Linear fit (slope={ridge_info['slope']:.1f} Hz/s)")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim([20, 500])
    
    return fig
```

### 8.4 Batch Processing

```python
def extract_all_ridges(spectrograms, freq_axis, time_axis):
    """Extract ridges from multiple spectrograms."""
    features_list = []
    failed_events = []
    
    for i, spec in enumerate(spectrograms):
        ridge_info = extract_ridge_peak_tracking(spec, freq_axis, time_axis)
        
        if ridge_info is None:
            failed_events.append(i)
            features_list.append(None)
        else:
            features = compute_ridge_features(ridge_info)
            features_list.append(features)
    
    print(f"Ridge extraction: {len(spectrograms) - len(failed_events)}/{len(spectrograms)} successful")
    if failed_events:
        print(f"Failed events: {failed_events}")
    
    return features_list
```

### 8.5 Expected Feature Values

| Feature | BBH (typical) | NS-present (typical) | Discriminative? |
|---------|---------------|---------------------|-----------------|
| chirp_slope | 50-200 Hz/s | 10-50 Hz/s | **Yes** |
| peak_frequency | 100-300 Hz | 500-1500 Hz | **Yes** |
| chirp_duration | 0.1-1.0 s | 1-30 s | **Yes** |
| frequency_span | 50-200 Hz | 200-1000 Hz | **Yes** |
| ridge_r_squared | >0.8 | >0.8 | No (quality metric) |

### 8.6 Handling Edge Cases

```python
def extract_ridge_robust(spectrogram, freq_axis, time_axis):
    """Ridge extraction with edge case handling."""
    
    # Check for empty/bad spectrogram
    if spectrogram is None or np.all(spectrogram == 0):
        return None
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(spectrogram)):
        spectrogram = np.nan_to_num(spectrogram, nan=0, posinf=0, neginf=0)
    
    # Standard extraction
    try:
        ridge_info = extract_ridge_peak_tracking(spectrogram, freq_axis, time_axis)
    except Exception as e:
        print(f"Ridge extraction failed: {e}")
        return None
    
    # Validate results
    if ridge_info is not None:
        if ridge_info['slope'] > 1000 or ridge_info['slope'] < -100:
            # Unreasonable slope - likely noise
            return None
        if ridge_info['duration'] < 0.01:
            # Too short - likely noise spike
            return None
    
    return ridge_info
```

---

## 9. References

- [Chirp Time-Frequency Analysis](https://arxiv.org/abs/gr-qc/9402014): GW chirp physics
- [Ridge Extraction Methods](https://ieeexplore.ieee.org/document/7178258): Survey of techniques
- [Peak Tracking for GW](https://arxiv.org/abs/1511.04398): Prior GW applications
- [Scipy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html): Implementation tools

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
