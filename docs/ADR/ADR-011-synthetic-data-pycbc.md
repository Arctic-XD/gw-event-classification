# ADR-011 — Synthetic Data via PyCBC

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-011 |
| **Title** | Synthetic Data Augmentation via PyCBC Injection |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

The fundamental challenge for GW event classification is extreme data scarcity:
- O1-O3: ~85 BBH events, ~6 NS-present events
- Total: ~91 labeled training samples

This is far below typical ML requirements:
- Simple models (logistic regression): 100+ samples
- Random Forest/XGBoost: 500+ samples for robust performance
- Deep CNNs: 10,000+ samples typical

With only 91 samples, we face:
- Overfitting to specific event characteristics
- Unreliable generalization estimates
- Inability to train deep models
- Statistical fragility in minority class (6 NS-present)

**Synthetic data injection** offers a solution: generate realistic gravitational-wave signals using waveform models, inject them into real detector noise, and label them with known ground truth. This is a standard technique in GW data analysis for sensitivity estimation and pipeline validation.

---

## 2. Problem Statement

We need to dramatically increase training data volume while maintaining physical realism:

1. How can we generate thousands of labeled training samples?
2. How do we ensure synthetic signals are realistic?
3. What parameter distributions should we use?
4. How do we validate synthetic-to-real transfer?
5. What tools are available for waveform generation?

**Key Question**: Should we augment our training set with synthetic injections, and if so, how?

---

## 3. Decision Drivers

1. **Data Quantity**: Need 10x-100x more training samples for deep learning

2. **Physical Realism**: Synthetic signals must be indistinguishable from real detections

3. **Parameter Coverage**: Want diverse parameter space (masses, spins, distances)

4. **Class Balance**: Can generate equal numbers of BBH and NS-present

5. **Validation**: Must verify synthetic data enables real-world generalization

6. **Tool Availability**: Need mature, validated waveform generation software

7. **Computational Cost**: Generation should be tractable on available hardware

---

## 4. Considered Options

### Option A: PyCBC Synthetic Injection Pipeline

**Description**: Use PyCBC library to generate CBC waveforms with known parameters, inject into real LIGO noise from GWOSC.

**Pros**:
- Industry-standard GW waveform library
- Same approximants used by LVK for analysis
- Inject into real noise (captures glitches, non-Gaussian features)
- Complete control over signal parameters
- Validated against LVK infrastructure
- Scalable to 10,000+ samples

**Cons**:
- Waveform models are approximations to true GR
- Does not capture unknown systematic effects
- Requires noise segment curation
- Computational cost for large samples

### Option B: Traditional Data Augmentation (No New Signals)

**Description**: Augment existing 91 real events with transformations: time shifts, noise injection, frequency masking.

**Pros**:
- Only uses real data
- No waveform model dependency
- Simple to implement

**Cons**:
- Limited diversity—still 91 base events
- Cannot balance classes (still 85:6 ratio)
- Transformations may create unrealistic artifacts
- Insufficient for deep learning

### Option C: Generative Models (GANs/VAEs)

**Description**: Train generative neural network on real spectrograms, generate synthetic examples.

**Pros**:
- Learns data distribution automatically
- Could capture unknown features

**Cons**:
- Cannot train GAN on 91 samples—need more data to start
- No physical ground truth for generated samples
- Mode collapse risk
- Chicken-and-egg problem

### Option D: Simulation Only (No Real Noise)

**Description**: Generate waveforms and add Gaussian noise based on detector PSD.

**Pros**:
- Simpler than injection pipeline
- Fully controlled noise

**Cons**:
- Misses non-Gaussian noise features (glitches)
- Real LIGO noise is NOT Gaussian
- Models trained on Gaussian noise fail on real data
- Unrealistic

---

## 5. Decision Outcome

**Chosen Option**: Option A — PyCBC Synthetic Injection Pipeline

**Rationale**:

PyCBC injection is the established, validated method for generating realistic GW training data:

1. **Physical Fidelity**: 
   - PyCBC implements state-of-the-art waveform approximants (IMRPhenomD, TaylorF2, IMRPhenomNSBH)
   - These models are calibrated against numerical relativity
   - Used by LVK for official analyses

2. **Real Noise Critical**:
   - LIGO noise is non-stationary and non-Gaussian
   - Contains glitches, lines, and environmental transients
   - Injecting into real noise teaches model to distinguish signal from noise artifacts
   - Gaussian-noise-only training produces models that fail on real data

3. **Parameter Control**:
   - Can sample from astrophysically motivated priors
   - Can enforce class balance (50% BBH, 50% NS-present)
   - Can target specific SNR ranges (8-30 for detectable events)
   - Know true parameters for physics-informed loss

4. **Scalability**:
   - Single waveform generation: <1 second
   - 10,000 injections: few hours on laptop
   - Parameters fully parallelizable

5. **Validation Strategy**:
   - Train on synthetic → Validate on real O1-O3 → Test on real O4a
   - If synthetic-trained model works on real data, synthetic is validated
   - This "synthetic-to-real transfer" is a key project contribution

---

## 6. Consequences

### 6.1 Positive Consequences

- **Massive data expansion**: 10,000+ training samples (100x increase)
- **Class balance**: Equal BBH and NS-present samples
- **Parameter diversity**: Cover full astrophysically relevant parameter space
- **Deep learning enabled**: Sufficient data for CNN training
- **Ground truth available**: Know true parameters for physics-informed learning
- **Noise robustness**: Model learns on realistic non-Gaussian noise

### 6.2 Negative Consequences

- **Waveform model dependency**: Signals are approximations
- **Unknown systematics**: May miss effects not in waveform models
- **Domain gap risk**: Synthetic→real transfer may not be perfect
- **Computational cost**: Generating 10K+ samples takes time
- **Noise curation required**: Must select clean noise segments

### 6.3 Neutral Consequences

- Adds PyCBC as project dependency
- Must document injection procedure thoroughly
- Creates separate "synthetic" dataset alongside "real"
- Validation specifically tests synthetic-to-real transfer

---

## 7. Validation

**Success Criteria**:
- Generate 10,000+ synthetic events (5,000 BBH + 5,000 NS-present)
- Model trained on synthetic achieves >75% accuracy on real O1-O3 events
- Synthetic spectrograms visually match real event spectrograms
- Parameter distributions match astrophysical expectations

**Review Date**: Week 5 (after synthetic generation)

**Reversal Trigger**:
- Synthetic-to-real transfer fails (<60% accuracy on real data)
- Generated waveforms have obvious artifacts
- Injection procedure introduces systematic biases

---

## 8. Implementation Notes

### 8.1 Waveform Approximants

| Source Type | Approximant | Mass Range | Spin Range | Notes |
|-------------|-------------|------------|------------|-------|
| BBH | IMRPhenomD | 5-100 M☉ | χ: -0.99 to 0.99 | Full IMR waveform |
| BNS | TaylorF2 | 1-2.5 M☉ | χ: 0 to 0.05 | Inspiral-only (tidal) |
| NSBH | IMRPhenomNSBH | BH: 5-50, NS: 1-2.5 M☉ | BH: high, NS: low | Includes disruption |

### 8.2 Parameter Sampling

```python
import numpy as np
from pycbc.distributions import uniform, cosine

def sample_bbh_parameters(n_samples, seed=42):
    """Sample BBH parameters from astrophysical priors."""
    np.random.seed(seed)
    
    params = []
    for i in range(n_samples):
        # Component masses (uniform in component mass)
        m1 = np.random.uniform(10, 80)  # Solar masses
        m2 = np.random.uniform(5, m1)   # m2 ≤ m1 by convention
        
        # Aligned spins (uniform)
        spin1z = np.random.uniform(-0.9, 0.9)
        spin2z = np.random.uniform(-0.9, 0.9)
        
        # Distance (uniform in comoving volume → p(d) ∝ d²)
        d_max = 2000  # Mpc
        distance = d_max * np.random.uniform(0, 1)**(1/3)
        
        # Inclination (uniform in cos)
        inclination = np.arccos(np.random.uniform(-1, 1))
        
        params.append({
            'mass1': m1, 'mass2': m2,
            'spin1z': spin1z, 'spin2z': spin2z,
            'distance': distance, 'inclination': inclination
        })
    
    return params
```

### 8.3 Waveform Generation

```python
from pycbc.waveform import get_td_waveform

def generate_waveform(params, sample_rate=4096, approximant='IMRPhenomD'):
    """Generate time-domain CBC waveform."""
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=params['mass1'],
        mass2=params['mass2'],
        spin1z=params['spin1z'],
        spin2z=params['spin2z'],
        distance=params['distance'],
        inclination=params['inclination'],
        delta_t=1.0/sample_rate,
        f_lower=20.0
    )
    return hp, hc
```

### 8.4 Injection into Noise

```python
from gwpy.timeseries import TimeSeries
import numpy as np

def inject_signal(noise_segment, waveform, injection_time, snr_target=None):
    """
    Inject waveform into noise segment.
    
    Parameters
    ----------
    noise_segment : TimeSeries
        Real LIGO noise from GWOSC
    waveform : array
        Generated h+ waveform
    injection_time : float
        GPS time for merger within segment
    snr_target : float, optional
        If provided, scale waveform to achieve target SNR
    """
    # Align waveform to injection time
    # Waveform ends at t=0 (merger), so shift appropriately
    
    # Create injection array
    injection = np.zeros(len(noise_segment))
    
    # Find index for merger time
    merger_idx = int((injection_time - noise_segment.t0.value) * noise_segment.sample_rate.value)
    
    # Place waveform (ends at merger)
    start_idx = merger_idx - len(waveform)
    if start_idx >= 0 and merger_idx < len(injection):
        injection[start_idx:merger_idx] = waveform
    
    # Scale to target SNR if specified
    if snr_target:
        current_snr = compute_snr(injection, noise_segment)
        injection *= snr_target / current_snr
    
    # Add to noise
    injected_data = noise_segment.value + injection
    
    return TimeSeries(injected_data, 
                      sample_rate=noise_segment.sample_rate,
                      t0=noise_segment.t0)
```

### 8.5 SNR Calculation

```python
from pycbc.filter import matched_filter, sigma
from pycbc.psd import interpolate, inverse_spectrum_truncation

def compute_optimal_snr(waveform, psd, sample_rate):
    """Compute optimal matched-filter SNR."""
    # Compute sigma (normalization)
    sig = sigma(waveform, psd, 
                low_frequency_cutoff=20.0,
                high_frequency_cutoff=1024.0)
    return sig
```

### 8.6 Target Dataset Composition

| Class | Source | Count | Total |
|-------|--------|-------|-------|
| BBH | Real O1-O3 | 85 | |
| BBH | Synthetic | 5,000 | **5,085** |
| NS-present | Real O1-O3 | 6 | |
| NS-present (BNS) | Synthetic | 2,500 | |
| NS-present (NSBH) | Synthetic | 2,500 | **5,006** |
| **Total** | | | **10,091** |

### 8.7 Noise Segment Selection

```python
def select_noise_segments(run, total_duration=100*3600, segment_length=128):
    """
    Select quiet noise segments from observing run.
    
    Criteria:
    - Data quality flag = 'CBC_CAT1' (science quality)
    - No catalog events within ±300 seconds
    - Continuous data available
    """
    # Use GWOSC segment queries
    # Return list of (start_time, end_time) for valid noise
    pass
```

---

## 9. References

- [PyCBC Documentation](https://pycbc.org/): Official library docs
- [PyCBC Waveforms](https://pycbc.org/pycbc/latest/html/waveform.html): Waveform generation
- [LVK Injection Studies](https://arxiv.org/abs/2010.14527): Standard injection methods
- [IMRPhenomD](https://arxiv.org/abs/1508.07250): BBH waveform model
- [TaylorF2](https://arxiv.org/abs/0907.0700): BNS inspiral model
- [Synthetic-to-Real Transfer](https://arxiv.org/abs/1812.05687): Domain adaptation in ML

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
