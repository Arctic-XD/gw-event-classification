# ADR-005 — Sample Rate Selection

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-005 |
| **Title** | Sample Rate Selection: 4096 Hz Default |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

GWOSC provides gravitational-wave strain data at multiple sample rates:
- **16384 Hz (16 kHz)**: Full bandwidth, captures signals up to ~8 kHz (Nyquist)
- **4096 Hz (4 kHz)**: Reduced bandwidth, captures signals up to ~2 kHz (Nyquist)

The choice of sample rate affects:
- **File size**: 4 kHz data is 4× smaller than 16 kHz
- **Frequency coverage**: Higher sample rate captures higher frequencies
- **Processing speed**: Lower sample rate processes faster
- **Spectrogram resolution**: Affects time-frequency trade-offs

Different gravitational-wave sources have different frequency content:
- **High-mass BBH** (total mass >100 M☉): Signal peaks below 200 Hz
- **Low-mass BBH** (total mass ~20-60 M☉): Signal peaks at 200-500 Hz
- **NSBH**: Signal peaks at 500-1000 Hz depending on NS mass
- **BNS**: Signal extends to 1500-2000+ Hz before merger

The Nyquist frequency (half the sample rate) determines the maximum observable frequency.

---

## 2. Problem Statement

We need to select a sample rate for strain data acquisition that:
1. Captures all relevant gravitational-wave signal frequencies
2. Minimizes data storage and processing requirements
3. Produces spectrograms with appropriate time-frequency resolution
4. Is consistently available across all observing runs (O1-O4a)

**Key Question**: Should we use 4 kHz or 16 kHz sample rate, and under what circumstances might we need to deviate?

---

## 3. Decision Drivers

1. **BNS Signal Coverage**: BNS signals can reach 1500-2000 Hz. 4 kHz (Nyquist = 2 kHz) just barely captures the full BNS inspiral.

2. **Storage Efficiency**: We're downloading ~90+ events. 4× reduction in file size matters for practical data management.

3. **Processing Speed**: Spectrogram computation scales with data length. 4 kHz is 4× faster.

4. **Data Availability**: Both 4 kHz and 16 kHz are available for O1-O4a, but 4 kHz is the "default" GWOSC product.

5. **BBH Sufficiency**: Most BBH signals are well below 500 Hz. 4 kHz is more than adequate.

6. **Feature Extraction**: Our spectrogram features focus on the dominant chirp structure, not high-frequency content.

7. **CNN Input Size**: Fixed 224×224 image input means high frequencies would be compressed anyway.

---

## 4. Considered Options

### Option A: 4096 Hz (4 kHz) for All Events

**Description**: Use 4 kHz sample rate uniformly for all events regardless of source type.

**Pros**:
- Consistent data format across all events
- 4× smaller files than 16 kHz
- Adequate for all BBH and most NSBH signals
- Marginally captures BNS (Nyquist = 2048 Hz)
- Faster processing
- Default GWOSC data product

**Cons**:
- May miss highest-frequency BNS content (>2 kHz)
- If BNS signal extends beyond 2 kHz, will be aliased (rare)

### Option B: 16384 Hz (16 kHz) for All Events

**Description**: Use maximum sample rate uniformly for complete frequency coverage.

**Pros**:
- Full frequency coverage (Nyquist = 8192 Hz)
- No risk of missing high-frequency signals
- Maximum available fidelity

**Cons**:
- 4× larger files
- 4× longer processing time
- Most of high-frequency content is just noise
- Overkill for BBH signals
- Larger spectrograms require more memory

### Option C: Adaptive Sample Rate by Source Type

**Description**: Use 4 kHz for BBH, 16 kHz for BNS/NSBH.

**Pros**:
- Optimal per source type
- Captures full BNS spectrum
- Efficient for BBH

**Cons**:
- Requires knowing source type before download (chicken-and-egg for classification)
- Inconsistent data format
- Complicates preprocessing pipeline
- Different spectrogram dimensions

### Option D: 4 kHz Default with 16 kHz BNS Exception

**Description**: Use 4 kHz for everything initially. Only if BNS classification accuracy is poor, re-download known BNS events at 16 kHz.

**Pros**:
- Efficient baseline (4 kHz)
- Allows targeted high-resolution for rare BNS
- Data-driven decision for upgrades

**Cons**:
- Two download passes potentially needed
- Slight inconsistency if 16 kHz used selectively

---

## 5. Decision Outcome

**Chosen Option**: Option A — 4096 Hz (4 kHz) for All Events

**Rationale**:

After analyzing the frequency content of different source types, 4 kHz is sufficient for our classification task:

1. **BBH Analysis**: All BBH signals have merger frequencies well below 500 Hz. Even the lightest stellar-mass BBH (~10+10 M☉) merges below 1 kHz. 4 kHz is more than adequate.

2. **NSBH Analysis**: NSBH signals can extend to 1-1.5 kHz depending on masses. 4 kHz comfortably captures these.

3. **BNS Consideration**: BNS signals can theoretically reach 2-3 kHz. However:
   - LIGO sensitivity drops significantly above 1 kHz
   - Most SNR comes from <1 kHz frequencies
   - Only 2 confirmed BNS events exist (GW170817, GW190425)
   - Classification features focus on chirp morphology, not high-frequency ringing

4. **Practical Factors**:
   - 4 kHz is GWOSC's default offering
   - Storage reduction (4×) enables faster iteration
   - Processing speed improvement matters for development

5. **Risk Assessment**: The risk of misclassifying a BNS because we missed >2 kHz content is extremely low. The chirp structure that distinguishes BNS from BBH (longer duration, higher final frequency) is fully captured at 4 kHz.

**Exception Policy**: If BNS/NSBH recall is significantly worse than BBH recall (>20% gap), we will investigate whether 16 kHz helps and potentially re-download NS events at higher sample rate.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Storage efficiency**: ~100 MB for all events instead of ~400 MB
- **Processing speed**: 4× faster spectrogram computation
- **Consistent format**: All events use identical sample rate
- **GWpy default**: Aligns with default GWOSC data product
- **Memory efficiency**: Smaller arrays, less RAM needed
- **Simpler pipeline**: No adaptive logic needed

### 6.2 Negative Consequences

- **Theoretical BNS limitation**: Cannot analyze content >2 kHz (unlikely to matter)
- **Aliasing risk**: Any signal content >2 kHz will alias (rare, weak)
- **Future-proofing**: If project expands to post-merger oscillations, would need 16 kHz

### 6.3 Neutral Consequences

- Spectrograms naturally focus on 0-2 kHz band
- High-frequency noise above 1 kHz is excluded from analysis
- Must document sample rate choice in methodology

---

## 7. Validation

**Success Criteria**:
- All events successfully downloaded at 4 kHz
- No visible aliasing artifacts in spectrograms
- BNS/NSBH classification accuracy ≥80% (demonstrating adequate frequency coverage)
- No significant correlation between misclassification and high-frequency content

**Review Date**: Week 4 (after Phase 1 feature extraction)

**Reversal Trigger**:
- BNS recall <60% with suspected frequency coverage as cause
- Expert reviewer identifies missing high-frequency features
- Aliasing visibly corrupts spectrograms

---

## 8. Implementation Notes

### GWpy Download Specification
```python
from gwpy.timeseries import TimeSeries

strain = TimeSeries.fetch_open_data(
    'L1',
    start_time,
    end_time,
    sample_rate=4096,  # Explicitly request 4 kHz
    cache=True
)
```

### Frequency Coverage Table

| Sample Rate | Nyquist Freq | BBH Coverage | BNS Coverage |
|-------------|--------------|--------------|--------------|
| 4096 Hz | 2048 Hz | 100% | ~95% |
| 16384 Hz | 8192 Hz | 100% | 100% |

### Spectrogram Frequency Limits
```python
# Recommended frequency range for spectrograms
f_min = 20   # Hz (below seismic wall)
f_max = 1024 # Hz (above most CBC signals, below Nyquist)
```

### Data Format Verification
```python
# Verify sample rate after download
assert strain.sample_rate.value == 4096.0, "Unexpected sample rate"
```

### Storage Estimates

| Scenario | Events | Duration | Sample Rate | Size |
|----------|--------|----------|-------------|------|
| O1-O3 Events | 90 | 64s each | 4 kHz | ~90 MB |
| O1-O3 Events | 90 | 64s each | 16 kHz | ~360 MB |
| + O4a Events | 170 | 64s each | 4 kHz | ~170 MB |
| + Noise Segments | +100hr | — | 4 kHz | ~6 GB |

---

## 9. References

- [GWOSC Data Products](https://gwosc.org/data/): Available sample rates by run
- [LIGO Sensitivity Curves](https://dcc.ligo.org/LIGO-T1800044): Frequency-dependent detector sensitivity
- [Nyquist-Shannon Theorem](https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem): Sampling theory fundamentals
- [GW170817 Spectrum](https://arxiv.org/abs/1710.05832): BNS frequency evolution
- [CBC Waveform Frequencies](https://arxiv.org/abs/1601.05588): Theoretical frequency content by source type

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
