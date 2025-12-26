# ADR-003 — Event-Centered Data Segments

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-003 |
| **Title** | Event-Centered Data Segments Rather Than Full-Run Continuous Data |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Gravitational-wave observatories generate continuous strain data throughout their observing runs. Each observing run spans months of operation:
- O1: ~4 months of data
- O2: ~9 months of data  
- O3a + O3b: ~11 months of data
- O4a: ~8 months of data

At 16 kHz sampling (or 4 kHz), this translates to enormous data volumes:
- 1 hour of 16 kHz data ≈ 230 MB per detector
- 1 month ≈ 165 GB per detector
- Full O1-O4a ≈ 5-10 TB total (both detectors, all runs)

However, actual gravitational-wave events are extremely rare. Across O1-O3, only ~90 confident events were detected in ~2 years of cumulative observation time. Each event lasts only seconds (for BBH) to minutes (for BNS). The vast majority of collected data contains only noise.

Two fundamentally different approaches exist for acquiring data:
1. **Full-run continuous data**: Download all strain data for entire observing runs
2. **Event-centered segments**: Download only short windows around known events

This decision has major implications for storage, processing time, and project scope.

---

## 2. Problem Statement

We need to determine the data acquisition strategy that optimally balances:
1. Scientific validity for the classification task
2. Storage and bandwidth requirements
3. Processing complexity and time
4. Alignment with project goals (rapid event classification)

**Key Question**: Should we download complete continuous strain data for all observing runs, or only short segments centered on cataloged events?

---

## 3. Decision Drivers

1. **Storage Constraints**: Local storage is limited; cloud storage incurs costs. Multi-terabyte datasets are impractical for a science fair project.

2. **Download Time**: Full observing runs require days-weeks of download time even with fast connections. Event segments can be downloaded in minutes-hours.

3. **Task Alignment**: The project focuses on "rapid classification after trigger"—we're classifying *known events*, not detecting new ones. We don't need continuous data for detection.

4. **Labeling Complexity**: With full continuous data, we'd need to handle the segmentation problem (where do events start/end?). Event-centered segments have clear boundaries.

5. **Reproducibility**: Smaller, well-defined datasets are easier for others to recreate and verify.

6. **Processing Efficiency**: Spectrogram computation on TB-scale data requires sophisticated chunking, parallelization, and resource management.

7. **Noise Characterization**: Some analysis (e.g., background estimation) benefits from continuous data, but this is not required for our classification task.

---

## 4. Considered Options

### Option A: Event-Centered Segments (±16-64 seconds around merger)

**Description**: For each cataloged event, download only a short time window (e.g., 64 seconds total) centered on the merger time. Store these segments as individual files.

**Pros**:
- Minimal storage: ~90 events × 64s × 4kHz × 4 bytes ≈ 100 MB (vs TB for full runs)
- Fast download: Minutes to hours for entire dataset
- Clean labeling: Each file corresponds to exactly one event with known class
- Matches use case: "Rapid classification after trigger" operates on short segments
- Simple data management: One file per event
- Easy reproducibility: Small, well-defined dataset

**Cons**:
- Cannot detect new events (only classify known ones)
- No access to pre-event background for PSD estimation (must use segment edges)
- Cannot study events not in GWTC catalogs
- Window size choice affects what's captured (too short may clip signal)

### Option B: Full Continuous Strain Data

**Description**: Download complete strain data for all of O1, O2, O3, and O4a. Process locally to extract events.

**Pros**:
- Complete data access for any analysis
- Can study background noise statistics
- Can search for sub-threshold events
- Could extend project to event detection
- PSD estimation from nearby off-source data

**Cons**:
- Massive storage: 5-10 TB across all runs and detectors
- Very long download time: Days to weeks
- Complex data management: Chunking, indexing, caching required
- Processing overhead: Must segment continuous data into event windows anyway
- Overkill for classification task
- Difficult for others to reproduce

### Option C: Hybrid Approach

**Description**: Download event-centered segments for labeled events, plus representative background noise segments from each run for noise characterization.

**Pros**:
- Manageable storage (still GB scale, not TB)
- Includes noise samples for injection pipeline
- Better PSD estimation
- Some flexibility for noise analysis

**Cons**:
- More complex than pure event-centered
- Must decide how much background to include
- Additional download and management overhead

---

## 5. Decision Outcome

**Chosen Option**: Option A — Event-Centered Segments (with elements of Option C for synthetic injection)

**Rationale**:

The project's core question is: *"Can spectrogram features classify event type?"* This is a **classification** problem, not a **detection** problem. We're asking "given that an event occurred, what type is it?"—not "did an event occur?"

Key arguments for event-centered segments:

1. **Task Match**: The "rapid spectrogram after trigger" framing explicitly assumes a trigger has already occurred. The input to our pipeline is an event time, and the output is a classification. We don't need continuous data because we're not triggering.

2. **Practical Feasibility**: Downloading and processing 10 TB of data is not practical for a 10-week science fair project. Event-centered segments can be acquired in an afternoon.

3. **Clear Scope**: This decision explicitly scopes the project to classification rather than detection. This is appropriate—detection is a mature, well-solved problem with dedicated pipelines (GstLAL, PyCBC Live, etc.).

4. **Resource Efficiency**: Why download 99.9999% noise when we only need the 0.0001% containing events?

5. **Reproducibility**: A 100 MB dataset is easily shareable and verifiable. A 10 TB dataset requires special infrastructure.

**Exception for Synthetic Injection**: When generating synthetic events with PyCBC (Phase 2), we need noise segments to inject signals into. We'll download representative noise segments (~100 hours total) for this purpose, but this is still far smaller than full continuous data.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Fast iteration**: Can download entire dataset in <1 hour, enabling quick experimentation
- **Minimal storage**: Dataset fits easily on laptop/Colab
- **Clean data structure**: One event = one file = one label
- **Focused scope**: Project clearly about classification, not detection
- **Easy sharing**: Can include raw data in project repository or supplement
- **Reduced complexity**: No need for sophisticated data chunking or streaming

### 6.2 Negative Consequences

- **No detection capability**: Cannot extend project to event detection without major changes
- **Fixed event list**: Limited to events in GWTC catalogs; cannot discover new events
- **PSD estimation constraints**: Must estimate PSD from segment edges or use pre-computed PSDs
- **Window edge effects**: Signals near window boundaries may be clipped
- **Dependent on catalog accuracy**: If GWTC merger time is wrong, signal may be off-center

### 6.3 Neutral Consequences

- Must choose window size carefully (too short = clipped signals, too long = unnecessary data)
- Classification ≠ detection is a limitation but also a clear scope statement
- Background segments for injection are a separate download task

---

## 7. Validation

**Success Criteria**:
- All ~90 O1-O3 events successfully downloaded
- All ~80+ O4a events successfully downloaded
- Total storage <5 GB (including spectrograms)
- Each segment contains visible signal in spectrogram
- Merger time is approximately centered in each segment

**Review Date**: End of Week 1

**Reversal Trigger**:
- GWTC merger times are systematically offset, causing signals to be clipped
- Background noise characterization proves essential for feature extraction
- Project scope expands to include detection (would require new ADR)

---

## 8. Implementation Notes

### Window Size Recommendations

| Event Type | Characteristic Duration | Recommended Window |
|------------|------------------------|-------------------|
| BBH (high mass) | 0.1-1 seconds in-band | ±16 seconds (32s total) |
| BBH (low mass) | 1-5 seconds in-band | ±32 seconds (64s total) |
| NSBH | 5-30 seconds in-band | ±32 seconds (64s total) |
| BNS | 30-100+ seconds in-band | ±64 seconds (128s total) |

**Default Choice**: ±32 seconds (64 seconds total) captures most BBH and NSBH signals with margin. BNS may require longer windows.

### Data Download Code Pattern
```python
from gwpy.timeseries import TimeSeries

def download_event_segment(event_name, gps_time, detector='L1', 
                           window=32, sample_rate=4096):
    """Download strain data centered on event."""
    start = gps_time - window
    end = gps_time + window
    
    strain = TimeSeries.fetch_open_data(
        detector, 
        start, 
        end,
        sample_rate=sample_rate,
        cache=True
    )
    return strain
```

### File Naming Convention
```
data/raw/{run}/{event_name}_{detector}.hdf5
```
Example: `data/raw/O2/GW170817_L1.hdf5`

### Noise Segments for Injection (Phase 2)
- Download ~100 hours of confirmed "quiet" time (no events, good data quality)
- Distribute across O1, O2, O3 for noise variety
- Use GWOSC data quality flags to select segments
- Store separately in `data/noise/`

### Segment Metadata
Each downloaded segment should record:
- Event name and GPS time
- Detector(s)
- Window size used
- Sample rate
- Download timestamp
- GWOSC data version

---

## 9. References

- [GWpy TimeSeries.fetch_open_data](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.fetch_open_data): API documentation
- [GWOSC Data Products](https://gwosc.org/data/): Available strain data formats
- [GWTC Event Durations](https://arxiv.org/abs/2111.03606): Characteristic signal durations by source type
- [GW Event Database](https://gwosc.org/eventapi/): API for event metadata including GPS times
- [LIGO Data Quality](https://gwosc.org/detector_status/): Data quality flags and segments

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
