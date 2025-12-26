# ADR-007 — GWpy for Data Access

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-007 |
| **Title** | Access Method: GWpy `fetch_open_data` for Automation |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Downloading gravitational-wave data from GWOSC can be accomplished through multiple methods:

1. **Manual web download**: Click through GWOSC website to download individual files
2. **Direct URL construction**: Build URLs programmatically, use wget/requests
3. **GWpy library**: Use dedicated Python API for GW data access
4. **GWOSC Python client**: Alternative official Python interface
5. **PyCBC data retrieval**: PyCBC's built-in data fetching functions

For a project involving ~100+ events across multiple observing runs, automation is essential. The choice of data access method affects reproducibility, error handling, and integration with downstream analysis.

**GWpy** is an open-source Python package developed and maintained by the LIGO Scientific Collaboration specifically for gravitational-wave data analysis. It provides high-level interfaces for data retrieval, processing, and visualization.

---

## 2. Problem Statement

We need to select a data access method that:
1. Can download strain data for ~100+ events programmatically
2. Handles GWOSC API interactions reliably
3. Integrates cleanly with downstream analysis (spectrograms, filtering)
4. Provides good error handling for missing or corrupted data
5. Is well-documented and maintainable
6. Produces reproducible results

**Key Question**: What is the optimal method for programmatically accessing GWOSC strain data for our classification pipeline?

---

## 3. Decision Drivers

1. **Automation**: Must handle 100+ downloads without manual intervention

2. **Error Handling**: Network failures, missing segments, and data gaps must be handled gracefully

3. **Integration**: Data should load directly into analysis-ready format (numpy arrays with time metadata)

4. **Documentation**: Well-documented library reduces implementation errors

5. **Community Standard**: Using community-standard tools improves reproducibility and credibility

6. **Maintenance**: Actively maintained libraries are preferred for long-term stability

7. **Feature Richness**: Built-in support for filtering, resampling, and spectrograms is valuable

---

## 4. Considered Options

### Option A: GWpy `TimeSeries.fetch_open_data()`

**Description**: Use GWpy's dedicated GWOSC interface to fetch strain data directly into TimeSeries objects.

**Pros**:
- Official LSC-developed library for GW data
- Direct GWOSC API integration
- Returns TimeSeries objects with time metadata
- Built-in caching to avoid re-downloads
- Excellent documentation and tutorials
- Integrated filtering, whitening, spectrogram methods
- Active maintenance by GW community
- Handles sample rate selection automatically

**Cons**:
- Additional dependency (gwpy + dependencies)
- Some features may be overkill for simple downloads
- Large install footprint

### Option B: Manual URL Construction + Requests

**Description**: Construct GWOSC URLs manually based on run/time, download with requests/wget.

**Pros**:
- Minimal dependencies
- Full control over download process
- Understand exactly what's happening

**Cons**:
- Must reverse-engineer URL structure
- No built-in error handling
- Must parse HDF5/GWF files manually
- No time metadata handling
- Reinventing the wheel
- Fragile to GWOSC URL changes

### Option C: GWOSC Python Client

**Description**: Use gwosc package (official GWOSC Python client) for URL lookups, combined with separate download.

**Pros**:
- Official GWOSC tool for URL discovery
- API for event catalogs and segment lists

**Cons**:
- Only provides URLs, not data loading
- Must combine with another tool for actual download
- Less featured than GWpy
- Limited filtering/analysis support

### Option D: PyCBC Data Retrieval

**Description**: Use PyCBC's data fetching capabilities (`pycbc.frame`, `pycbc.catalog`).

**Pros**:
- Powerful library for CBC analysis
- Integrated waveform generation
- Well-tested in production

**Cons**:
- Heavier dependency than needed for just downloading
- More complex API
- Less intuitive for simple fetch tasks
- Overkill for data acquisition alone

---

## 5. Decision Outcome

**Chosen Option**: Option A — GWpy `TimeSeries.fetch_open_data()`

**Rationale**:

GWpy is the clear choice for gravitational-wave data access:

1. **Purpose-Built**: GWpy was specifically designed for GW data analysis. `fetch_open_data()` is the standard interface to GWOSC used by researchers worldwide.

2. **One-Line Downloads**: Simple, readable code:
   ```python
   strain = TimeSeries.fetch_open_data('L1', t0-32, t0+32, sample_rate=4096)
   ```
   vs. dozens of lines for manual URL construction + file parsing.

3. **Integrated Processing**: The returned `TimeSeries` object has methods for:
   - Filtering: `strain.bandpass(20, 500)`
   - Whitening: `strain.whiten()`
   - Spectrograms: `strain.spectrogram()`
   - Plotting: `strain.plot()`
   
   This eliminates separate library dependencies for common operations.

4. **Built-in Caching**: GWpy caches downloaded data locally, avoiding redundant network requests during development.

5. **Error Handling**: GWpy handles missing segments, network errors, and data gaps with informative exceptions.

6. **Community Standard**: This is what LIGO scientists use. Using GWpy signals that we follow best practices.

7. **Documentation Excellence**: GWpy has extensive tutorials, API documentation, and examples specifically for GWOSC data access.

**Note**: We will still use PyCBC for *waveform generation* (Phase 2 synthetic data), but GWpy handles *data acquisition*.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Rapid development**: Single function call downloads data
- **Reproducibility**: Standard interface produces consistent results
- **Integration**: TimeSeries integrates with filtering, spectrograms
- **Community alignment**: Using community-standard tools
- **Error resilience**: Built-in handling of common failure modes
- **Caching**: Avoid re-downloading during development
- **Documentation**: Easy to learn and use

### 6.2 Negative Consequences

- **Dependency**: GWpy has substantial dependencies (numpy, scipy, astropy, h5py, etc.)
- **Black box**: Less visibility into download mechanics
- **GWOSC coupling**: Tied to GWOSC's API structure (but this is our data source anyway)

### 6.3 Neutral Consequences

- Must install gwpy in project environment
- GWpy version may affect reproducibility (pin version)
- Some GWpy features won't be used (we only need data fetch + basic processing)

---

## 7. Validation

**Success Criteria**:
- Successfully download all ~90 O1-O3 events
- Handle missing/unavailable segments gracefully (log, skip)
- Data loads into analysis pipeline without format issues
- Download time <2 hours for full dataset
- Cached downloads work correctly on second run

**Review Date**: Week 1 (data acquisition phase)

**Reversal Trigger**:
- GWpy API changes break our download scripts
- GWOSC changes URL structure incompatibly
- Performance is unacceptably slow
- Critical bugs in GWpy without fixes available

---

## 8. Implementation Notes

### Installation
```bash
pip install gwpy
# or in requirements.txt:
# gwpy>=3.0
```

### Basic Usage Pattern
```python
from gwpy.timeseries import TimeSeries
from gwpy.time import to_gps

def download_event(event_name, gps_time, detector='L1', 
                   window=32, sample_rate=4096, output_dir='data/raw'):
    """
    Download strain data for a single event.
    
    Parameters
    ----------
    event_name : str
        Event identifier (e.g., 'GW150914')
    gps_time : float
        GPS time of merger
    detector : str
        Detector name ('L1', 'H1', 'V1')
    window : float
        Half-width of time window in seconds
    sample_rate : int
        Desired sample rate in Hz
    output_dir : str
        Directory to save downloaded data
    
    Returns
    -------
    TimeSeries or None
        Downloaded strain data, or None if failed
    """
    start = gps_time - window
    end = gps_time + window
    
    try:
        strain = TimeSeries.fetch_open_data(
            detector,
            start,
            end,
            sample_rate=sample_rate,
            cache=True,
            verbose=True
        )
        
        # Save to file
        output_path = f"{output_dir}/{event_name}_{detector}.hdf5"
        strain.write(output_path, overwrite=True)
        
        return strain
        
    except Exception as e:
        print(f"Failed to download {event_name} from {detector}: {e}")
        return None
```

### Batch Download Script
```python
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def download_all_events(manifest_path, output_dir, max_workers=4):
    """Download all events in manifest."""
    manifest = pd.read_csv(manifest_path)
    
    results = []
    for _, row in manifest.iterrows():
        result = download_event(
            row['event_name'],
            row['gps_time'],
            row['detector'],
            output_dir=output_dir
        )
        results.append({
            'event': row['event_name'],
            'success': result is not None
        })
    
    return pd.DataFrame(results)
```

### Error Handling
```python
from gwpy.timeseries import TimeSeries
from requests.exceptions import HTTPError

try:
    strain = TimeSeries.fetch_open_data('L1', start, end)
except HTTPError as e:
    if e.response.status_code == 404:
        print("Data not available for this time range")
    else:
        raise
except ValueError as e:
    print(f"Invalid time range or detector: {e}")
```

### Caching Behavior
- GWpy caches downloaded files in `~/.cache/gwpy/` by default
- Set `cache=True` to enable (recommended during development)
- Set `cache=False` for production to ensure fresh data

### Version Pinning
```
# requirements.txt
gwpy>=3.0.0,<4.0.0
```

---

## 9. References

- [GWpy Documentation](https://gwpy.github.io/docs/stable/): Official documentation
- [GWpy TimeSeries.fetch_open_data](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.fetch_open_data): API reference
- [GWpy GWOSC Tutorial](https://gwpy.github.io/docs/stable/examples/timeseries/open-data.html): Example usage
- [GWpy GitHub](https://github.com/gwpy/gwpy): Source code and issues
- [GWOSC API Documentation](https://gwosc.org/apidocs/): Underlying API that GWpy uses

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
