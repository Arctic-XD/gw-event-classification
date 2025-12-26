# ADR-001 — Data Source: GWOSC as Source of Truth

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-001 |
| **Title** | Data Source: GWOSC as Source of Truth |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

This project aims to develop a rapid classification system for gravitational-wave (GW) events, distinguishing between binary black hole (BBH) mergers and events involving neutron stars (BNS/NSBH). To train and validate machine learning models for this task, we require access to authentic gravitational-wave strain data from LIGO/Virgo/KAGRA detectors, along with reliable event catalogs that provide ground-truth labels.

Gravitational-wave data presents unique challenges compared to typical machine learning datasets:
- The raw strain data is collected by extremely sensitive interferometers
- Data quality varies across different observing runs and time periods
- Events are rare (approximately 90 confident detections across O1-O3)
- Labels come from sophisticated parameter estimation pipelines run by the LVK collaboration
- Reproducibility requires access to the exact same data versions used in publications

Multiple potential data sources exist, including direct collaboration data (restricted access), processed data products, and public open-data releases. The choice of data source fundamentally impacts reproducibility, accessibility, and scientific credibility of the project.

---

## 2. Problem Statement

We need to select a primary data source for gravitational-wave strain data and event metadata that satisfies the following requirements:
1. Contains authentic LIGO/Virgo strain data at sufficient quality for spectrogram analysis
2. Provides reliable event catalogs with source classifications (BBH/BNS/NSBH)
3. Is accessible without requiring LVK collaboration membership
4. Supports reproducible science with versioned, citable data products
5. Covers multiple observing runs (O1, O2, O3, O4a) for training and testing

**Key Question**: Which data source should serve as the single source of truth for all datasets and metadata in this project?

---

## 3. Decision Drivers

1. **Reproducibility**: Science fair projects require that results can be independently verified. The data source must provide stable, versioned datasets that others can access.

2. **Accessibility**: As an independent research project (not affiliated with LVK collaboration), we can only use publicly available data. Restricted collaboration data is not an option.

3. **Authenticity**: The data must be the official, calibrated strain data used in published scientific analyses, not simulations or approximations.

4. **Documentation**: Comprehensive documentation is essential for understanding data formats, quality flags, and proper usage.

5. **Citability**: Published research requires proper citations. The data source must provide DOIs or equivalent permanent identifiers.

6. **Coverage**: Must include events from O1-O3 (training) and O4a (testing) with consistent format and quality.

7. **API Access**: Programmatic data access is strongly preferred over manual downloads for ~100+ events.

---

## 4. Considered Options

### Option A: GWOSC (Gravitational-Wave Open Science Center)

**Description**: The official open-data platform operated by the LIGO-Virgo-KAGRA collaboration. Provides calibrated strain data, event catalogs (GWTC series), documentation, and educational resources. Data is released ~18 months after collection following proprietary analysis.

**Pros**:
- Official LVK-sanctioned data release (highest credibility)
- Comprehensive GWTC catalogs with vetted classifications
- Multiple data formats (HDF5, GWF) and sample rates (4 kHz, 16 kHz)
- Well-documented with tutorials and example code
- Free API access via GWpy integration
- DOI-citable data releases
- Active maintenance and user support
- Covers O1, O2, O3a, O3b, and O4a

**Cons**:
- Data release lags real-time by ~18 months
- Limited to what collaboration chooses to publish
- Some data segments may have quality issues (documented)

### Option B: Direct LIGO Data (LIGO Data Grid)

**Description**: Access to full LIGO data archives through collaboration infrastructure, requiring authentication and membership.

**Pros**:
- Complete data access including proprietary periods
- Real-time or near-real-time data availability
- Full auxiliary channel access

**Cons**:
- Requires LVK collaboration membership (not accessible)
- Complex authentication and access procedures
- Not suitable for reproducible public science
- Would exclude this project entirely

### Option C: Published Paper Data Products

**Description**: Download data products directly from published papers, supplementary materials, or author-provided repositories.

**Pros**:
- Pre-processed for specific analyses
- Directly tied to published results

**Cons**:
- Inconsistent formats across papers
- Often incomplete (only selected events)
- No standardized API
- May not include raw strain
- Reproducibility depends on individual authors

### Option D: Simulated Data Only

**Description**: Use entirely synthetic gravitational-wave data generated with waveform models (e.g., PyCBC, LALSuite).

**Pros**:
- Complete control over signal parameters
- Unlimited data generation
- No access restrictions

**Cons**:
- Missing real detector noise characteristics
- Cannot validate on actual detected events
- Reduced scientific credibility
- Not representative of real-world performance

---

## 5. Decision Outcome

**Chosen Option**: Option A — GWOSC (Gravitational-Wave Open Science Center)

**Rationale**:

GWOSC is the only option that satisfies all decision drivers:

1. **Reproducibility**: GWOSC provides versioned data releases with DOIs (e.g., GWTC-3 data release). Any researcher worldwide can access identical datasets.

2. **Accessibility**: Completely open access, no authentication required, designed specifically for public science and education.

3. **Authenticity**: This IS the official LVK data. GWOSC releases are the same calibrated strain used in collaboration papers.

4. **Documentation**: Extensive documentation including data format specifications, quality flags, event parameters, and usage tutorials.

5. **Citability**: Each data release has a DOI. GWTC catalogs are published in peer-reviewed journals.

6. **Coverage**: GWOSC covers O1 (2015-2016), O2 (2016-2017), O3a (2019), O3b (2019-2020), and O4a (2023-2024).

7. **API Access**: Native integration with GWpy allows programmatic data fetching via `TimeSeries.fetch_open_data()`.

Option B is not accessible. Options C and D would compromise scientific credibility and reproducibility. GWOSC is the clear choice for any independent GW research project.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Maximum credibility**: Using official LVK data releases provides the strongest foundation for scientific claims
- **Full reproducibility**: Any reviewer or judge can independently verify results using identical data
- **Community alignment**: Following standard practices used by GW research community
- **Documentation support**: Extensive GWOSC documentation reduces implementation errors
- **Future compatibility**: GWOSC will continue releasing data from future observing runs
- **Educational value**: Demonstrates proper open-science practices

### 6.2 Negative Consequences

- **Data lag**: O4 data release is limited to O4a; cannot access latest O4b/O4c events until public release (~2025-2026)
- **Fixed scope**: Cannot analyze events not included in GWTC catalogs (e.g., sub-threshold triggers)
- **GWOSC dependency**: Project relies on GWOSC infrastructure availability
- **Format constraints**: Must work within GWOSC's provided formats and conventions

### 6.3 Neutral Consequences

- Must explicitly define "O4" as "O4a" when discussing test set, since only O4a is publicly available
- Event labels are limited to GWTC classifications; cannot create custom labels without re-running parameter estimation

---

## 7. Validation

**Success Criteria**:
- All training data (O1-O3 events) successfully downloaded and verified
- All test data (O4a events) successfully downloaded and verified
- Data formats match GWOSC documentation specifications
- Event labels match GWTC catalog classifications
- Analysis pipeline runs reproducibly on downloaded data

**Review Date**: End of Week 1 (after initial data download)

**Reversal Trigger**:
- GWOSC infrastructure becomes unavailable for extended period
- Critical events are identified that are not in GWOSC releases
- Data quality issues make GWOSC data unsuitable for spectrogram analysis

---

## 8. Implementation Notes

1. **Primary data interface**: Use GWpy's `TimeSeries.fetch_open_data()` for all data downloads
2. **Event catalog**: Download GWTC-1, GWTC-2.1, and GWTC-3 event lists from GWOSC event portal
3. **O4a events**: Use O4a public event list when available (check GWOSC announcements)
4. **Data caching**: Store downloaded data locally to avoid repeated API calls
5. **Version tracking**: Record GWTC catalog versions used in manifest files
6. **Quality flags**: Respect data quality segments; exclude times with known issues

**Code Example**:
```python
from gwpy.timeseries import TimeSeries

# Fetch strain data for a specific event
strain = TimeSeries.fetch_open_data(
    'L1',  # Detector
    start_time,  # GPS time
    end_time,
    sample_rate=4096,
    cache=True
)
```

---

## 9. References

- [GWOSC Main Site](https://gwosc.org/): Primary portal for gravitational-wave open science data
- [GWOSC Event Portal](https://gwosc.org/eventapi/): API access to GWTC event catalogs
- [GWOSC O4 Data Release](https://gwosc.org/O4/): O4a data availability and documentation
- [GWpy Documentation](https://gwpy.github.io/): Python library for GW data analysis
- [GWTC-3 Paper](https://arxiv.org/abs/2111.03606): Third Gravitational-Wave Transient Catalog
- [GWOSC Data Use Agreement](https://gwosc.org/terms/): Terms of use for open data

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
