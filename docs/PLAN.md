# Project Plan: GW Event Classification

## Overview

**Goal:** Classify gravitational-wave events (BBH vs NS-present) using spectrogram features + ML/DL  
**Duration:** 10 weeks  
**Phases:** Setup → Phase 1 (Baseline) → Phase 2 (Deep Learning) → Evaluation → Write-up

---

# Milestones

| ID | Milestone | Target Week | Description |
|----|-----------|-------------|-------------|
| M1 | Environment & Data Ready | Week 1 | Dev environment, real events downloaded, manifest built |
| M2 | Phase 1 Complete | Week 4 | Feature extraction + RF baseline trained and validated |
| M3 | Synthetic Pipeline Ready | Week 5 | 10K+ synthetic samples generated via PyCBC |
| M4 | Phase 2 Complete | Week 7 | CNN + physics-informed model trained on Colab A100 |
| M5 | Final Evaluation | Week 8 | O4a test, domain shift analysis, latency benchmarks |
| M6 | Paper Ready | Week 10 | Figures, write-up, presentation materials |

---

# Epics

## Epic 1: Project Setup & Infrastructure
**Milestone:** M1  
**Goal:** Development environment ready, data pipeline functional

---

### E1-1: Environment Setup
**Priority:** 🔴 Critical | **Estimate:** 2-3 hours

**Objective:** Create reproducible Python environment with all dependencies

**Implementation Steps:**
1. Create conda environment:
   ```bash
   conda create -n gw-classify python=3.10 -y
   conda activate gw-classify
   ```

2. Install core dependencies:
   ```bash
   # GW data access
   pip install gwpy ligo-segments
   
   # Waveform generation (Phase 2 prep)
   conda install -c conda-forge pycbc  # or pip install pycbc
   
   # ML/DL stack
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install scikit-learn xgboost shap
   
   # Data science
   pip install numpy scipy pandas matplotlib seaborn
   pip install jupyter ipykernel
   
   # Utilities
   pip install pyyaml tqdm h5py
   ```

3. Verify imports:
   ```python
   import gwpy
   import torch
   print(f"GWpy: {gwpy.__version__}")
   print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
   ```

4. Register Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name gw-classify
   ```

5. Export environment:
   ```bash
   pip freeze > requirements.txt
   conda env export > environment.yml
   ```

**Acceptance Criteria:**
- [ ] All imports work without errors
- [ ] `gwpy.timeseries.TimeSeries.fetch_open_data` accessible
- [ ] PyTorch detects GPU (if available)
- [ ] requirements.txt updated

**Files Modified:**
- `requirements.txt`
- `environment.yml` (new)

---

### E1-2: Build Event Manifest
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Create ground-truth CSV of all GWTC events with labels

**Implementation Steps:**
1. Query GWOSC event database:
   ```python
   from gwosc.datasets import find_datasets, event_gps
   from gwosc import datasets
   
   # Get all GWTC events
   gwtc1 = datasets.find_datasets(type='event', catalog='GWTC-1-confident')
   gwtc2 = datasets.find_datasets(type='event', catalog='GWTC-2.1-confident')
   gwtc3 = datasets.find_datasets(type='event', catalog='GWTC-3-confident')
   ```

2. For each event, extract:
   - `event_name`: GW150914, GW170817, etc.
   - `gps_time`: GPS merger time
   - `run`: O1, O2, O3a, O3b (derived from GPS)
   - `detectors`: H1, L1, V1 availability
   - `source_class`: BBH, BNS, NSBH (from catalog or mass-based)
   - `m1`, `m2`: Component masses (for validation)
   - `chirp_mass`: Mc for physics-informed loss validation

3. Classification logic:
   ```python
   def classify_event(m1, m2):
       """Classify based on component masses"""
       ns_mass_max = 3.0  # Solar masses
       if m1 < ns_mass_max and m2 < ns_mass_max:
           return 'BNS'
       elif m1 >= ns_mass_max and m2 < ns_mass_max:
           return 'NSBH'
       else:
           return 'BBH'
   
   # Binary label for our task
   df['label_binary'] = df['source_class'].apply(
       lambda x: 'NS-present' if x in ['BNS', 'NSBH'] else 'BBH'
   )
   ```

4. GPS → Run mapping:
   ```python
   RUN_GPS_RANGES = {
       'O1': (1126051217, 1137254417),   # Sep 2015 - Jan 2016
       'O2': (1164556817, 1187733618),   # Nov 2016 - Aug 2017
       'O3a': (1238166018, 1253977218),  # Apr 2019 - Oct 2019
       'O3b': (1256655618, 1269363618),  # Nov 2019 - Mar 2020
       'O4a': (1369008018, 1388534418),  # May 2023 - Jan 2024
   }
   ```

5. Save manifest:
   ```python
   df.to_csv('manifests/event_manifest.csv', index=False)
   ```

**Expected Output:**
```
event_name,gps_time,run,detectors,source_class,label_binary,m1,m2,chirp_mass,far
GW150914,1126259462.4,O1,H1L1,BBH,BBH,35.6,30.6,28.6,<1e-7
GW170817,1187008882.4,O2,H1L1V1,BNS,NS-present,1.46,1.27,1.186,<1e-7
...
```

**Acceptance Criteria:**
- [ ] ~90 events from O1-O3 in manifest
- [ ] All events have valid GPS times
- [ ] Labels verified against GWTC papers
- [ ] Train (O1-O3) and test (O4a) clearly separated

**Files Created:**
- `manifests/event_manifest.csv`
- `notebooks/01_data_exploration.ipynb` (manifest building code)

---

### E1-3: Download Real Events
**Priority:** 🔴 Critical | **Estimate:** 4-6 hours (mostly waiting)

**Objective:** Fetch strain data for all manifest events

**Implementation Steps:**
1. Create download function in `src/data/fetcher.py`:
   ```python
   from gwpy.timeseries import TimeSeries
   import os
   
   def download_event(event_name, gps_time, detector='L1', 
                      window=32, sample_rate=4096, output_dir='data/raw'):
       """Download strain segment around event"""
       t0 = gps_time - window
       t1 = gps_time + window
       
       try:
           strain = TimeSeries.fetch_open_data(
               detector, t0, t1,
               sample_rate=sample_rate,
               cache=True
           )
           
           # Save to HDF5
           run = get_run_from_gps(gps_time)
           outpath = f"{output_dir}/{run}/{event_name}_{detector}.hdf5"
           os.makedirs(os.path.dirname(outpath), exist_ok=True)
           strain.write(outpath, overwrite=True)
           
           return {'status': 'success', 'path': outpath, 'duration': len(strain)/sample_rate}
       
       except Exception as e:
           return {'status': 'failed', 'error': str(e)}
   ```

2. Batch download script `scripts/download_events.py`:
   ```python
   import pandas as pd
   from tqdm import tqdm
   from src.data.fetcher import download_event
   
   manifest = pd.read_csv('manifests/event_manifest.csv')
   results = []
   
   for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
       for detector in ['L1', 'H1']:
           result = download_event(
               row['event_name'], 
               row['gps_time'],
               detector=detector
           )
           results.append({**row.to_dict(), 'detector': detector, **result})
   
   pd.DataFrame(results).to_csv('manifests/download_log.csv', index=False)
   ```

3. Handle failures gracefully:
   - Some events may have data gaps
   - Some detectors may be offline
   - Log all failures for manual review

4. Organize by run:
   ```
   data/raw/
   ├── O1/
   │   ├── GW150914_L1.hdf5
   │   ├── GW150914_H1.hdf5
   │   └── ...
   ├── O2/
   ├── O3/
   └── O4a/  (test set, download later)
   ```

**Acceptance Criteria:**
- [ ] ≥80% of events downloaded successfully for L1
- [ ] Download log documents all failures
- [ ] Total data size ~5-10 GB
- [ ] Files readable with GWpy

**Files Created:**
- `src/data/fetcher.py`
- `scripts/download_events.py`
- `manifests/download_log.csv`
- `data/raw/{O1,O2,O3}/*.hdf5`

---

### E1-4: Data Validation
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Verify data integrity and document issues

**Implementation Steps:**
1. Create validation script:
   ```python
   def validate_event(filepath):
       """Check strain file integrity"""
       try:
           strain = TimeSeries.read(filepath)
           return {
               'valid': True,
               'duration': strain.duration.value,
               'sample_rate': strain.sample_rate.value,
               'has_nans': np.isnan(strain.value).any(),
               'has_gaps': check_for_gaps(strain),
               'snr_proxy': np.std(strain.value)
           }
       except Exception as e:
           return {'valid': False, 'error': str(e)}
   ```

2. Check for common issues:
   - NaN values in strain
   - Data gaps (discontinuities)
   - Wrong sample rate
   - Corrupted HDF5 files
   - Duration mismatches

3. Generate validation report:
   ```python
   validation_df = pd.DataFrame(validation_results)
   print(f"Valid files: {validation_df['valid'].sum()}/{len(validation_df)}")
   print(f"Files with NaNs: {validation_df['has_nans'].sum()}")
   print(f"Files with gaps: {validation_df['has_gaps'].sum()}")
   ```

4. Document exclusions:
   - Create `manifests/excluded_events.csv` with reasons
   - Update main manifest with `data_quality` column

**Acceptance Criteria:**
- [ ] All files validated
- [ ] Validation report generated
- [ ] Excluded events documented with reasons
- [ ] Final "clean" event count established

**Files Created:**
- `manifests/validation_report.csv`
- `manifests/excluded_events.csv`

---

### E1-5: Config System
**Priority:** 🟡 Medium | **Estimate:** 1-2 hours

**Objective:** Centralize all parameters in YAML configs

**Implementation Steps:**
1. Update `configs/config.yaml`:
   ```yaml
   # Main project configuration
   project:
     name: "gw-event-classification"
     seed: 42
     
   data:
     sample_rate: 4096
     window_seconds: 32  # ±32s around merger
     primary_detector: "L1"
     secondary_detector: "H1"
     
   paths:
     raw_data: "data/raw"
     spectrograms: "data/spectrograms"
     features: "data/features"
     models: "outputs/models"
     
   preprocessing:
     bandpass_low: 20  # Hz
     bandpass_high: 512  # Hz
     whiten: true
     
   spectrogram:
     method: "q_transform"  # or "stft"
     fmin: 20
     fmax: 512
     qrange: [4, 64]
     outseg_duration: 4  # seconds around merger
   ```

2. Create config loader utility:
   ```python
   # src/utils/config.py
   import yaml
   
   def load_config(path='configs/config.yaml'):
       with open(path, 'r') as f:
           return yaml.safe_load(f)
   
   CONFIG = load_config()
   ```

3. Use config throughout codebase:
   ```python
   from src.utils.config import CONFIG
   
   sample_rate = CONFIG['data']['sample_rate']
   ```

**Acceptance Criteria:**
- [ ] All hardcoded values moved to config
- [ ] Config loader implemented
- [ ] Configs validated (no typos, correct types)

**Files Modified:**
- `configs/config.yaml`
- `configs/injection_params.yaml`
- `configs/model_params.yaml`
- `src/utils/config.py` (new)

---

## Epic 2: Preprocessing Pipeline
**Milestone:** M2  
**Goal:** Raw strain → clean, normalized spectrograms

---

### E2-1: Preprocessing Module
**Priority:** 🔴 Critical | **Estimate:** 4-5 hours

**Objective:** Implement standardized preprocessing for all strain data

**Implementation Steps:**
1. Update `src/data/preprocessing.py`:
   ```python
   from gwpy.timeseries import TimeSeries
   from gwpy.signal import filter_design
   import numpy as np
   
   class StrainPreprocessor:
       def __init__(self, config):
           self.sample_rate = config['data']['sample_rate']
           self.bandpass_low = config['preprocessing']['bandpass_low']
           self.bandpass_high = config['preprocessing']['bandpass_high']
           self.whiten = config['preprocessing']['whiten']
       
       def preprocess(self, strain: TimeSeries) -> TimeSeries:
           """Full preprocessing pipeline"""
           # 1. Resample if needed
           if strain.sample_rate.value != self.sample_rate:
               strain = strain.resample(self.sample_rate)
           
           # 2. Remove DC offset and linear trend
           strain = strain.detrend('linear')
           
           # 3. Bandpass filter (20-512 Hz)
           strain = strain.bandpass(
               self.bandpass_low, 
               self.bandpass_high,
               filtfilt=True
           )
           
           # 4. Whitening (spectral normalization)
           if self.whiten:
               strain = strain.whiten(
                   fftlength=4,  # seconds
                   overlap=2     # seconds
               )
           
           # 5. Notch filter for 60Hz power line (optional)
           strain = self._notch_filter(strain, [60, 120, 180])
           
           return strain
       
       def _notch_filter(self, strain, frequencies):
           """Remove power line harmonics"""
           for freq in frequencies:
               notch = filter_design.notch(freq, strain.sample_rate.value, Q=30)
               strain = strain.filter(notch, filtfilt=True)
           return strain
       
       def crop_around_merger(self, strain, gps_merger, window=4):
           """Extract segment centered on merger"""
           t0 = gps_merger - window/2
           t1 = gps_merger + window/2
           return strain.crop(t0, t1)
   ```

2. Test on known event (GW150914):
   ```python
   from gwpy.timeseries import TimeSeries
   
   # Load raw strain
   strain = TimeSeries.read('data/raw/O1/GW150914_L1.hdf5')
   
   # Preprocess
   preprocessor = StrainPreprocessor(CONFIG)
   clean_strain = preprocessor.preprocess(strain)
   
   # Visual check
   plot = clean_strain.plot()
   plot.savefig('outputs/figures/preprocessing_test.png')
   ```

3. Verify signal preservation:
   - Compare raw vs preprocessed Q-scans
   - Ensure chirp visible in preprocessed data
   - Check that whitening doesn't destroy signal

**Acceptance Criteria:**
- [ ] Preprocessing removes noise without destroying signal
- [ ] GW150914 chirp clearly visible after preprocessing
- [ ] Pipeline handles edge cases (short segments, gaps)
- [ ] Processing time < 1s per event

**Files Modified:**
- `src/data/preprocessing.py`

---

### E2-2: Spectrogram Generation
**Priority:** 🔴 Critical | **Estimate:** 4-5 hours

**Objective:** Convert preprocessed strain to time-frequency spectrograms

**Implementation Steps:**
1. Update `src/features/spectrogram.py`:
   ```python
   from gwpy.timeseries import TimeSeries
   from gwpy.spectrogram import Spectrogram
   import numpy as np
   from PIL import Image
   
   class SpectrogramGenerator:
       def __init__(self, config):
           self.method = config['spectrogram']['method']
           self.fmin = config['spectrogram']['fmin']
           self.fmax = config['spectrogram']['fmax']
           self.qrange = config['spectrogram']['qrange']
           self.duration = config['spectrogram']['outseg_duration']
       
       def generate(self, strain: TimeSeries, gps_merger: float) -> Spectrogram:
           """Generate spectrogram around merger time"""
           
           if self.method == 'q_transform':
               # Q-transform (better time-freq resolution)
               qgram = strain.q_transform(
                   frange=(self.fmin, self.fmax),
                   qrange=self.qrange,
                   outseg=(gps_merger - self.duration/2, 
                          gps_merger + self.duration/2)
               )
               return qgram
           
           elif self.method == 'stft':
               # Short-time Fourier transform
               specgram = strain.spectrogram2(
                   fftlength=0.5,  # 500ms windows
                   overlap=0.25,   # 50% overlap
                   window='hann'
               )
               # Crop frequency range
               return specgram.crop_frequencies(self.fmin, self.fmax)
       
       def to_image(self, specgram: Spectrogram, size=(224, 224)) -> np.ndarray:
           """Convert spectrogram to normalized image array"""
           # Get power values (log scale)
           data = np.abs(specgram.value)
           data = np.log10(data + 1e-10)
           
           # Normalize to 0-255
           data = (data - data.min()) / (data.max() - data.min() + 1e-10)
           data = (data * 255).astype(np.uint8)
           
           # Resize for CNN
           img = Image.fromarray(data)
           img = img.resize(size, Image.BILINEAR)
           
           return np.array(img)
       
       def save_outputs(self, specgram, event_name, output_dir):
           """Save both matrix and image formats"""
           # Save raw matrix (for feature extraction)
           matrix_path = f"{output_dir}/matrices/{event_name}.npy"
           np.save(matrix_path, specgram.value)
           
           # Save image (for CNN)
           img_array = self.to_image(specgram)
           img_path = f"{output_dir}/images/{event_name}.png"
           Image.fromarray(img_array).save(img_path)
           
           # Save metadata
           meta = {
               'times': specgram.times.value.tolist(),
               'frequencies': specgram.frequencies.value.tolist(),
               'shape': specgram.shape
           }
           
           return {'matrix': matrix_path, 'image': img_path, 'meta': meta}
   ```

2. Test Q-transform on GW150914:
   ```python
   # Generate Q-scan
   qgram = generator.generate(clean_strain, gps_time)
   
   # Plot for verification
   plot = qgram.plot()
   ax = plot.gca()
   ax.set_yscale('log')
   ax.set_ylim(20, 512)
   ax.colorbar(label='Normalized energy')
   plot.savefig('outputs/figures/qscan_GW150914.png')
   ```

3. Compare BBH vs BNS spectrograms:
   - BBH: Lower frequencies, shorter duration
   - BNS: Higher frequencies, longer chirp
   - Document visual differences for paper

**Acceptance Criteria:**
- [ ] Q-transform produces clear chirp visualization
- [ ] Both matrix (.npy) and image (.png) outputs saved
- [ ] Images sized correctly for CNN (224×224)
- [ ] Processing time < 2s per event

**Files Modified:**
- `src/features/spectrogram.py`

---

### E2-3: Per-Event Normalization
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Normalize spectrograms to remove SNR dependence (ADR-015)

**Implementation Steps:**
1. Implement normalization methods:
   ```python
   class SpectrogramNormalizer:
       def __init__(self, method='zscore'):
           self.method = method
       
       def normalize(self, specgram: np.ndarray) -> np.ndarray:
           """Normalize spectrogram per-event"""
           
           if self.method == 'zscore':
               # Z-score: (x - mean) / std
               mean = np.mean(specgram)
               std = np.std(specgram)
               return (specgram - mean) / (std + 1e-10)
           
           elif self.method == 'minmax':
               # Min-max: scale to [0, 1]
               min_val = np.min(specgram)
               max_val = np.max(specgram)
               return (specgram - min_val) / (max_val - min_val + 1e-10)
           
           elif self.method == 'percentile':
               # Robust: use 1st and 99th percentile
               p1, p99 = np.percentile(specgram, [1, 99])
               clipped = np.clip(specgram, p1, p99)
               return (clipped - p1) / (p99 - p1 + 1e-10)
           
           elif self.method == 'log_zscore':
               # Log transform then z-score
               log_spec = np.log10(np.abs(specgram) + 1e-10)
               return (log_spec - np.mean(log_spec)) / (np.std(log_spec) + 1e-10)
   ```

2. Compare normalization methods:
   ```python
   # Test on events with different SNRs
   high_snr_event = 'GW150914'  # SNR ~24
   low_snr_event = 'GW190425'   # SNR ~12
   
   for method in ['zscore', 'minmax', 'percentile', 'log_zscore']:
       normalizer = SpectrogramNormalizer(method)
       # Compare distributions after normalization
   ```

3. Choose best method based on:
   - Feature distributions similar across SNR ranges
   - Chirp structure preserved
   - No information loss

**Acceptance Criteria:**
- [ ] Normalization reduces SNR dependence
- [ ] Feature distributions comparable across events
- [ ] Chirp structure not destroyed
- [ ] Method documented in ADR-015

**Files Modified:**
- `src/features/spectrogram.py` (add SpectrogramNormalizer class)

---

### E2-4: Preprocessing Notebook
**Priority:** 🟠 High | **Estimate:** 3-4 hours

**Objective:** Interactive notebook for visual validation of pipeline

**Implementation Steps:**
1. Create `notebooks/04_spectrogram_generation.ipynb`:
   ```python
   # Cell 1: Setup
   import sys
   sys.path.append('..')
   from src.data.preprocessing import StrainPreprocessor
   from src.features.spectrogram import SpectrogramGenerator
   from src.utils.config import CONFIG
   
   # Cell 2: Load example event
   event_name = 'GW150914'
   strain = TimeSeries.read(f'../data/raw/O1/{event_name}_L1.hdf5')
   gps_time = 1126259462.4
   
   # Cell 3: Raw data visualization
   plot = strain.plot()
   plt.title(f'{event_name} - Raw Strain')
   
   # Cell 4: Preprocessing steps (one by one)
   # Show effect of each step
   
   # Cell 5: Final spectrogram
   # Side-by-side: BBH vs BNS examples
   
   # Cell 6: Normalization comparison
   # Show different normalization methods
   ```

2. Include diagnostic plots:
   - Raw vs preprocessed time series
   - Power spectral density before/after whitening
   - Spectrogram with/without normalization
   - BBH vs BNS comparison

3. Add markdown explanations:
   - Why each preprocessing step
   - Parameter choices rationale
   - Links to relevant ADRs

**Acceptance Criteria:**
- [ ] Notebook runs end-to-end without errors
- [ ] All preprocessing steps visualized
- [ ] BBH vs BNS differences clear
- [ ] Suitable for paper figures

**Files Created:**
- `notebooks/04_spectrogram_generation.ipynb`

---

### E2-5: Batch Processing Script
**Priority:** 🔴 Critical | **Estimate:** 2-3 hours

**Objective:** Process all events through pipeline

**Implementation Steps:**
1. Create `scripts/generate_spectrograms.py`:
   ```python
   #!/usr/bin/env python
   """Batch generate spectrograms for all events"""
   
   import argparse
   import pandas as pd
   from tqdm import tqdm
   from pathlib import Path
   import logging
   
   from src.data.preprocessing import StrainPreprocessor
   from src.features.spectrogram import SpectrogramGenerator, SpectrogramNormalizer
   from src.utils.config import load_config
   
   def main(args):
       config = load_config(args.config)
       
       # Initialize processors
       preprocessor = StrainPreprocessor(config)
       generator = SpectrogramGenerator(config)
       normalizer = SpectrogramNormalizer(config['spectrogram'].get('normalization', 'zscore'))
       
       # Load manifest
       manifest = pd.read_csv(args.manifest)
       
       # Filter by run if specified
       if args.run:
           manifest = manifest[manifest['run'] == args.run]
       
       results = []
       
       for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
           event_name = row['event_name']
           gps_time = row['gps_time']
           run = row['run']
           
           try:
               # Load strain
               strain_path = Path(config['paths']['raw_data']) / run / f"{event_name}_L1.hdf5"
               strain = TimeSeries.read(strain_path)
               
               # Preprocess
               clean_strain = preprocessor.preprocess(strain)
               
               # Generate spectrogram
               specgram = generator.generate(clean_strain, gps_time)
               
               # Normalize
               specgram_norm = normalizer.normalize(specgram.value)
               
               # Save outputs
               output_dir = Path(config['paths']['spectrograms'])
               outputs = generator.save_outputs(specgram_norm, event_name, output_dir)
               
               results.append({
                   'event_name': event_name,
                   'status': 'success',
                   **outputs
               })
               
           except Exception as e:
               logging.error(f"Failed {event_name}: {e}")
               results.append({
                   'event_name': event_name,
                   'status': 'failed',
                   'error': str(e)
               })
       
       # Save processing log
       pd.DataFrame(results).to_csv(args.output_log, index=False)
       print(f"Processed {len([r for r in results if r['status']=='success'])}/{len(results)} events")
   
   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--config', default='configs/config.yaml')
       parser.add_argument('--manifest', default='manifests/event_manifest.csv')
       parser.add_argument('--run', default=None, help='Process specific run only')
       parser.add_argument('--output-log', default='manifests/spectrogram_log.csv')
       args = parser.parse_args()
       main(args)
   ```

2. Run processing:
   ```bash
   # Process all events
   python scripts/generate_spectrograms.py
   
   # Or by run
   python scripts/generate_spectrograms.py --run O1
   python scripts/generate_spectrograms.py --run O2
   python scripts/generate_spectrograms.py --run O3
   ```

3. Verify outputs:
   ```bash
   # Check output counts
   ls data/spectrograms/images/*.png | wc -l
   ls data/spectrograms/matrices/*.npy | wc -l
   ```

**Acceptance Criteria:**
- [ ] All events processed (≥80% success rate)
- [ ] Processing log saved with status
- [ ] Output files match manifest events
- [ ] Total processing time < 30 min for ~90 events

**Files Created:**
- `scripts/generate_spectrograms.py`
- `manifests/spectrogram_log.csv`
- `data/spectrograms/images/*.png`
- `data/spectrograms/matrices/*.npy`

---

## Epic 3: Feature Extraction (Phase 1)
**Milestone:** M2  
**Goal:** Extract interpretable features from spectrograms

| Issue | Title | Description |
|-------|-------|-------------|
| E3-1 | Ridge extraction | Implement peak tracking for chirp ridge (ADR-018) |
| E3-2 | Chirp geometry features | Ridge slope, duration, time-to-peak |
| E3-3 | Frequency features | Peak freq, band power ratios, spectral centroid |
| E3-4 | Statistical features | Spectral entropy, kurtosis, skewness |
| E3-5 | Feature dataset | Generate features CSV for all events |
| E3-6 | Feature exploration | Notebook with distributions, correlations, class separation |

---

## Epic 4: Baseline Model (Phase 1)
**Milestone:** M2  
**Goal:** Train interpretable ML model on real events

| Issue | Title | Description |
|-------|-------|-------------|
| E4-1 | Train/val split | Stratified split preserving class balance |
| E4-2 | Random Forest baseline | Train RF with class weights, 5-fold CV |
| E4-3 | XGBoost comparison | Train XGBoost, compare to RF |
| E4-4 | Feature importance | SHAP values, permutation importance |
| E4-5 | Baseline evaluation | Accuracy, F1, AUC, confusion matrix |
| E4-6 | Phase 1 report | Document baseline results, identify best features |

---

## Epic 5: Synthetic Data Pipeline (Phase 2)
**Milestone:** M3  
**Goal:** Generate 10K+ synthetic injections via PyCBC

| Issue | Title | Description |
|-------|-------|-------------|
| E5-1 | PyCBC waveform setup | Implement BBH/BNS/NSBH waveform generation |
| E5-2 | Noise segment collection | Download background noise from O1-O3 |
| E5-3 | Injection pipeline | Inject waveforms into noise at various SNRs |
| E5-4 | Synthetic manifest | Track all injections with parameters |
| E5-5 | Generate BBH samples | 5,000 BBH injections |
| E5-6 | Generate NS-present samples | 5,000 BNS + NSBH injections |
| E5-7 | Synthetic spectrograms | Process all synthetic → spectrograms + images |
| E5-8 | Quality validation | Visual check, SNR distribution analysis |

---

## Epic 6: CNN Model (Phase 2)
**Milestone:** M4  
**Goal:** Train deep learning classifier on synthetic + real data

| Issue | Title | Description |
|-------|-------|-------------|
| E6-1 | PyTorch Dataset | Implement SpectrogramDataset with augmentation |
| E6-2 | CNN architecture | ResNet18/50 with custom head |
| E6-3 | Data augmentation | Time jitter, noise injection, freq masking (ADR-014) |
| E6-4 | Colab notebook | A100-optimized training notebook |
| E6-5 | Training loop | AMP, gradient accumulation, early stopping |
| E6-6 | Hyperparameter tuning | LR, batch size, weight decay sweep |
| E6-7 | Model checkpointing | Save best model by val loss |
| E6-8 | CNN evaluation | Test on real events, compare to baseline |

---

## Epic 7: Physics-Informed Model (Phase 2)
**Milestone:** M4  
**Goal:** Add physics constraints to improve classification

| Issue | Title | Description |
|-------|-------|-------------|
| E7-1 | Physics loss function | Implement chirp mass constraint loss (ADR-013) |
| E7-2 | Multi-task head | Classification + Mc regression |
| E7-3 | Combined loss | α·L_class + β·L_physics |
| E7-4 | PINN training | Train with physics-informed loss |
| E7-5 | Ablation study | With vs without physics loss comparison |
| E7-6 | Mc validation | Compare estimated Mc to catalog values |

---

## Epic 8: Final Evaluation
**Milestone:** M5  
**Goal:** Comprehensive evaluation on held-out O4a data

| Issue | Title | Description |
|-------|-------|-------------|
| E8-1 | O4a data download | Fetch O4a events (test set) |
| E8-2 | O4a preprocessing | Apply same pipeline to O4a |
| E8-3 | Final predictions | Run all models on O4a |
| E8-4 | Domain shift analysis | Compare O3→O4a performance drop |
| E8-5 | Synthetic-to-real analysis | Performance on real vs synthetic |
| E8-6 | Latency benchmark | Measure end-to-end inference time |
| E8-7 | Error analysis | Analyze misclassified events |
| E8-8 | Final metrics table | Compile all results |

---

## Epic 9: Documentation & Write-up
**Milestone:** M6  
**Goal:** Publication-ready documentation

| Issue | Title | Description |
|-------|-------|-------------|
| E9-1 | Example spectrograms | Generate figures for paper |
| E9-2 | Feature distribution plots | Violin/box plots by class |
| E9-3 | ROC/PR curves | Publication-quality evaluation plots |
| E9-4 | Confusion matrices | Phase 1 vs Phase 2 comparison |
| E9-5 | Architecture diagram | Visual pipeline overview |
| E9-6 | Methods section | Write data/methods for paper |
| E9-7 | Results section | Write results with tables/figures |
| E9-8 | Discussion/limitations | Honest assessment of limitations |
| E9-9 | README polish | Final project README with setup instructions |
| E9-10 | Presentation prep | Slides for science fair |

---

# Issue Summary

| Epic | Issues | Priority |
|------|--------|----------|
| E1: Setup | 5 | 🔴 Critical |
| E2: Preprocessing | 5 | 🔴 Critical |
| E3: Features | 6 | 🔴 Critical |
| E4: Baseline | 6 | 🟠 High |
| E5: Synthetic | 8 | 🟠 High |
| E6: CNN | 8 | 🟠 High |
| E7: Physics | 6 | 🟡 Medium |
| E8: Evaluation | 8 | 🔴 Critical |
| E9: Write-up | 10 | 🟠 High |
| **Total** | **62** | |

---

# Dependency Graph

```
E1 (Setup)
    ↓
E2 (Preprocessing) ←─────────────────┐
    ↓                                │
E3 (Features)                    E5 (Synthetic)
    ↓                                │
E4 (Baseline Phase 1)                ↓
    ↓                            E6 (CNN)
    │                                │
    │                            E7 (Physics)
    │                                │
    └────────────→ E8 (Evaluation) ←─┘
                        ↓
                   E9 (Write-up)
```

---

# Quick Reference: Next Actions

**Week 1 Focus:**
1. E1-1: Environment setup
2. E1-2: Build event manifest
3. E1-3: Download real events

**Blockers to Watch:**
- GWOSC API availability
- PyCBC installation (can be tricky on Windows)
- Colab GPU quota limits

---

# GitHub Labels (Suggested)

| Label | Color | Description |
|-------|-------|-------------|
| `epic:setup` | #0052CC | Project setup tasks |
| `epic:preprocessing` | #5319E7 | Data preprocessing |
| `epic:features` | #006B75 | Feature extraction |
| `epic:baseline` | #B60205 | Phase 1 baseline model |
| `epic:synthetic` | #FBCA04 | Synthetic data generation |
| `epic:cnn` | #D93F0B | CNN model |
| `epic:physics` | #0E8A16 | Physics-informed model |
| `epic:evaluation` | #1D76DB | Final evaluation |
| `epic:docs` | #C5DEF5 | Documentation |
| `priority:critical` | #B60205 | Must complete |
| `priority:high` | #D93F0B | Important |
| `priority:medium` | #FBCA04 | Nice to have |
| `phase:1` | #BFD4F2 | Phase 1 work |
| `phase:2` | #D4C5F9 | Phase 2 work |
