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

---

### E3-1: Ridge Extraction
**Priority:** 🔴 Critical | **Estimate:** 4-5 hours

**Objective:** Implement chirp ridge tracking via peak detection (ADR-018)

**Implementation Steps:**
1. Update `src/features/ridge.py`:
   ```python
   import numpy as np
   from scipy.ndimage import maximum_filter, gaussian_filter1d
   from scipy.stats import linregress
   from dataclasses import dataclass
   from typing import Tuple, Optional
   
   @dataclass
   class RidgeResult:
       """Container for ridge extraction results"""
       times: np.ndarray           # Time bins where ridge detected
       frequencies: np.ndarray     # Peak frequency at each time
       powers: np.ndarray          # Power at each ridge point
       slope: float                # Linear fit slope (Hz/s)
       intercept: float            # Linear fit intercept
       r_squared: float            # Goodness of fit
       duration: float             # Ridge duration in seconds
       
   class ChirpRidgeExtractor:
       def __init__(self, config):
           self.power_threshold = config.get('ridge_power_threshold', 0.3)
           self.min_ridge_points = config.get('min_ridge_points', 10)
           self.smooth_sigma = config.get('ridge_smooth_sigma', 2)
           
       def extract(self, specgram: np.ndarray, times: np.ndarray, 
                   frequencies: np.ndarray) -> Optional[RidgeResult]:
           """
           Extract chirp ridge using per-time-bin peak tracking
           
           Algorithm:
           1. For each time bin, find frequency with maximum power
           2. Apply power threshold to filter noise
           3. Smooth the ridge trajectory
           4. Fit linear regression for slope estimation
           """
           n_times, n_freqs = specgram.shape
           
           # Step 1: Find peak frequency at each time bin
           peak_indices = np.argmax(specgram, axis=1)
           peak_frequencies = frequencies[peak_indices]
           peak_powers = specgram[np.arange(n_times), peak_indices]
           
           # Step 2: Normalize powers and apply threshold
           norm_powers = (peak_powers - peak_powers.min()) / (peak_powers.max() - peak_powers.min() + 1e-10)
           valid_mask = norm_powers > self.power_threshold
           
           # Need minimum number of valid points
           if valid_mask.sum() < self.min_ridge_points:
               return None
           
           # Step 3: Extract valid ridge points
           valid_times = times[valid_mask]
           valid_freqs = peak_frequencies[valid_mask]
           valid_powers = peak_powers[valid_mask]
           
           # Step 4: Smooth the ridge (reduce noise)
           if len(valid_freqs) > self.smooth_sigma * 2:
               smoothed_freqs = gaussian_filter1d(valid_freqs, self.smooth_sigma)
           else:
               smoothed_freqs = valid_freqs
           
           # Step 5: Linear regression for slope
           slope, intercept, r_value, _, _ = linregress(valid_times, smoothed_freqs)
           
           return RidgeResult(
               times=valid_times,
               frequencies=smoothed_freqs,
               powers=valid_powers,
               slope=slope,
               intercept=intercept,
               r_squared=r_value**2,
               duration=valid_times[-1] - valid_times[0]
           )
       
       def extract_multi_ridge(self, specgram: np.ndarray, times: np.ndarray,
                               frequencies: np.ndarray, n_ridges: int = 3) -> list:
           """Extract multiple ridges (for complex signals)"""
           ridges = []
           remaining_specgram = specgram.copy()
           
           for _ in range(n_ridges):
               ridge = self.extract(remaining_specgram, times, frequencies)
               if ridge is None:
                   break
               ridges.append(ridge)
               
               # Mask out detected ridge for next iteration
               for t_idx, f in zip(range(len(ridge.times)), ridge.frequencies):
                   f_idx = np.argmin(np.abs(frequencies - f))
                   # Zero out neighborhood
                   remaining_specgram[t_idx, max(0,f_idx-5):min(len(frequencies),f_idx+5)] = 0
           
           return ridges
   ```

2. Test on GW150914:
   ```python
   # Load spectrogram
   specgram = np.load('data/spectrograms/matrices/GW150914.npy')
   
   # Extract ridge
   extractor = ChirpRidgeExtractor(CONFIG)
   ridge = extractor.extract(specgram, times, frequencies)
   
   print(f"Ridge slope: {ridge.slope:.2f} Hz/s")
   print(f"Duration: {ridge.duration:.3f} s")
   print(f"R²: {ridge.r_squared:.3f}")
   ```

3. Visualize ridge overlay:
   ```python
   plt.figure(figsize=(10, 6))
   plt.pcolormesh(times, frequencies, specgram.T, shading='auto')
   plt.plot(ridge.times, ridge.frequencies, 'r-', linewidth=2, label='Detected ridge')
   plt.xlabel('Time (s)')
   plt.ylabel('Frequency (Hz)')
   plt.colorbar(label='Power')
   plt.legend()
   plt.savefig('outputs/figures/ridge_detection_example.png')
   ```

**Acceptance Criteria:**
- [ ] Ridge detected on >90% of events
- [ ] Slope sign correct (positive for chirp)
- [ ] R² > 0.7 for clean events
- [ ] Visual validation on BBH and BNS examples

**Files Modified:**
- `src/features/ridge.py`

---

### E3-2: Chirp Geometry Features
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Extract features describing chirp shape and evolution

**Implementation Steps:**
1. Add to `src/features/extractor.py`:
   ```python
   from src.features.ridge import ChirpRidgeExtractor, RidgeResult
   import numpy as np
   from scipy.stats import linregress
   
   class ChirpGeometryExtractor:
       """Extract features from chirp ridge geometry"""
       
       def __init__(self, config):
           self.ridge_extractor = ChirpRidgeExtractor(config)
       
       def extract(self, specgram: np.ndarray, times: np.ndarray,
                   frequencies: np.ndarray, gps_merger: float) -> dict:
           """Extract chirp geometry features"""
           
           ridge = self.ridge_extractor.extract(specgram, times, frequencies)
           
           if ridge is None:
               return self._null_features()
           
           features = {}
           
           # 1. Ridge slope (Hz/s) - key discriminator
           # BBH: steeper slope (faster chirp), BNS: gentler slope
           features['ridge_slope'] = ridge.slope
           features['ridge_slope_abs'] = abs(ridge.slope)
           
           # 2. Ridge curvature (deviation from linear)
           # Fit quadratic and measure curvature
           if len(ridge.times) > 5:
               coeffs = np.polyfit(ridge.times, ridge.frequencies, 2)
               features['ridge_curvature'] = coeffs[0]  # Quadratic coefficient
           else:
               features['ridge_curvature'] = 0.0
           
           # 3. Chirp duration (seconds)
           features['chirp_duration'] = ridge.duration
           
           # 4. Frequency range spanned
           features['freq_start'] = ridge.frequencies[0]
           features['freq_end'] = ridge.frequencies[-1]
           features['freq_range'] = ridge.frequencies[-1] - ridge.frequencies[0]
           
           # 5. Time to peak frequency (relative to merger)
           peak_idx = np.argmax(ridge.frequencies)
           features['time_to_peak'] = ridge.times[peak_idx] - gps_merger
           
           # 6. Peak frequency
           features['peak_frequency'] = ridge.frequencies.max()
           
           # 7. Goodness of fit (R²)
           features['ridge_r_squared'] = ridge.r_squared
           
           # 8. Ridge power statistics
           features['ridge_power_mean'] = ridge.powers.mean()
           features['ridge_power_max'] = ridge.powers.max()
           features['ridge_power_std'] = ridge.powers.std()
           
           # 9. Frequency acceleration (second derivative proxy)
           if len(ridge.frequencies) > 10:
               freq_diff = np.diff(ridge.frequencies)
               freq_accel = np.diff(freq_diff)
               features['freq_acceleration_mean'] = freq_accel.mean()
               features['freq_acceleration_max'] = freq_accel.max()
           else:
               features['freq_acceleration_mean'] = 0.0
               features['freq_acceleration_max'] = 0.0
           
           # 10. Chirp mass proxy (from f_peak and slope)
           # f ∝ M_c^(-5/8), so M_c ∝ f^(-8/5)
           # This is a rough estimate, not calibrated
           if features['peak_frequency'] > 0:
               features['chirp_mass_proxy'] = (features['peak_frequency'] / 100) ** (-8/5) * 30
           else:
               features['chirp_mass_proxy'] = 0.0
           
           return features
       
       def _null_features(self) -> dict:
           """Return null features when ridge not detected"""
           return {
               'ridge_slope': 0.0,
               'ridge_slope_abs': 0.0,
               'ridge_curvature': 0.0,
               'chirp_duration': 0.0,
               'freq_start': 0.0,
               'freq_end': 0.0,
               'freq_range': 0.0,
               'time_to_peak': 0.0,
               'peak_frequency': 0.0,
               'ridge_r_squared': 0.0,
               'ridge_power_mean': 0.0,
               'ridge_power_max': 0.0,
               'ridge_power_std': 0.0,
               'freq_acceleration_mean': 0.0,
               'freq_acceleration_max': 0.0,
               'chirp_mass_proxy': 0.0,
           }
   ```

2. Expected feature differences:

   | Feature | BBH (typical) | BNS (typical) |
   |---------|---------------|---------------|
   | ridge_slope | 500-2000 Hz/s | 100-500 Hz/s |
   | chirp_duration | 0.1-0.5 s | 1-10 s |
   | peak_frequency | 100-300 Hz | 500-1500 Hz |
   | freq_range | 50-200 Hz | 200-1000 Hz |

**Acceptance Criteria:**
- [ ] 16 geometry features extracted per event
- [ ] Features show expected BBH vs BNS differences
- [ ] No NaN/Inf values in output
- [ ] Feature extraction < 0.5s per event

**Files Modified:**
- `src/features/extractor.py`

---

### E3-3: Frequency Features
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Extract frequency-domain features from spectrograms

**Implementation Steps:**
1. Add `FrequencyFeatureExtractor` class:
   ```python
   class FrequencyFeatureExtractor:
       """Extract frequency-domain features from spectrograms"""
       
       def __init__(self, config):
           # Define frequency bands for analysis
           self.bands = {
               'low': (20, 80),      # Low-freq: heavy BBH
               'mid': (80, 200),     # Mid-freq: light BBH, heavy NSBH
               'high': (200, 500),   # High-freq: BNS, light NSBH
               'very_high': (500, 1000)  # Very high: BNS inspiral
           }
       
       def extract(self, specgram: np.ndarray, frequencies: np.ndarray) -> dict:
           """Extract frequency features"""
           features = {}
           
           # 1. Band power ratios
           total_power = specgram.sum()
           for band_name, (f_low, f_high) in self.bands.items():
               mask = (frequencies >= f_low) & (frequencies < f_high)
               band_power = specgram[:, mask].sum()
               features[f'power_{band_name}'] = band_power / (total_power + 1e-10)
           
           # 2. Band power ratios (discriminative)
           features['power_ratio_low_high'] = (
               features['power_low'] / (features['power_high'] + 1e-10)
           )
           features['power_ratio_mid_high'] = (
               features['power_mid'] / (features['power_high'] + 1e-10)
           )
           
           # 3. Spectral centroid (center of mass)
           power_per_freq = specgram.sum(axis=0)
           features['spectral_centroid'] = np.average(frequencies, weights=power_per_freq + 1e-10)
           
           # 4. Spectral bandwidth (spread around centroid)
           centroid = features['spectral_centroid']
           features['spectral_bandwidth'] = np.sqrt(
               np.average((frequencies - centroid)**2, weights=power_per_freq + 1e-10)
           )
           
           # 5. Spectral rolloff (frequency below which 85% power)
           cumsum = np.cumsum(power_per_freq)
           rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
           features['spectral_rolloff'] = frequencies[min(rolloff_idx, len(frequencies)-1)]
           
           # 6. Spectral flatness (tonal vs noise-like)
           geometric_mean = np.exp(np.mean(np.log(power_per_freq + 1e-10)))
           arithmetic_mean = np.mean(power_per_freq)
           features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
           
           # 7. Peak frequency (global maximum)
           max_power_freq_idx = np.argmax(power_per_freq)
           features['peak_frequency_global'] = frequencies[max_power_freq_idx]
           
           # 8. Frequency at max power (time-resolved)
           time_of_max = np.unravel_index(specgram.argmax(), specgram.shape)[0]
           freq_of_max = np.argmax(specgram[time_of_max, :])
           features['freq_at_max_power'] = frequencies[freq_of_max]
           
           # 9. Number of significant peaks
           from scipy.signal import find_peaks
           peaks, _ = find_peaks(power_per_freq, height=power_per_freq.max() * 0.1)
           features['n_spectral_peaks'] = len(peaks)
           
           return features
   ```

2. Expected discriminative power:
   - `power_ratio_low_high`: BBH >> BNS (BBH has more low-freq power)
   - `spectral_centroid`: BBH < BNS (BBH centered at lower freq)
   - `spectral_rolloff`: BBH < BNS

**Acceptance Criteria:**
- [ ] 12+ frequency features extracted
- [ ] Band power ratios discriminate BBH vs BNS
- [ ] Spectral centroid shows expected pattern
- [ ] No division by zero errors

**Files Modified:**
- `src/features/extractor.py`

---

### E3-4: Statistical Features
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Extract statistical/texture features from spectrograms

**Implementation Steps:**
1. Add `StatisticalFeatureExtractor` class:
   ```python
   from scipy.stats import kurtosis, skew, entropy
   
   class StatisticalFeatureExtractor:
       """Extract statistical features from spectrograms"""
       
       def extract(self, specgram: np.ndarray) -> dict:
           """Extract statistical features"""
           features = {}
           
           # Flatten for global statistics
           flat = specgram.flatten()
           
           # 1. Basic statistics
           features['power_mean'] = np.mean(flat)
           features['power_std'] = np.std(flat)
           features['power_max'] = np.max(flat)
           features['power_min'] = np.min(flat)
           features['power_range'] = features['power_max'] - features['power_min']
           
           # 2. Higher-order moments
           features['power_skewness'] = skew(flat)
           features['power_kurtosis'] = kurtosis(flat)
           
           # 3. Spectral entropy (information content)
           # Normalize to probability distribution
           prob = flat / (flat.sum() + 1e-10)
           features['spectral_entropy'] = entropy(prob + 1e-10)
           
           # 4. Temporal statistics (variation over time)
           power_per_time = specgram.sum(axis=1)
           features['temporal_std'] = np.std(power_per_time)
           features['temporal_max_idx'] = np.argmax(power_per_time) / len(power_per_time)
           
           # 5. Contrast (local variation)
           # Use gradient magnitude as proxy
           grad_t = np.gradient(specgram, axis=0)
           grad_f = np.gradient(specgram, axis=1)
           gradient_mag = np.sqrt(grad_t**2 + grad_f**2)
           features['contrast'] = np.mean(gradient_mag)
           
           # 6. Homogeneity proxy (inverse of contrast)
           features['homogeneity'] = 1.0 / (features['contrast'] + 1e-10)
           
           # 7. Energy concentration
           # Fraction of power in top 10% of bins
           threshold = np.percentile(flat, 90)
           features['energy_concentration'] = flat[flat > threshold].sum() / (flat.sum() + 1e-10)
           
           # 8. Signal-to-noise proxy
           # Ratio of peak to median
           features['snr_proxy'] = features['power_max'] / (np.median(flat) + 1e-10)
           
           return features
   ```

**Acceptance Criteria:**
- [ ] 14+ statistical features extracted
- [ ] Entropy meaningful (higher for noise-like)
- [ ] SNR proxy correlates with catalog SNR
- [ ] No numerical instabilities

**Files Modified:**
- `src/features/extractor.py`

---

### E3-5: Feature Dataset Generation
**Priority:** 🔴 Critical | **Estimate:** 2-3 hours

**Objective:** Generate complete feature CSV for all events

**Implementation Steps:**
1. Create `scripts/extract_features.py`:
   ```python
   #!/usr/bin/env python
   """Extract features from all spectrograms"""
   
   import argparse
   import pandas as pd
   import numpy as np
   from tqdm import tqdm
   from pathlib import Path
   
   from src.features.extractor import (
       ChirpGeometryExtractor,
       FrequencyFeatureExtractor, 
       StatisticalFeatureExtractor
   )
   from src.utils.config import load_config
   
   def main(args):
       config = load_config(args.config)
       
       # Initialize extractors
       geometry_ext = ChirpGeometryExtractor(config)
       frequency_ext = FrequencyFeatureExtractor(config)
       statistical_ext = StatisticalFeatureExtractor()
       
       # Load manifest
       manifest = pd.read_csv(args.manifest)
       
       all_features = []
       
       for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
           event_name = row['event_name']
           gps_time = row['gps_time']
           
           try:
               # Load spectrogram
               matrix_path = Path(config['paths']['spectrograms']) / 'matrices' / f"{event_name}.npy"
               specgram = np.load(matrix_path)
               
               # Load metadata for times/frequencies
               # (assuming saved during spectrogram generation)
               meta_path = matrix_path.with_suffix('.json')
               with open(meta_path) as f:
                   meta = json.load(f)
               times = np.array(meta['times'])
               frequencies = np.array(meta['frequencies'])
               
               # Extract all features
               geom_features = geometry_ext.extract(specgram, times, frequencies, gps_time)
               freq_features = frequency_ext.extract(specgram, frequencies)
               stat_features = statistical_ext.extract(specgram)
               
               # Combine
               features = {
                   'event_name': event_name,
                   'label': row['label_binary'],
                   'run': row['run'],
                   **geom_features,
                   **freq_features,
                   **stat_features
               }
               
               all_features.append(features)
               
           except Exception as e:
               print(f"Failed {event_name}: {e}")
               continue
       
       # Create DataFrame and save
       df = pd.DataFrame(all_features)
       df.to_csv(args.output, index=False)
       
       print(f"\nFeature extraction complete!")
       print(f"Events: {len(df)}")
       print(f"Features: {len(df.columns) - 3}")  # Exclude name, label, run
       print(f"Output: {args.output}")
   
   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--config', default='configs/config.yaml')
       parser.add_argument('--manifest', default='manifests/event_manifest.csv')
       parser.add_argument('--output', default='data/features/features.csv')
       args = parser.parse_args()
       main(args)
   ```

2. Run extraction:
   ```bash
   python scripts/extract_features.py
   ```

3. Verify output:
   ```python
   df = pd.read_csv('data/features/features.csv')
   print(df.shape)  # (90, ~45)
   print(df.describe())
   print(df.isnull().sum())  # Check for missing values
   ```

**Expected Output:**
```
event_name,label,run,ridge_slope,chirp_duration,peak_frequency,...
GW150914,BBH,O1,1234.5,0.23,156.7,...
GW170817,NS-present,O2,345.6,2.1,892.3,...
```

**Acceptance Criteria:**
- [ ] ~40+ features per event
- [ ] No missing values
- [ ] Labels correctly assigned
- [ ] CSV readable by pandas

**Files Created:**
- `scripts/extract_features.py`
- `data/features/features.csv`

---

### E3-6: Feature Exploration Notebook
**Priority:** 🟠 High | **Estimate:** 3-4 hours

**Objective:** Analyze feature distributions and class separability

**Implementation Steps:**
1. Create `notebooks/05_feature_extraction.ipynb`:
   ```python
   # Cell 1: Load features
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   df = pd.read_csv('../data/features/features.csv')
   feature_cols = [c for c in df.columns if c not in ['event_name', 'label', 'run']]
   
   print(f"Events: {len(df)}")
   print(f"BBH: {(df['label']=='BBH').sum()}")
   print(f"NS-present: {(df['label']=='NS-present').sum()}")
   
   # Cell 2: Feature distributions by class
   fig, axes = plt.subplots(4, 4, figsize=(16, 16))
   for ax, col in zip(axes.flat, feature_cols[:16]):
       for label in ['BBH', 'NS-present']:
           data = df[df['label']==label][col]
           ax.hist(data, alpha=0.5, label=label, bins=20)
       ax.set_title(col)
       ax.legend()
   plt.tight_layout()
   plt.savefig('../outputs/figures/feature_distributions.png')
   
   # Cell 3: Correlation matrix
   corr = df[feature_cols].corr()
   plt.figure(figsize=(14, 12))
   sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
   plt.title('Feature Correlation Matrix')
   plt.savefig('../outputs/figures/feature_correlations.png')
   
   # Cell 4: Class separability analysis
   from sklearn.feature_selection import mutual_info_classif
   
   X = df[feature_cols].values
   y = (df['label'] == 'NS-present').astype(int).values
   
   mi_scores = mutual_info_classif(X, y, random_state=42)
   mi_df = pd.DataFrame({'feature': feature_cols, 'MI': mi_scores})
   mi_df = mi_df.sort_values('MI', ascending=False)
   
   plt.figure(figsize=(10, 8))
   plt.barh(mi_df['feature'][:20], mi_df['MI'][:20])
   plt.xlabel('Mutual Information')
   plt.title('Top 20 Features by Mutual Information')
   plt.savefig('../outputs/figures/feature_importance_mi.png')
   
   # Cell 5: t-SNE visualization
   from sklearn.manifold import TSNE
   
   tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
   X_tsne = tsne.fit_transform(X)
   
   plt.figure(figsize=(10, 8))
   for label, color in [('BBH', 'blue'), ('NS-present', 'red')]:
       mask = df['label'] == label
       plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=label, alpha=0.7)
   plt.legend()
   plt.title('t-SNE Feature Space')
   plt.savefig('../outputs/figures/feature_tsne.png')
   
   # Cell 6: Top discriminative features
   print("\nTop 10 Most Discriminative Features:")
   print(mi_df.head(10).to_string(index=False))
   ```

2. Key visualizations:
   - Violin plots: feature distributions by class
   - Correlation heatmap: identify redundant features
   - t-SNE: 2D visualization of feature space
   - Bar chart: feature importance ranking

**Acceptance Criteria:**
- [ ] Clear class separation visible in key features
- [ ] Correlation analysis identifies redundant features
- [ ] t-SNE shows some clustering by class
- [ ] Top 10 features identified for baseline model

**Files Created:**
- `notebooks/05_feature_extraction.ipynb`
- `outputs/figures/feature_distributions.png`
- `outputs/figures/feature_correlations.png`
- `outputs/figures/feature_importance_mi.png`
- `outputs/figures/feature_tsne.png`

---

## Epic 4: Baseline Model (Phase 1)
**Milestone:** M2  
**Goal:** Train interpretable ML model on real events

---

### E4-1: Train/Validation Split
**Priority:** 🔴 Critical | **Estimate:** 1-2 hours

**Objective:** Create stratified train/val split preserving class balance

**Implementation Steps:**
1. Create split logic:
   ```python
   from sklearn.model_selection import StratifiedKFold, train_test_split
   import json
   
   def create_splits(df, test_size=0.2, n_folds=5, seed=42):
       """
       Create train/val splits with stratification
       
       Strategy:
       - O1-O3 for training/validation (cross-validation)
       - O4a held out for final testing (not used in Phase 1)
       """
       
       # Filter to training runs only
       train_df = df[df['run'].isin(['O1', 'O2', 'O3a', 'O3b'])]
       test_df = df[df['run'] == 'O4a']  # Held out
       
       X = train_df['event_name'].values
       y = train_df['label'].values
       
       # Create k-fold splits for cross-validation
       skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
       
       splits = {
           'train_events': train_df['event_name'].tolist(),
           'test_events': test_df['event_name'].tolist(),
           'folds': []
       }
       
       for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
           splits['folds'].append({
               'fold': fold,
               'train': X[train_idx].tolist(),
               'val': X[val_idx].tolist()
           })
       
       # Also create a single 80/20 split
       train_events, val_events = train_test_split(
           X, test_size=test_size, stratify=y, random_state=seed
       )
       splits['simple_split'] = {
           'train': train_events.tolist(),
           'val': val_events.tolist()
       }
       
       return splits
   
   # Save splits
   splits = create_splits(df)
   with open('manifests/splits.json', 'w') as f:
       json.dump(splits, f, indent=2)
   ```

2. Verify class balance:
   ```python
   for fold in splits['folds']:
       train_labels = df[df['event_name'].isin(fold['train'])]['label']
       val_labels = df[df['event_name'].isin(fold['val'])]['label']
       print(f"Fold {fold['fold']}: Train BBH={sum(train_labels=='BBH')}, "
             f"NS={sum(train_labels=='NS-present')} | "
             f"Val BBH={sum(val_labels=='BBH')}, NS={sum(val_labels=='NS-present')}")
   ```

**Expected Split:**
```
Total events: ~90 (O1-O3)
- BBH: ~85
- NS-present: ~5

Each fold:
- Train: ~72 events
- Val: ~18 events
```

**Acceptance Criteria:**
- [ ] Stratified splits maintain class ratio
- [ ] 5-fold CV setup ready
- [ ] O4a events excluded from training
- [ ] splits.json saved

**Files Created:**
- `manifests/splits.json`

---

### E4-2: Random Forest Baseline
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Train Random Forest classifier with cross-validation

**Implementation Steps:**
1. Update `src/models/baseline.py`:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_predict, cross_validate
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import Pipeline
   import numpy as np
   import joblib
   
   class BaselineClassifier:
       def __init__(self, model_type='rf', config=None):
           self.model_type = model_type
           self.config = config or {}
           self.pipeline = None
           
       def build_pipeline(self):
           """Build sklearn pipeline with scaling + classifier"""
           
           if self.model_type == 'rf':
               classifier = RandomForestClassifier(
                   n_estimators=self.config.get('n_estimators', 200),
                   max_depth=self.config.get('max_depth', 10),
                   min_samples_split=self.config.get('min_samples_split', 5),
                   min_samples_leaf=self.config.get('min_samples_leaf', 2),
                   class_weight='balanced',  # Handle imbalance
                   random_state=42,
                   n_jobs=-1
               )
           else:
               raise ValueError(f"Unknown model type: {self.model_type}")
           
           self.pipeline = Pipeline([
               ('scaler', StandardScaler()),
               ('classifier', classifier)
           ])
           
           return self.pipeline
       
       def cross_validate(self, X, y, cv=5):
           """Perform cross-validation with multiple metrics"""
           
           if self.pipeline is None:
               self.build_pipeline()
           
           scoring = {
               'accuracy': 'accuracy',
               'f1': 'f1',
               'precision': 'precision',
               'recall': 'recall',
               'roc_auc': 'roc_auc'
           }
           
           results = cross_validate(
               self.pipeline, X, y,
               cv=cv,
               scoring=scoring,
               return_train_score=True
           )
           
           return results
       
       def get_cv_predictions(self, X, y, cv=5):
           """Get cross-validated predictions for all samples"""
           
           if self.pipeline is None:
               self.build_pipeline()
           
           y_pred = cross_val_predict(self.pipeline, X, y, cv=cv)
           y_proba = cross_val_predict(self.pipeline, X, y, cv=cv, method='predict_proba')
           
           return y_pred, y_proba
       
       def fit(self, X, y):
           """Fit on full training data"""
           if self.pipeline is None:
               self.build_pipeline()
           self.pipeline.fit(X, y)
           return self
       
       def predict(self, X):
           """Predict class labels"""
           return self.pipeline.predict(X)
       
       def predict_proba(self, X):
           """Predict class probabilities"""
           return self.pipeline.predict_proba(X)
       
       def save(self, path):
           """Save trained model"""
           joblib.dump(self.pipeline, path)
       
       def load(self, path):
           """Load trained model"""
           self.pipeline = joblib.load(path)
           return self
   ```

2. Create training notebook `notebooks/06_phase1_baseline.ipynb`:
   ```python
   # Cell 1: Load data
   import pandas as pd
   import numpy as np
   from src.models.baseline import BaselineClassifier
   
   df = pd.read_csv('../data/features/features.csv')
   feature_cols = [c for c in df.columns if c not in ['event_name', 'label', 'run']]
   
   # Filter to training data (O1-O3)
   train_df = df[df['run'].isin(['O1', 'O2', 'O3a', 'O3b'])]
   
   X = train_df[feature_cols].values
   y = (train_df['label'] == 'NS-present').astype(int).values
   
   # Cell 2: Cross-validation
   rf_model = BaselineClassifier(model_type='rf', config={
       'n_estimators': 200,
       'max_depth': 10
   })
   
   cv_results = rf_model.cross_validate(X, y, cv=5)
   
   print("Cross-Validation Results:")
   for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
       scores = cv_results[f'test_{metric}']
       print(f"  {metric}: {scores.mean():.3f} ± {scores.std():.3f}")
   
   # Cell 3: Get predictions for confusion matrix
   y_pred, y_proba = rf_model.get_cv_predictions(X, y, cv=5)
   
   from sklearn.metrics import confusion_matrix, classification_report
   print("\nClassification Report:")
   print(classification_report(y, y_pred, target_names=['BBH', 'NS-present']))
   
   # Cell 4: Plot confusion matrix
   cm = confusion_matrix(y, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['BBH', 'NS-present'],
               yticklabels=['BBH', 'NS-present'])
   plt.xlabel('Predicted')
   plt.ylabel('Actual')
   plt.title('Random Forest - Confusion Matrix (5-Fold CV)')
   plt.savefig('../outputs/figures/rf_confusion_matrix.png')
   
   # Cell 5: Train final model on all training data
   rf_model.fit(X, y)
   rf_model.save('../outputs/models/phase1/rf_baseline.joblib')
   ```

**Acceptance Criteria:**
- [ ] 5-fold CV completed
- [ ] All metrics reported with std dev
- [ ] Confusion matrix shows NS-present recall ≥ 50% (given imbalance)
- [ ] Model saved to outputs/models/phase1/

**Files Modified:**
- `src/models/baseline.py`

**Files Created:**
- `notebooks/06_phase1_baseline.ipynb`
- `outputs/models/phase1/rf_baseline.joblib`
- `outputs/figures/rf_confusion_matrix.png`

---

### E4-3: XGBoost Comparison
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Train XGBoost and compare to Random Forest

**Implementation Steps:**
1. Add XGBoost to `src/models/baseline.py`:
   ```python
   from xgboost import XGBClassifier
   
   class BaselineClassifier:
       def build_pipeline(self):
           if self.model_type == 'rf':
               # ... existing RF code ...
               
           elif self.model_type == 'xgboost':
               # Calculate scale_pos_weight for imbalance
               classifier = XGBClassifier(
                   n_estimators=self.config.get('n_estimators', 200),
                   max_depth=self.config.get('max_depth', 6),
                   learning_rate=self.config.get('learning_rate', 0.1),
                   subsample=self.config.get('subsample', 0.8),
                   colsample_bytree=self.config.get('colsample_bytree', 0.8),
                   scale_pos_weight=self.config.get('scale_pos_weight', 15),  # ~85/5
                   random_state=42,
                   use_label_encoder=False,
                   eval_metric='logloss'
               )
   ```

2. Compare models in notebook:
   ```python
   # Train XGBoost
   xgb_model = BaselineClassifier(model_type='xgboost', config={
       'n_estimators': 200,
       'max_depth': 6,
       'scale_pos_weight': 15
   })
   
   xgb_results = xgb_model.cross_validate(X, y, cv=5)
   
   # Comparison table
   comparison = pd.DataFrame({
       'Metric': ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC-AUC'],
       'RF': [cv_results[f'test_{m}'].mean() for m in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']],
       'XGBoost': [xgb_results[f'test_{m}'].mean() for m in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']]
   })
   print(comparison.to_string(index=False))
   ```

3. ROC curve comparison:
   ```python
   from sklearn.metrics import roc_curve, auc
   
   # Get probabilities from both models
   _, rf_proba = rf_model.get_cv_predictions(X, y, cv=5)
   _, xgb_proba = xgb_model.get_cv_predictions(X, y, cv=5)
   
   plt.figure(figsize=(8, 6))
   for name, proba in [('RF', rf_proba), ('XGBoost', xgb_proba)]:
       fpr, tpr, _ = roc_curve(y, proba[:, 1])
       roc_auc = auc(fpr, tpr)
       plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
   
   plt.plot([0, 1], [0, 1], 'k--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('ROC Curve Comparison')
   plt.legend()
   plt.savefig('../outputs/figures/roc_comparison.png')
   ```

**Acceptance Criteria:**
- [ ] XGBoost trained with same CV setup
- [ ] Comparison table generated
- [ ] ROC curves plotted
- [ ] Best model identified

**Files Modified:**
- `src/models/baseline.py`
- `notebooks/06_phase1_baseline.ipynb`

**Files Created:**
- `outputs/models/phase1/xgb_baseline.joblib`
- `outputs/figures/roc_comparison.png`

---

### E4-4: Feature Importance Analysis
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Identify most important features using SHAP and permutation importance

**Implementation Steps:**
1. Add SHAP analysis:
   ```python
   import shap
   
   # Train model on full data
   rf_model.fit(X, y)
   
   # SHAP values
   explainer = shap.TreeExplainer(rf_model.pipeline.named_steps['classifier'])
   X_scaled = rf_model.pipeline.named_steps['scaler'].transform(X)
   shap_values = explainer.shap_values(X_scaled)
   
   # Summary plot
   plt.figure(figsize=(10, 8))
   shap.summary_plot(shap_values[1], X_scaled, feature_names=feature_cols, show=False)
   plt.tight_layout()
   plt.savefig('../outputs/figures/shap_summary.png')
   
   # Bar plot (mean absolute SHAP)
   plt.figure(figsize=(10, 8))
   shap.summary_plot(shap_values[1], X_scaled, feature_names=feature_cols, 
                     plot_type='bar', show=False)
   plt.tight_layout()
   plt.savefig('../outputs/figures/shap_importance.png')
   ```

2. Permutation importance:
   ```python
   from sklearn.inspection import permutation_importance
   
   perm_importance = permutation_importance(
       rf_model.pipeline, X, y, 
       n_repeats=30, 
       random_state=42,
       scoring='f1'
   )
   
   perm_df = pd.DataFrame({
       'feature': feature_cols,
       'importance': perm_importance.importances_mean,
       'std': perm_importance.importances_std
   }).sort_values('importance', ascending=False)
   
   print("Top 10 Features (Permutation Importance):")
   print(perm_df.head(10).to_string(index=False))
   ```

3. Compare importance methods:
   ```python
   # Random Forest built-in importance
   rf_importance = rf_model.pipeline.named_steps['classifier'].feature_importances_
   
   # Create comparison DataFrame
   importance_df = pd.DataFrame({
       'feature': feature_cols,
       'RF_builtin': rf_importance,
       'Permutation': perm_importance.importances_mean,
       'SHAP': np.abs(shap_values[1]).mean(axis=0)
   })
   
   # Rank correlation between methods
   from scipy.stats import spearmanr
   print(f"Spearman correlation (RF vs SHAP): {spearmanr(importance_df['RF_builtin'], importance_df['SHAP'])[0]:.3f}")
   ```

**Acceptance Criteria:**
- [ ] SHAP summary plot generated
- [ ] Top 10 features identified by each method
- [ ] Feature importance consistent across methods
- [ ] Interpretable features (ridge_slope, etc.) rank high

**Files Created:**
- `outputs/figures/shap_summary.png`
- `outputs/figures/shap_importance.png`
- `outputs/figures/permutation_importance.png`

---

### E4-5: Baseline Evaluation
**Priority:** 🔴 Critical | **Estimate:** 2-3 hours

**Objective:** Comprehensive evaluation of baseline models

**Implementation Steps:**
1. Create evaluation module `src/evaluation/metrics.py`:
   ```python
   from sklearn.metrics import (
       accuracy_score, f1_score, precision_score, recall_score,
       roc_auc_score, confusion_matrix, classification_report,
       precision_recall_curve, roc_curve
   )
   import numpy as np
   
   def evaluate_classifier(y_true, y_pred, y_proba=None):
       """Compute comprehensive classification metrics"""
       
       metrics = {
           'accuracy': accuracy_score(y_true, y_pred),
           'f1': f1_score(y_true, y_pred),
           'precision': precision_score(y_true, y_pred),
           'recall': recall_score(y_true, y_pred),  # NS-present recall!
           'specificity': recall_score(y_true, y_pred, pos_label=0),
       }
       
       if y_proba is not None:
           metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
       
       metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
       
       return metrics
   
   def print_evaluation_report(metrics, model_name='Model'):
       """Print formatted evaluation report"""
       print(f"\n{'='*50}")
       print(f"{model_name} Evaluation Report")
       print(f"{'='*50}")
       print(f"Accuracy:    {metrics['accuracy']:.3f}")
       print(f"F1 Score:    {metrics['f1']:.3f}")
       print(f"Precision:   {metrics['precision']:.3f}")
       print(f"Recall:      {metrics['recall']:.3f}  ← NS-present recall (target: >85%)")
       print(f"Specificity: {metrics['specificity']:.3f}")
       if 'roc_auc' in metrics:
           print(f"ROC-AUC:     {metrics['roc_auc']:.3f}")
       print(f"\nConfusion Matrix:")
       print(metrics['confusion_matrix'])
   ```

2. Evaluate all models:
   ```python
   # Evaluate RF
   y_pred_rf, y_proba_rf = rf_model.get_cv_predictions(X, y, cv=5)
   rf_metrics = evaluate_classifier(y, y_pred_rf, y_proba_rf[:, 1])
   print_evaluation_report(rf_metrics, 'Random Forest')
   
   # Evaluate XGBoost
   y_pred_xgb, y_proba_xgb = xgb_model.get_cv_predictions(X, y, cv=5)
   xgb_metrics = evaluate_classifier(y, y_pred_xgb, y_proba_xgb[:, 1])
   print_evaluation_report(xgb_metrics, 'XGBoost')
   ```

3. Create evaluation summary table:
   ```python
   summary = pd.DataFrame({
       'Model': ['Random Forest', 'XGBoost'],
       'Accuracy': [rf_metrics['accuracy'], xgb_metrics['accuracy']],
       'F1': [rf_metrics['f1'], xgb_metrics['f1']],
       'Precision': [rf_metrics['precision'], xgb_metrics['precision']],
       'Recall (NS)': [rf_metrics['recall'], xgb_metrics['recall']],
       'ROC-AUC': [rf_metrics['roc_auc'], xgb_metrics['roc_auc']]
   })
   summary.to_csv('../outputs/results/phase1_results.csv', index=False)
   print(summary.to_markdown(index=False))
   ```

**Acceptance Criteria:**
- [ ] All metrics computed for both models
- [ ] NS-present recall explicitly reported
- [ ] Results saved to CSV
- [ ] Best model selected based on recall

**Files Created:**
- `src/evaluation/metrics.py`
- `outputs/results/phase1_results.csv`

---

### E4-6: Phase 1 Report
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Document Phase 1 results and findings

**Implementation Steps:**
1. Create summary in notebook:
   ```python
   # Final notebook cell: Phase 1 Summary
   
   print("""
   # Phase 1 Results Summary
   
   ## Dataset
   - Total events: {n_events}
   - BBH: {n_bbh}
   - NS-present: {n_ns}
   - Features: {n_features}
   
   ## Best Model: {best_model}
   - Accuracy: {acc:.1%}
   - F1 Score: {f1:.3f}
   - NS-present Recall: {recall:.1%}
   - ROC-AUC: {auc:.3f}
   
   ## Top 5 Features:
   1. {f1}
   2. {f2}
   3. {f3}
   4. {f4}
   5. {f5}
   
   ## Key Findings:
   - Ridge slope is most discriminative feature
   - Frequency-based features (spectral_centroid, band_power_ratio) highly predictive
   - Class imbalance limits NS-present precision
   - Model generalizes across O1-O3 runs
   
   ## Limitations:
   - Small NS-present sample size (n=5-6)
   - High variance in CV metrics due to small dataset
   - Unable to validate on O4a (held out for Phase 2)
   
   ## Recommendations for Phase 2:
   - Synthetic data needed to improve NS-present classification
   - CNN may capture features not hand-engineered
   - Physics-informed loss could improve precision
   """.format(...))
   ```

2. Generate all Phase 1 figures:
   - Feature distributions (from E3-6)
   - Confusion matrices (RF and XGBoost)
   - ROC curves
   - SHAP importance plots
   - Example spectrograms (BBH vs BNS)

3. Save final metrics:
   ```python
   phase1_summary = {
       'best_model': 'Random Forest',
       'accuracy': rf_metrics['accuracy'],
       'f1': rf_metrics['f1'],
       'recall_ns': rf_metrics['recall'],
       'roc_auc': rf_metrics['roc_auc'],
       'top_features': perm_df['feature'].head(10).tolist(),
       'n_events': len(train_df),
       'n_features': len(feature_cols)
   }
   
   with open('../outputs/results/phase1_summary.json', 'w') as f:
       json.dump(phase1_summary, f, indent=2)
   ```

**Acceptance Criteria:**
- [ ] Summary document written
- [ ] All figures generated
- [ ] Key findings documented
- [ ] Limitations acknowledged
- [ ] Phase 2 recommendations provided

**Files Created:**
- `outputs/results/phase1_summary.json`
- `outputs/figures/phase1_summary.png` (combined figure)

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
