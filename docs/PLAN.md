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

---

### E5-1: PyCBC Waveform Setup
**Priority:** 🔴 Critical | **Estimate:** 4-5 hours

**Objective:** Implement waveform generation for BBH, BNS, and NSBH systems

**Implementation Steps:**
1. Update `src/data/injection.py`:
   ```python
   from pycbc.waveform import get_td_waveform, get_fd_waveform
   from pycbc.detector import Detector
   from pycbc.types import TimeSeries as PyCBCTimeSeries
   import numpy as np
   from dataclasses import dataclass
   from typing import Tuple, Optional
   
   @dataclass
   class WaveformParams:
       """Parameters for CBC waveform generation"""
       mass1: float          # Primary mass (solar masses)
       mass2: float          # Secondary mass (solar masses)
       spin1z: float         # Primary spin (aligned)
       spin2z: float         # Secondary spin (aligned)
       distance: float       # Luminosity distance (Mpc)
       inclination: float    # Inclination angle (rad)
       coa_phase: float      # Coalescence phase (rad)
       approximant: str      # Waveform model
       f_lower: float = 20.0 # Lower frequency cutoff
       
       @property
       def chirp_mass(self) -> float:
           """Calculate chirp mass"""
           return (self.mass1 * self.mass2)**(3/5) / (self.mass1 + self.mass2)**(1/5)
       
       @property
       def mass_ratio(self) -> float:
           """Mass ratio q = m2/m1 <= 1"""
           return min(self.mass1, self.mass2) / max(self.mass1, self.mass2)
   
   class WaveformGenerator:
       """Generate CBC waveforms using PyCBC"""
       
       def __init__(self, config):
           self.sample_rate = config['data']['sample_rate']
           self.f_lower = config.get('waveform', {}).get('f_lower', 20.0)
           
           # Approximants by source type
           self.approximants = {
               'BBH': 'IMRPhenomD',
               'BNS': 'TaylorF2',
               'NSBH': 'IMRPhenomNSBH'
           }
       
       def generate(self, params: WaveformParams) -> Tuple[np.ndarray, np.ndarray]:
           """
           Generate plus and cross polarizations
           
           Returns:
               hp: Plus polarization
               hc: Cross polarization
           """
           hp, hc = get_td_waveform(
               approximant=params.approximant,
               mass1=params.mass1,
               mass2=params.mass2,
               spin1z=params.spin1z,
               spin2z=params.spin2z,
               distance=params.distance,
               inclination=params.inclination,
               coa_phase=params.coa_phase,
               delta_t=1.0/self.sample_rate,
               f_lower=params.f_lower
           )
           
           return np.array(hp), np.array(hc)
       
       def project_to_detector(self, hp: np.ndarray, hc: np.ndarray,
                                detector: str, ra: float, dec: float,
                                polarization: float, gps_time: float) -> np.ndarray:
           """Project waveform onto detector"""
           det = Detector(detector)
           
           # Get antenna pattern
           fp, fc = det.antenna_pattern(ra, dec, polarization, gps_time)
           
           # Combine polarizations
           strain = fp * hp + fc * hc
           
           return strain
   
   class ParameterSampler:
       """Sample waveform parameters from astrophysical priors"""
       
       def __init__(self, config):
           self.config = config
           self.rng = np.random.default_rng(config.get('seed', 42))
       
       def sample_bbh(self) -> WaveformParams:
           """Sample BBH parameters"""
           # Mass range: 5-100 solar masses
           m1 = self.rng.uniform(10, 80)
           m2 = self.rng.uniform(5, m1)  # m2 <= m1
           
           # Spins: aligned, -0.99 to 0.99
           s1z = self.rng.uniform(-0.5, 0.99)
           s2z = self.rng.uniform(-0.5, 0.99)
           
           # Distance: 100-2000 Mpc (affects SNR)
           distance = self.rng.uniform(100, 1500)
           
           # Angles
           inclination = np.arccos(self.rng.uniform(-1, 1))
           coa_phase = self.rng.uniform(0, 2*np.pi)
           
           return WaveformParams(
               mass1=m1, mass2=m2,
               spin1z=s1z, spin2z=s2z,
               distance=distance,
               inclination=inclination,
               coa_phase=coa_phase,
               approximant='IMRPhenomD'
           )
       
       def sample_bns(self) -> WaveformParams:
           """Sample BNS parameters"""
           # NS mass range: 1.0-2.5 solar masses
           m1 = self.rng.uniform(1.2, 2.2)
           m2 = self.rng.uniform(1.0, m1)
           
           # NS spins are small
           s1z = self.rng.uniform(-0.05, 0.05)
           s2z = self.rng.uniform(-0.05, 0.05)
           
           # BNS visible to smaller distances
           distance = self.rng.uniform(40, 300)
           
           inclination = np.arccos(self.rng.uniform(-1, 1))
           coa_phase = self.rng.uniform(0, 2*np.pi)
           
           return WaveformParams(
               mass1=m1, mass2=m2,
               spin1z=s1z, spin2z=s2z,
               distance=distance,
               inclination=inclination,
               coa_phase=coa_phase,
               approximant='TaylorF2',
               f_lower=30.0  # BNS needs higher f_lower for long waveforms
           )
       
       def sample_nsbh(self) -> WaveformParams:
           """Sample NSBH parameters"""
           # BH: 5-50 solar masses, NS: 1-2.5
           m1 = self.rng.uniform(5, 30)  # BH
           m2 = self.rng.uniform(1.0, 2.5)  # NS
           
           # BH can have significant spin, NS small
           s1z = self.rng.uniform(-0.5, 0.99)
           s2z = self.rng.uniform(-0.05, 0.05)
           
           distance = self.rng.uniform(50, 500)
           
           inclination = np.arccos(self.rng.uniform(-1, 1))
           coa_phase = self.rng.uniform(0, 2*np.pi)
           
           return WaveformParams(
               mass1=m1, mass2=m2,
               spin1z=s1z, spin2z=s2z,
               distance=distance,
               inclination=inclination,
               coa_phase=coa_phase,
               approximant='IMRPhenomNSBH'
           )
   ```

2. Test waveform generation:
   ```python
   generator = WaveformGenerator(CONFIG)
   sampler = ParameterSampler(CONFIG)
   
   # Generate BBH
   params = sampler.sample_bbh()
   hp, hc = generator.generate(params)
   print(f"BBH waveform: {len(hp)} samples, Mc={params.chirp_mass:.1f} Msun")
   
   # Generate BNS
   params = sampler.sample_bns()
   hp, hc = generator.generate(params)
   print(f"BNS waveform: {len(hp)} samples, Mc={params.chirp_mass:.2f} Msun")
   ```

**Acceptance Criteria:**
- [ ] All three source types generate valid waveforms
- [ ] Parameter ranges match astrophysical expectations
- [ ] Waveforms have correct sample rate
- [ ] Chirp mass calculated correctly

**Files Modified:**
- `src/data/injection.py`

---

### E5-2: Noise Segment Collection
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Download background noise segments for injection

**Implementation Steps:**
1. Create noise downloader:
   ```python
   from gwpy.timeseries import TimeSeries
   from gwpy.segments import DataQualityFlag
   import numpy as np
   from pathlib import Path
   
   class NoiseCollector:
       """Collect background noise segments from GWOSC"""
       
       def __init__(self, config):
           self.sample_rate = config['data']['sample_rate']
           self.segment_duration = config.get('noise', {}).get('segment_duration', 64)
           self.output_dir = Path(config['paths'].get('noise', 'data/noise'))
           
       def find_quiet_times(self, run: str, detector: str, 
                            n_segments: int = 100) -> list:
           """
           Find times away from known events for noise collection
           
           Strategy: Use times between events, avoiding ±60s of any event
           """
           # GPS ranges for runs
           run_ranges = {
               'O1': (1126051217, 1137254417),
               'O2': (1164556817, 1187733618),
               'O3a': (1238166018, 1253977218),
               'O3b': (1256655618, 1269363618),
           }
           
           start, end = run_ranges[run]
           
           # Sample random times within run
           rng = np.random.default_rng(42)
           
           # Avoid first/last hour of run
           valid_start = start + 3600
           valid_end = end - 3600
           
           # Generate candidate times
           candidates = rng.uniform(valid_start, valid_end, size=n_segments * 3)
           
           # Filter to get n_segments valid times
           # (In production, would check against event catalog)
           selected = candidates[:n_segments]
           
           return selected.tolist()
       
       def download_noise_segment(self, gps_time: float, detector: str,
                                   run: str) -> dict:
           """Download a single noise segment"""
           try:
               t0 = gps_time - self.segment_duration / 2
               t1 = gps_time + self.segment_duration / 2
               
               strain = TimeSeries.fetch_open_data(
                   detector, t0, t1,
                   sample_rate=self.sample_rate,
                   cache=True
               )
               
               # Save
               self.output_dir.mkdir(parents=True, exist_ok=True)
               filename = f"noise_{run}_{detector}_{int(gps_time)}.hdf5"
               filepath = self.output_dir / filename
               strain.write(filepath, overwrite=True)
               
               return {
                   'status': 'success',
                   'path': str(filepath),
                   'gps_time': gps_time,
                   'detector': detector,
                   'run': run,
                   'duration': self.segment_duration
               }
               
           except Exception as e:
               return {
                   'status': 'failed',
                   'error': str(e),
                   'gps_time': gps_time
               }
       
       def collect_noise_batch(self, run: str, detector: str = 'L1',
                               n_segments: int = 100) -> list:
           """Collect batch of noise segments"""
           times = self.find_quiet_times(run, detector, n_segments)
           
           results = []
           for t in tqdm(times, desc=f"Downloading {run} {detector} noise"):
               result = self.download_noise_segment(t, detector, run)
               results.append(result)
           
           return results
   ```

2. Download noise for all runs:
   ```bash
   # Script: scripts/download_noise.py
   python scripts/download_noise.py --run O1 --n-segments 50
   python scripts/download_noise.py --run O2 --n-segments 50
   python scripts/download_noise.py --run O3a --n-segments 100
   python scripts/download_noise.py --run O3b --n-segments 100
   ```

**Expected Output:**
```
data/noise/
├── noise_O1_L1_1126259462.hdf5
├── noise_O2_L1_1187008882.hdf5
├── noise_O3a_L1_1245678901.hdf5
└── ...
```

**Acceptance Criteria:**
- [ ] 300+ noise segments downloaded
- [ ] Coverage across O1, O2, O3
- [ ] No event contamination
- [ ] Noise manifest created

**Files Created:**
- `scripts/download_noise.py`
- `data/noise/*.hdf5`
- `manifests/noise_manifest.csv`

---

### E5-3: Injection Pipeline
**Priority:** 🔴 Critical | **Estimate:** 4-5 hours

**Objective:** Inject waveforms into noise at target SNRs

**Implementation Steps:**
1. Create injection pipeline:
   ```python
   from gwpy.timeseries import TimeSeries
   import numpy as np
   from scipy.signal import resample
   
   class InjectionPipeline:
       """Inject waveforms into real detector noise"""
       
       def __init__(self, config):
           self.config = config
           self.sample_rate = config['data']['sample_rate']
           self.target_snr_range = config.get('injection', {}).get('snr_range', [8, 30])
           
       def calculate_optimal_snr(self, waveform: np.ndarray, 
                                  noise_psd: np.ndarray,
                                  df: float) -> float:
           """Calculate optimal matched-filter SNR"""
           # FFT of waveform
           wf_fft = np.fft.rfft(waveform)
           freqs = np.fft.rfftfreq(len(waveform), 1/self.sample_rate)
           
           # Interpolate PSD to waveform frequencies
           psd_interp = np.interp(freqs, np.arange(len(noise_psd)) * df, noise_psd)
           
           # SNR^2 = 4 * integral(|h(f)|^2 / S_n(f)) df
           integrand = np.abs(wf_fft)**2 / (psd_interp + 1e-50)
           snr_sq = 4 * np.sum(integrand) * (freqs[1] - freqs[0])
           
           return np.sqrt(snr_sq)
       
       def scale_to_snr(self, waveform: np.ndarray, noise: np.ndarray,
                        target_snr: float) -> np.ndarray:
           """Scale waveform to achieve target SNR"""
           # Estimate noise PSD
           from scipy.signal import welch
           freqs, psd = welch(noise, fs=self.sample_rate, nperseg=4096)
           
           # Calculate current SNR
           current_snr = self.calculate_optimal_snr(waveform, psd, freqs[1]-freqs[0])
           
           # Scale factor
           if current_snr > 0:
               scale = target_snr / current_snr
           else:
               scale = 1.0
           
           return waveform * scale
       
       def inject(self, waveform: np.ndarray, noise: TimeSeries,
                  injection_time: float, target_snr: float = None) -> TimeSeries:
           """
           Inject waveform into noise segment
           
           Args:
               waveform: Strain waveform (merger at end)
               noise: Noise TimeSeries
               injection_time: GPS time for merger
               target_snr: Target SNR (if None, use natural SNR)
           """
           noise_array = noise.value.copy()
           
           # Scale to target SNR if specified
           if target_snr is not None:
               waveform = self.scale_to_snr(waveform, noise_array, target_snr)
           
           # Find injection sample index
           t0 = noise.t0.value
           dt = 1.0 / self.sample_rate
           merger_idx = int((injection_time - t0) / dt)
           
           # Waveform ends at merger
           start_idx = merger_idx - len(waveform)
           
           if start_idx < 0:
               # Truncate waveform if it extends before noise start
               waveform = waveform[-start_idx:]
               start_idx = 0
           
           if merger_idx > len(noise_array):
               # Truncate if extends past noise end
               excess = merger_idx - len(noise_array)
               waveform = waveform[:-excess]
               merger_idx = len(noise_array)
           
           # Add waveform to noise
           end_idx = start_idx + len(waveform)
           noise_array[start_idx:end_idx] += waveform
           
           # Create new TimeSeries with injection
           injected = TimeSeries(
               noise_array,
               t0=noise.t0,
               sample_rate=noise.sample_rate,
               name=f"{noise.name}_injected"
           )
           
           return injected
       
       def create_injection(self, waveform_params: WaveformParams,
                            noise_path: str, detector: str = 'L1') -> dict:
           """Full injection pipeline"""
           # Load noise
           noise = TimeSeries.read(noise_path)
           
           # Generate waveform
           generator = WaveformGenerator(self.config)
           hp, hc = generator.generate(waveform_params)
           
           # Project to detector (simplified: just use hp)
           # In production, would properly project with sky location
           waveform = hp
           
           # Sample target SNR
           rng = np.random.default_rng()
           target_snr = rng.uniform(*self.target_snr_range)
           
           # Inject at center of noise segment
           injection_time = noise.t0.value + noise.duration.value / 2
           
           injected = self.inject(waveform, noise, injection_time, target_snr)
           
           return {
               'strain': injected,
               'params': waveform_params,
               'injection_time': injection_time,
               'target_snr': target_snr,
               'noise_file': noise_path
           }
   ```

**Acceptance Criteria:**
- [ ] Injections at specified SNR
- [ ] Waveform properly aligned with merger time
- [ ] No clipping or overflow
- [ ] SNR distribution matches target range

**Files Modified:**
- `src/data/injection.py`

---

### E5-4: Synthetic Manifest
**Priority:** 🟠 High | **Estimate:** 1-2 hours

**Objective:** Track all synthetic injections with parameters

**Implementation Steps:**
1. Create manifest structure:
   ```python
   def create_synthetic_manifest(injections: list, output_path: str):
       """Create manifest of all synthetic injections"""
       records = []
       
       for inj in injections:
           params = inj['params']
           records.append({
               'injection_id': inj['id'],
               'source_type': inj['source_type'],  # BBH, BNS, NSBH
               'label': 'BBH' if inj['source_type'] == 'BBH' else 'NS-present',
               
               # Masses
               'mass1': params.mass1,
               'mass2': params.mass2,
               'chirp_mass': params.chirp_mass,
               'mass_ratio': params.mass_ratio,
               
               # Spins
               'spin1z': params.spin1z,
               'spin2z': params.spin2z,
               
               # Extrinsic
               'distance': params.distance,
               'inclination': params.inclination,
               
               # Injection details
               'target_snr': inj['target_snr'],
               'injection_time': inj['injection_time'],
               'noise_file': inj['noise_file'],
               'noise_run': inj['noise_run'],
               
               # Output files
               'strain_file': inj['strain_path'],
               'spectrogram_file': inj.get('spectrogram_path', ''),
               'image_file': inj.get('image_path', '')
           })
       
       df = pd.DataFrame(records)
       df.to_csv(output_path, index=False)
       return df
   ```

**Expected Manifest:**
```csv
injection_id,source_type,label,mass1,mass2,chirp_mass,target_snr,...
syn_bbh_0001,BBH,BBH,35.2,28.1,27.4,15.3,...
syn_bns_0001,BNS,NS-present,1.4,1.3,1.2,12.1,...
```

**Files Created:**
- `manifests/synthetic_manifest.csv`

---

### E5-5: Generate BBH Samples
**Priority:** 🔴 Critical | **Estimate:** 6-8 hours (compute)

**Objective:** Generate 5,000 BBH injections

**Implementation Steps:**
1. Create batch generation script:
   ```python
   # scripts/generate_synthetic.py
   
   def generate_bbh_batch(n_samples: int, output_dir: str, 
                          noise_manifest: pd.DataFrame):
       """Generate batch of BBH injections"""
       
       sampler = ParameterSampler(CONFIG)
       pipeline = InjectionPipeline(CONFIG)
       
       noise_files = noise_manifest['path'].tolist()
       
       injections = []
       
       for i in tqdm(range(n_samples), desc="Generating BBH"):
           # Sample parameters
           params = sampler.sample_bbh()
           
           # Random noise file
           noise_path = np.random.choice(noise_files)
           
           # Create injection
           try:
               result = pipeline.create_injection(params, noise_path)
               
               # Save strain
               strain_path = f"{output_dir}/bbh/syn_bbh_{i:05d}.hdf5"
               result['strain'].write(strain_path, overwrite=True)
               
               injections.append({
                   'id': f'syn_bbh_{i:05d}',
                   'source_type': 'BBH',
                   'params': params,
                   'target_snr': result['target_snr'],
                   'injection_time': result['injection_time'],
                   'noise_file': noise_path,
                   'strain_path': strain_path
               })
               
           except Exception as e:
               print(f"Failed BBH {i}: {e}")
               continue
       
       return injections
   ```

2. Run generation:
   ```bash
   python scripts/generate_synthetic.py --type bbh --n-samples 5000
   ```

**Acceptance Criteria:**
- [ ] 5,000 BBH injections generated
- [ ] Mass distribution: 5-100 M☉
- [ ] SNR distribution: 8-30
- [ ] All files saved correctly

**Files Created:**
- `data/synthetic/bbh/syn_bbh_*.hdf5`

---

### E5-6: Generate NS-present Samples
**Priority:** 🔴 Critical | **Estimate:** 6-8 hours (compute)

**Objective:** Generate 5,000 NS-present injections (BNS + NSBH)

**Implementation Steps:**
1. Generate mixed BNS/NSBH:
   ```python
   def generate_ns_present_batch(n_samples: int, output_dir: str,
                                  noise_manifest: pd.DataFrame):
       """Generate BNS and NSBH injections"""
       
       # Split: 60% BNS, 40% NSBH
       n_bns = int(n_samples * 0.6)
       n_nsbh = n_samples - n_bns
       
       sampler = ParameterSampler(CONFIG)
       pipeline = InjectionPipeline(CONFIG)
       
       injections = []
       
       # Generate BNS
       for i in tqdm(range(n_bns), desc="Generating BNS"):
           params = sampler.sample_bns()
           # ... same injection logic ...
           injections.append({
               'id': f'syn_bns_{i:05d}',
               'source_type': 'BNS',
               ...
           })
       
       # Generate NSBH
       for i in tqdm(range(n_nsbh), desc="Generating NSBH"):
           params = sampler.sample_nsbh()
           # ... same injection logic ...
           injections.append({
               'id': f'syn_nsbh_{i:05d}',
               'source_type': 'NSBH',
               ...
           })
       
       return injections
   ```

2. Run generation:
   ```bash
   python scripts/generate_synthetic.py --type ns_present --n-samples 5000
   ```

**Acceptance Criteria:**
- [ ] 3,000 BNS + 2,000 NSBH generated
- [ ] BNS masses: 1-2.5 M☉
- [ ] NSBH masses: BH 5-50, NS 1-2.5 M☉
- [ ] All files saved correctly

**Files Created:**
- `data/synthetic/ns_present/syn_bns_*.hdf5`
- `data/synthetic/ns_present/syn_nsbh_*.hdf5`

---

### E5-7: Synthetic Spectrograms
**Priority:** 🔴 Critical | **Estimate:** 4-6 hours (compute)

**Objective:** Process all synthetic injections to spectrograms

**Implementation Steps:**
1. Batch processing:
   ```python
   def process_synthetic_spectrograms(manifest_path: str):
       """Generate spectrograms for all synthetic data"""
       
       manifest = pd.read_csv(manifest_path)
       preprocessor = StrainPreprocessor(CONFIG)
       generator = SpectrogramGenerator(CONFIG)
       
       for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
           try:
               # Load strain
               strain = TimeSeries.read(row['strain_file'])
               
               # Preprocess
               clean = preprocessor.preprocess(strain)
               
               # Generate spectrogram
               specgram = generator.generate(clean, row['injection_time'])
               
               # Save
               base_name = Path(row['strain_file']).stem
               matrix_path = f"data/spectrograms/matrices/{base_name}.npy"
               image_path = f"data/spectrograms/images/{base_name}.png"
               
               generator.save_outputs(specgram, base_name, 'data/spectrograms')
               
               # Update manifest
               manifest.loc[_, 'spectrogram_file'] = matrix_path
               manifest.loc[_, 'image_file'] = image_path
               
           except Exception as e:
               print(f"Failed {row['injection_id']}: {e}")
       
       # Save updated manifest
       manifest.to_csv(manifest_path, index=False)
   ```

**Acceptance Criteria:**
- [ ] 10,000 spectrograms generated
- [ ] Both matrix and image formats
- [ ] Manifest updated with paths
- [ ] Processing time ~2-4 hours

**Files Created:**
- `data/spectrograms/matrices/syn_*.npy`
- `data/spectrograms/images/syn_*.png`

---

### E5-8: Quality Validation
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Validate synthetic data quality

**Implementation Steps:**
1. Create validation notebook:
   ```python
   # notebooks/03_synthetic_injection.ipynb
   
   # Cell 1: Load manifest
   manifest = pd.read_csv('../manifests/synthetic_manifest.csv')
   print(f"Total synthetic: {len(manifest)}")
   print(manifest['source_type'].value_counts())
   
   # Cell 2: SNR distribution
   fig, ax = plt.subplots(1, 2, figsize=(12, 4))
   
   for label in ['BBH', 'NS-present']:
       data = manifest[manifest['label']==label]['target_snr']
       ax[0].hist(data, alpha=0.5, label=label, bins=30)
   ax[0].set_xlabel('Target SNR')
   ax[0].legend()
   
   # Cell 3: Mass distributions
   ax[1].scatter(manifest['mass1'], manifest['mass2'], 
                 c=manifest['label'].map({'BBH': 'blue', 'NS-present': 'red'}),
                 alpha=0.3, s=5)
   ax[1].set_xlabel('Mass 1 (M☉)')
   ax[1].set_ylabel('Mass 2 (M☉)')
   
   # Cell 4: Visual inspection of random samples
   sample_ids = manifest.sample(9)['injection_id'].tolist()
   
   fig, axes = plt.subplots(3, 3, figsize=(12, 12))
   for ax, inj_id in zip(axes.flat, sample_ids):
       img = plt.imread(f"../data/spectrograms/images/{inj_id}.png")
       ax.imshow(img)
       ax.set_title(inj_id[:12])
       ax.axis('off')
   
   # Cell 5: Compare to real events
   # Side-by-side: Real GW150914 vs Synthetic BBH
   ```

**Acceptance Criteria:**
- [ ] SNR distribution uniform in [8, 30]
- [ ] Mass distributions match priors
- [ ] Spectrograms show clear chirps
- [ ] No obvious artifacts

**Files Created:**
- `notebooks/03_synthetic_injection.ipynb`
- `outputs/figures/synthetic_distributions.png`

---

## Epic 6: CNN Model (Phase 2)
**Milestone:** M4  
**Goal:** Train deep learning classifier on synthetic + real data

---

### E6-1: PyTorch Dataset
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Implement efficient PyTorch Dataset for spectrograms

**Implementation Steps:**
1. Update `src/data/dataset.py`:
   ```python
   import torch
   from torch.utils.data import Dataset, DataLoader
   from PIL import Image
   import numpy as np
   import pandas as pd
   from pathlib import Path
   from typing import Optional, Callable, Tuple
   
   class SpectrogramDataset(Dataset):
       """PyTorch Dataset for spectrogram images"""
       
       def __init__(self, manifest_path: str, 
                    image_dir: str,
                    transform: Optional[Callable] = None,
                    label_col: str = 'label',
                    split: str = None):
           """
           Args:
               manifest_path: Path to CSV manifest
               image_dir: Directory containing spectrogram images
               transform: Optional torchvision transforms
               label_col: Column name for labels
               split: Filter to specific split ('train', 'val', 'test')
           """
           self.manifest = pd.read_csv(manifest_path)
           self.image_dir = Path(image_dir)
           self.transform = transform
           self.label_col = label_col
           
           # Filter by split if specified
           if split and 'split' in self.manifest.columns:
               self.manifest = self.manifest[self.manifest['split'] == split]
           
           # Create label mapping
           self.label_map = {'BBH': 0, 'NS-present': 1}
           
           # Get file paths
           self.samples = []
           for _, row in self.manifest.iterrows():
               if 'image_file' in row:
                   img_path = row['image_file']
               else:
                   # Construct from ID
                   img_name = row.get('event_name', row.get('injection_id'))
                   img_path = self.image_dir / f"{img_name}.png"
               
               label = self.label_map[row[label_col]]
               self.samples.append((img_path, label))
       
       def __len__(self) -> int:
           return len(self.samples)
       
       def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
           img_path, label = self.samples[idx]
           
           # Load image
           image = Image.open(img_path).convert('RGB')
           
           # Apply transforms
           if self.transform:
               image = self.transform(image)
           else:
               # Default: convert to tensor
               image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
           
           return image, label
       
       def get_class_weights(self) -> torch.Tensor:
           """Calculate class weights for imbalanced data"""
           labels = [s[1] for s in self.samples]
           class_counts = np.bincount(labels)
           weights = 1.0 / class_counts
           weights = weights / weights.sum() * len(class_counts)
           return torch.FloatTensor(weights)
   
   
   class CombinedDataset(Dataset):
       """Combine real and synthetic datasets"""
       
       def __init__(self, real_manifest: str, synthetic_manifest: str,
                    image_dir: str, transform=None,
                    real_weight: float = 1.0):
           """
           Args:
               real_manifest: Path to real events manifest
               synthetic_manifest: Path to synthetic manifest
               image_dir: Directory for images
               transform: Transforms to apply
               real_weight: Upsampling weight for real events
           """
           self.real_dataset = SpectrogramDataset(real_manifest, image_dir, transform)
           self.synthetic_dataset = SpectrogramDataset(synthetic_manifest, image_dir, transform)
           
           # Upsample real events
           self.real_indices = list(range(len(self.real_dataset)))
           if real_weight > 1.0:
               self.real_indices = self.real_indices * int(real_weight)
           
           self.synthetic_indices = list(range(len(self.synthetic_dataset)))
       
       def __len__(self):
           return len(self.real_indices) + len(self.synthetic_indices)
       
       def __getitem__(self, idx):
           if idx < len(self.real_indices):
               real_idx = self.real_indices[idx]
               return self.real_dataset[real_idx]
           else:
               syn_idx = idx - len(self.real_indices)
               return self.synthetic_dataset[syn_idx]
   ```

2. Create data module for training:
   ```python
   from torchvision import transforms
   
   def get_transforms(train: bool = True, img_size: int = 224):
       """Get transforms for training/validation"""
       
       if train:
           return transforms.Compose([
               transforms.Resize((img_size, img_size)),
               transforms.RandomHorizontalFlip(p=0.5),  # Time reversal symmetry
               transforms.ColorJitter(brightness=0.1, contrast=0.1),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
           ])
       else:
           return transforms.Compose([
               transforms.Resize((img_size, img_size)),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
           ])
   
   def create_dataloaders(config, batch_size=64, num_workers=4):
       """Create train/val/test dataloaders"""
       
       train_transform = get_transforms(train=True)
       val_transform = get_transforms(train=False)
       
       # Training: synthetic + real
       train_dataset = CombinedDataset(
           real_manifest='manifests/event_manifest.csv',
           synthetic_manifest='manifests/synthetic_manifest.csv',
           image_dir='data/spectrograms/images',
           transform=train_transform,
           real_weight=10.0  # Upsample real events 10x
       )
       
       # Validation: real only (or synthetic subset)
       val_dataset = SpectrogramDataset(
           'manifests/event_manifest.csv',
           'data/spectrograms/images',
           transform=val_transform,
           split='val'
       )
       
       train_loader = DataLoader(
           train_dataset,
           batch_size=batch_size,
           shuffle=True,
           num_workers=num_workers,
           pin_memory=True
       )
       
       val_loader = DataLoader(
           val_dataset,
           batch_size=batch_size,
           shuffle=False,
           num_workers=num_workers,
           pin_memory=True
       )
       
       return train_loader, val_loader
   ```

**Acceptance Criteria:**
- [ ] Dataset loads images correctly
- [ ] Labels mapped to 0/1
- [ ] Class weights calculated
- [ ] DataLoader with proper batching

**Files Modified:**
- `src/data/dataset.py`

---

### E6-2: CNN Architecture
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Implement ResNet-based classifier

**Implementation Steps:**
1. Update `src/models/cnn.py`:
   ```python
   import torch
   import torch.nn as nn
   import torchvision.models as models
   from typing import Optional, Tuple
   
   class SpectrogramCNN(nn.Module):
       """CNN classifier for spectrogram images"""
       
       def __init__(self, 
                    backbone: str = 'resnet18',
                    num_classes: int = 2,
                    pretrained: bool = True,
                    dropout: float = 0.5):
           super().__init__()
           
           self.backbone_name = backbone
           self.num_classes = num_classes
           
           # Load pretrained backbone
           if backbone == 'resnet18':
               self.backbone = models.resnet18(
                   weights='IMAGENET1K_V1' if pretrained else None
               )
               in_features = self.backbone.fc.in_features  # 512
               
           elif backbone == 'resnet50':
               self.backbone = models.resnet50(
                   weights='IMAGENET1K_V1' if pretrained else None
               )
               in_features = self.backbone.fc.in_features  # 2048
               
           elif backbone == 'efficientnet_b0':
               self.backbone = models.efficientnet_b0(
                   weights='IMAGENET1K_V1' if pretrained else None
               )
               in_features = self.backbone.classifier[1].in_features
               self.backbone.classifier = nn.Identity()
           
           else:
               raise ValueError(f"Unknown backbone: {backbone}")
           
           # Replace final layer
           if 'resnet' in backbone:
               self.backbone.fc = nn.Identity()
           
           # Classification head
           self.classifier = nn.Sequential(
               nn.Dropout(dropout),
               nn.Linear(in_features, 256),
               nn.ReLU(),
               nn.Dropout(dropout/2),
               nn.Linear(256, num_classes)
           )
           
           # Feature extractor (for physics-informed)
           self.feature_dim = in_features
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass"""
           features = self.backbone(x)
           logits = self.classifier(features)
           return logits
       
       def extract_features(self, x: torch.Tensor) -> torch.Tensor:
           """Extract features before classification head"""
           return self.backbone(x)
       
       def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
           """Get class probabilities"""
           logits = self.forward(x)
           return torch.softmax(logits, dim=1)
   
   
   class SpectrogramCNNWithMc(nn.Module):
       """CNN with chirp mass regression head for physics-informed training"""
       
       def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
           super().__init__()
           
           # Shared backbone
           self.base_cnn = SpectrogramCNN(backbone, num_classes=2, pretrained=pretrained)
           
           # Chirp mass regression head
           self.mc_head = nn.Sequential(
               nn.Linear(self.base_cnn.feature_dim, 128),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(128, 1),
               nn.Softplus()  # Ensure positive output
           )
       
       def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
           """
           Returns:
               logits: Classification logits [B, 2]
               mc_pred: Chirp mass predictions [B, 1]
           """
           features = self.base_cnn.extract_features(x)
           logits = self.base_cnn.classifier(features)
           mc_pred = self.mc_head(features)
           return logits, mc_pred
   ```

2. Test model:
   ```python
   model = SpectrogramCNN(backbone='resnet18', num_classes=2)
   x = torch.randn(4, 3, 224, 224)
   out = model(x)
   print(f"Output shape: {out.shape}")  # [4, 2]
   
   # Count parameters
   params = sum(p.numel() for p in model.parameters())
   print(f"Parameters: {params/1e6:.1f}M")  # ~11.2M for ResNet18
   ```

**Acceptance Criteria:**
- [ ] ResNet18 and ResNet50 supported
- [ ] Pretrained weights load correctly
- [ ] Output shape correct [B, 2]
- [ ] Feature extraction works

**Files Modified:**
- `src/models/cnn.py`

---

### E6-3: Data Augmentation
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Implement physics-preserving augmentations (ADR-014)

**Implementation Steps:**
1. Create custom augmentations:
   ```python
   import torch
   import numpy as np
   from torchvision import transforms
   import random
   
   class TimeJitter(object):
       """Random time shift (crop from different position)"""
       
       def __init__(self, max_shift: float = 0.1):
           self.max_shift = max_shift  # Fraction of image width
       
       def __call__(self, img):
           if random.random() < 0.5:
               return img
           
           w = img.size[0] if hasattr(img, 'size') else img.shape[-1]
           shift = int(w * random.uniform(-self.max_shift, self.max_shift))
           
           # Roll the image horizontally
           if isinstance(img, torch.Tensor):
               return torch.roll(img, shifts=shift, dims=-1)
           else:
               img_array = np.array(img)
               img_array = np.roll(img_array, shift, axis=1)
               return Image.fromarray(img_array)
   
   
   class GaussianNoise(object):
       """Add Gaussian noise to simulate different SNRs"""
       
       def __init__(self, std_range: tuple = (0.01, 0.1)):
           self.std_range = std_range
       
       def __call__(self, img):
           if random.random() < 0.5:
               return img
           
           std = random.uniform(*self.std_range)
           
           if isinstance(img, torch.Tensor):
               noise = torch.randn_like(img) * std
               return torch.clamp(img + noise, 0, 1)
           else:
               img_array = np.array(img).astype(np.float32) / 255.0
               noise = np.random.randn(*img_array.shape) * std
               img_array = np.clip(img_array + noise, 0, 1)
               return Image.fromarray((img_array * 255).astype(np.uint8))
   
   
   class FrequencyMask(object):
       """Mask random frequency band (SpecAugment-style)"""
       
       def __init__(self, max_mask_pct: float = 0.1):
           self.max_mask_pct = max_mask_pct
       
       def __call__(self, img):
           if random.random() < 0.5:
               return img
           
           if isinstance(img, torch.Tensor):
               h = img.shape[-2]
               mask_size = int(h * random.uniform(0, self.max_mask_pct))
               start = random.randint(0, h - mask_size)
               img_copy = img.clone()
               img_copy[..., start:start+mask_size, :] = 0
               return img_copy
           else:
               img_array = np.array(img)
               h = img_array.shape[0]
               mask_size = int(h * random.uniform(0, self.max_mask_pct))
               start = random.randint(0, h - mask_size)
               img_array[start:start+mask_size, :] = 0
               return Image.fromarray(img_array)
   
   
   def get_augmentation_transforms(img_size: int = 224):
       """Get full augmentation pipeline"""
       return transforms.Compose([
           transforms.Resize((img_size, img_size)),
           transforms.RandomHorizontalFlip(p=0.5),  # Time reversal
           TimeJitter(max_shift=0.05),
           GaussianNoise(std_range=(0.01, 0.05)),
           FrequencyMask(max_mask_pct=0.1),
           transforms.ColorJitter(brightness=0.1, contrast=0.1),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
       ])
   ```

**Acceptance Criteria:**
- [ ] Time jitter preserves chirp structure
- [ ] Noise injection varies SNR
- [ ] Frequency masking doesn't destroy signal
- [ ] Augmentations applied during training only

**Files Modified:**
- `src/data/dataset.py` (add augmentation classes)

---

### E6-4: Colab Training Notebook
**Priority:** 🔴 Critical | **Estimate:** 4-5 hours

**Objective:** A100-optimized training notebook for Google Colab

**Implementation Steps:**
1. Create `notebooks/07_phase2_cnn.ipynb`:
   ```python
   # Cell 1: Setup and GPU check
   !nvidia-smi
   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   
   # Cell 2: Clone repo and install dependencies
   !git clone https://github.com/Arctic-XD/gw-event-classification.git
   %cd gw-event-classification
   !pip install -r requirements.txt
   
   # Cell 3: Mount Google Drive for data
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copy data from Drive (assumes pre-uploaded)
   !cp -r /content/drive/MyDrive/gw-data/* data/
   
   # Cell 4: Configuration for A100
   CONFIG = {
       'batch_size': 128,      # A100 can handle 256+ but 128 is safe
       'learning_rate': 1e-3,
       'weight_decay': 1e-4,
       'epochs': 50,
       'backbone': 'resnet50',  # Can use larger model on A100
       'num_workers': 4,
       'use_amp': True,         # Mixed precision
       'accumulation_steps': 1, # No accumulation needed with large batch
   }
   
   # Cell 5: Create dataloaders
   from src.data.dataset import create_dataloaders, get_augmentation_transforms
   
   train_loader, val_loader = create_dataloaders(
       CONFIG,
       batch_size=CONFIG['batch_size'],
       num_workers=CONFIG['num_workers']
   )
   
   print(f"Train batches: {len(train_loader)}")
   print(f"Val batches: {len(val_loader)}")
   
   # Cell 6: Initialize model
   from src.models.cnn import SpectrogramCNN
   
   model = SpectrogramCNN(
       backbone=CONFIG['backbone'],
       num_classes=2,
       pretrained=True,
       dropout=0.5
   )
   model = model.cuda()
   
   # Cell 7: Loss, optimizer, scheduler
   criterion = torch.nn.CrossEntropyLoss(
       weight=train_loader.dataset.get_class_weights().cuda()
   )
   
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=CONFIG['learning_rate'],
       weight_decay=CONFIG['weight_decay']
   )
   
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
       optimizer, T_max=CONFIG['epochs']
   )
   
   # Cell 8: Training loop with AMP
   from torch.cuda.amp import autocast, GradScaler
   from tqdm.notebook import tqdm
   
   scaler = GradScaler()
   best_val_acc = 0
   
   for epoch in range(CONFIG['epochs']):
       model.train()
       train_loss = 0
       
       pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
       for batch_idx, (images, labels) in enumerate(pbar):
           images, labels = images.cuda(), labels.cuda()
           
           optimizer.zero_grad()
           
           with autocast():
               outputs = model(images)
               loss = criterion(outputs, labels)
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()
           
           train_loss += loss.item()
           pbar.set_postfix({'loss': loss.item()})
       
       # Validation
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           for images, labels in val_loader:
               images, labels = images.cuda(), labels.cuda()
               outputs = model(images)
               _, predicted = outputs.max(1)
               total += labels.size(0)
               correct += predicted.eq(labels).sum().item()
       
       val_acc = correct / total
       print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
       
       # Save best model
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save(model.state_dict(), 'outputs/models/phase2/best_cnn.pth')
       
       scheduler.step()
   
   # Cell 9: Copy results to Drive
   !cp -r outputs/ /content/drive/MyDrive/gw-outputs/
   ```

**Acceptance Criteria:**
- [ ] Notebook runs on Colab A100
- [ ] Mixed precision training works
- [ ] Batch size 128+ without OOM
- [ ] Model saves to Drive

**Files Created:**
- `notebooks/07_phase2_cnn.ipynb`

---

### E6-5: Training Loop
**Priority:** 🔴 Critical | **Estimate:** 3-4 hours

**Objective:** Implement robust training loop with all features

**Implementation Steps:**
1. Update `src/training/trainer.py`:
   ```python
   import torch
   import torch.nn as nn
   from torch.cuda.amp import autocast, GradScaler
   from torch.utils.data import DataLoader
   from typing import Dict, Optional, Callable
   import logging
   from pathlib import Path
   from tqdm import tqdm
   import json
   
   class Trainer:
       """Training loop with AMP, gradient accumulation, and callbacks"""
       
       def __init__(self,
                    model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    config: dict = None,
                    device: str = 'cuda'):
           
           self.model = model.to(device)
           self.train_loader = train_loader
           self.val_loader = val_loader
           self.criterion = criterion
           self.optimizer = optimizer
           self.scheduler = scheduler
           self.config = config or {}
           self.device = device
           
           # Mixed precision
           self.use_amp = self.config.get('use_amp', True)
           self.scaler = GradScaler() if self.use_amp else None
           
           # Gradient accumulation
           self.accumulation_steps = self.config.get('accumulation_steps', 1)
           
           # Logging
           self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_recall': []}
           self.best_metric = 0
           
           # Output directory
           self.output_dir = Path(self.config.get('output_dir', 'outputs/models/phase2'))
           self.output_dir.mkdir(parents=True, exist_ok=True)
       
       def train_epoch(self) -> float:
           """Train for one epoch"""
           self.model.train()
           total_loss = 0
           
           pbar = tqdm(self.train_loader, desc="Training")
           
           for batch_idx, (images, labels) in enumerate(pbar):
               images = images.to(self.device)
               labels = labels.to(self.device)
               
               # Forward pass with AMP
               with autocast(enabled=self.use_amp):
                   outputs = self.model(images)
                   loss = self.criterion(outputs, labels)
                   loss = loss / self.accumulation_steps
               
               # Backward pass
               if self.use_amp:
                   self.scaler.scale(loss).backward()
               else:
                   loss.backward()
               
               # Gradient accumulation
               if (batch_idx + 1) % self.accumulation_steps == 0:
                   if self.use_amp:
                       self.scaler.step(self.optimizer)
                       self.scaler.update()
                   else:
                       self.optimizer.step()
                   self.optimizer.zero_grad()
               
               total_loss += loss.item() * self.accumulation_steps
               pbar.set_postfix({'loss': loss.item() * self.accumulation_steps})
           
           return total_loss / len(self.train_loader)
       
       @torch.no_grad()
       def validate(self) -> Dict[str, float]:
           """Validation pass"""
           self.model.eval()
           total_loss = 0
           all_preds = []
           all_labels = []
           
           for images, labels in tqdm(self.val_loader, desc="Validation"):
               images = images.to(self.device)
               labels = labels.to(self.device)
               
               outputs = self.model(images)
               loss = self.criterion(outputs, labels)
               
               total_loss += loss.item()
               preds = outputs.argmax(dim=1)
               all_preds.extend(preds.cpu().numpy())
               all_labels.extend(labels.cpu().numpy())
           
           # Metrics
           all_preds = np.array(all_preds)
           all_labels = np.array(all_labels)
           
           accuracy = (all_preds == all_labels).mean()
           
           # NS-present recall (class 1)
           ns_mask = all_labels == 1
           recall = all_preds[ns_mask].sum() / ns_mask.sum() if ns_mask.sum() > 0 else 0
           
           return {
               'loss': total_loss / len(self.val_loader),
               'accuracy': accuracy,
               'recall_ns': recall
           }
       
       def train(self, epochs: int, early_stopping: int = 10):
           """Full training loop"""
           patience_counter = 0
           
           for epoch in range(epochs):
               print(f"\nEpoch {epoch+1}/{epochs}")
               
               # Train
               train_loss = self.train_epoch()
               
               # Validate
               val_metrics = self.validate()
               
               # Update scheduler
               if self.scheduler:
                   self.scheduler.step()
               
               # Log
               self.history['train_loss'].append(train_loss)
               self.history['val_loss'].append(val_metrics['loss'])
               self.history['val_acc'].append(val_metrics['accuracy'])
               self.history['val_recall'].append(val_metrics['recall_ns'])
               
               print(f"Train Loss: {train_loss:.4f}")
               print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, NS Recall: {val_metrics['recall_ns']:.4f}")
               
               # Save best model (by recall)
               if val_metrics['recall_ns'] > self.best_metric:
                   self.best_metric = val_metrics['recall_ns']
                   self.save_checkpoint('best_model.pth')
                   patience_counter = 0
               else:
                   patience_counter += 1
               
               # Early stopping
               if patience_counter >= early_stopping:
                   print(f"Early stopping at epoch {epoch+1}")
                   break
           
           # Save final model
           self.save_checkpoint('final_model.pth')
           self.save_history()
           
           return self.history
       
       def save_checkpoint(self, filename: str):
           """Save model checkpoint"""
           path = self.output_dir / filename
           torch.save({
               'model_state_dict': self.model.state_dict(),
               'optimizer_state_dict': self.optimizer.state_dict(),
               'best_metric': self.best_metric,
               'config': self.config
           }, path)
           print(f"Saved checkpoint: {path}")
       
       def save_history(self):
           """Save training history"""
           path = self.output_dir / 'training_history.json'
           with open(path, 'w') as f:
               json.dump(self.history, f, indent=2)
   ```

**Acceptance Criteria:**
- [ ] AMP working correctly
- [ ] Gradient accumulation working
- [ ] Early stopping implemented
- [ ] Checkpoints saved properly

**Files Modified:**
- `src/training/trainer.py`

---

### E6-6: Hyperparameter Tuning
**Priority:** 🟠 High | **Estimate:** 4-6 hours (compute)

**Objective:** Find optimal hyperparameters

**Implementation Steps:**
1. Grid search configuration:
   ```python
   HP_SEARCH_SPACE = {
       'learning_rate': [1e-4, 5e-4, 1e-3],
       'weight_decay': [1e-5, 1e-4, 1e-3],
       'batch_size': [64, 128, 256],
       'backbone': ['resnet18', 'resnet50'],
       'dropout': [0.3, 0.5, 0.7]
   }
   
   def run_hp_search(search_space, n_trials=20):
       """Random search over hyperparameters"""
       results = []
       
       for trial in range(n_trials):
           # Sample hyperparameters
           config = {
               k: random.choice(v) for k, v in search_space.items()
           }
           
           # Train model
           model = SpectrogramCNN(backbone=config['backbone'], dropout=config['dropout'])
           trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config=config)
           history = trainer.train(epochs=20, early_stopping=5)
           
           results.append({
               **config,
               'best_recall': max(history['val_recall']),
               'best_acc': max(history['val_acc'])
           })
       
       return pd.DataFrame(results)
   ```

**Expected Best Config:**
```python
BEST_CONFIG = {
    'backbone': 'resnet50',
    'batch_size': 128,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'dropout': 0.5,
    'epochs': 50
}
```

**Files Created:**
- `outputs/results/hp_search_results.csv`

---

### E6-7: Model Checkpointing
**Priority:** 🟠 High | **Estimate:** 1-2 hours

**Objective:** Robust checkpoint saving and loading

**Implementation Steps:**
1. Add to trainer:
   ```python
   def load_checkpoint(self, path: str):
       """Load model from checkpoint"""
       checkpoint = torch.load(path)
       self.model.load_state_dict(checkpoint['model_state_dict'])
       self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       self.best_metric = checkpoint.get('best_metric', 0)
       return checkpoint.get('config', {})
   ```

**Files Modified:**
- `src/training/trainer.py`

---

### E6-8: CNN Evaluation
**Priority:** 🔴 Critical | **Estimate:** 2-3 hours

**Objective:** Evaluate CNN on real events and compare to baseline

**Implementation Steps:**
1. Evaluation script:
   ```python
   def evaluate_cnn(model_path: str, test_manifest: str):
       """Evaluate CNN on test set"""
       
       # Load model
       model = SpectrogramCNN(backbone='resnet50')
       model.load_state_dict(torch.load(model_path)['model_state_dict'])
       model = model.cuda().eval()
       
       # Load test data
       test_dataset = SpectrogramDataset(test_manifest, 'data/spectrograms/images',
                                         transform=get_transforms(train=False))
       test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
       
       # Predictions
       all_preds = []
       all_probs = []
       all_labels = []
       
       with torch.no_grad():
           for images, labels in test_loader:
               images = images.cuda()
               outputs = model(images)
               probs = torch.softmax(outputs, dim=1)
               preds = outputs.argmax(dim=1)
               
               all_preds.extend(preds.cpu().numpy())
               all_probs.extend(probs[:, 1].cpu().numpy())
               all_labels.extend(labels.numpy())
       
       # Metrics
       from sklearn.metrics import classification_report, roc_auc_score
       
       print("CNN Evaluation on Real Events:")
       print(classification_report(all_labels, all_preds, target_names=['BBH', 'NS-present']))
       print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")
       
       return all_preds, all_probs, all_labels
   ```

2. Compare to Phase 1 baseline:
   ```python
   # Load Phase 1 results
   phase1 = pd.read_csv('outputs/results/phase1_results.csv')
   
   # CNN results
   cnn_metrics = evaluate_cnn('outputs/models/phase2/best_model.pth', 
                              'manifests/event_manifest.csv')
   
   # Comparison table
   comparison = pd.DataFrame({
       'Model': ['Random Forest', 'XGBoost', 'CNN (ResNet50)'],
       'Accuracy': [phase1_rf_acc, phase1_xgb_acc, cnn_acc],
       'NS Recall': [phase1_rf_recall, phase1_xgb_recall, cnn_recall],
       'ROC-AUC': [phase1_rf_auc, phase1_xgb_auc, cnn_auc]
   })
   ```

**Acceptance Criteria:**
- [ ] CNN evaluated on real events
- [ ] Comparison table generated
- [ ] CNN improves over baseline (expected)
- [ ] Results saved

**Files Created:**
- `outputs/results/phase2_cnn_results.csv`
- `outputs/figures/cnn_confusion_matrix.png`

---

## Epic 7: Physics-Informed Model (Phase 2)
**Milestone:** M4  
**Goal:** Add physics constraints to improve classification

---

### E7-1: Physics Loss Function
**Priority:** 🟠 High | **Estimate:** 3-4 hours

**Objective:** Implement chirp mass constraint loss per ADR-013

**Implementation Steps:**
1. Update `src/models/physics_loss.py`:
   ```python
   import torch
   import torch.nn as nn
   import numpy as np
   from typing import Tuple, Optional
   
   # Physical constants
   G = 6.67430e-11  # m^3 kg^-1 s^-2
   c = 299792458.0  # m/s
   M_SUN = 1.98847e30  # kg
   
   class ChirpMassLoss(nn.Module):
       """Physics-informed loss based on chirp mass estimation"""
       
       def __init__(self, mc_threshold: float = 2.5, margin: float = 0.5):
           """
           Args:
               mc_threshold: Chirp mass threshold between NS-present and BBH
                            NS-present: Mc < 2.5 M☉, BBH: Mc > 2.5 M☉
               margin: Soft margin for transition region
           """
           super().__init__()
           self.mc_threshold = mc_threshold
           self.margin = margin
       
       def forward(self, mc_pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
           """
           Compute physics constraint loss.
           
           Args:
               mc_pred: Predicted chirp mass [B, 1]
               labels: Ground truth labels (0=BBH, 1=NS-present) [B]
           
           Returns:
               Physics loss scalar
           """
           mc_pred = mc_pred.squeeze(-1)  # [B]
           
           # For NS-present (label=1): penalize if Mc > threshold
           # For BBH (label=0): penalize if Mc < threshold
           
           ns_mask = labels == 1
           bbh_mask = labels == 0
           
           loss = torch.tensor(0.0, device=mc_pred.device)
           
           # NS-present: Mc should be < threshold
           if ns_mask.sum() > 0:
               ns_violation = torch.relu(mc_pred[ns_mask] - self.mc_threshold + self.margin)
               loss = loss + ns_violation.mean()
           
           # BBH: Mc should be > threshold
           if bbh_mask.sum() > 0:
               bbh_violation = torch.relu(self.mc_threshold + self.margin - mc_pred[bbh_mask])
               loss = loss + bbh_violation.mean()
           
           return loss
   
   
   class ChirpMassRegressionLoss(nn.Module):
       """Direct chirp mass regression loss (when true Mc is available)"""
       
       def __init__(self, log_scale: bool = True):
           super().__init__()
           self.log_scale = log_scale
           self.mse = nn.MSELoss()
       
       def forward(self, mc_pred: torch.Tensor, mc_true: torch.Tensor) -> torch.Tensor:
           """
           Args:
               mc_pred: Predicted chirp mass [B, 1]
               mc_true: True chirp mass [B] (in solar masses)
           """
           mc_pred = mc_pred.squeeze(-1)
           
           if self.log_scale:
               # Log scale to handle wide range (1-100 M☉)
               return self.mse(torch.log(mc_pred + 1e-6), torch.log(mc_true + 1e-6))
           else:
               return self.mse(mc_pred, mc_true)
   
   
   def estimate_chirp_mass_from_f0_fdot(f0: float, fdot: float) -> float:
       """
       Estimate chirp mass from initial frequency and frequency derivative.
       
       Using: f_dot = (96/5) * pi^(8/3) * (G*Mc/c^3)^(5/3) * f^(11/3)
       
       Args:
           f0: Initial GW frequency (Hz)
           fdot: Frequency derivative (Hz/s)
       
       Returns:
           Chirp mass in solar masses
       """
       if fdot <= 0 or f0 <= 0:
           return np.nan
       
       # Solve for Mc
       coeff = (96/5) * np.pi**(8/3)
       mc_si = ((fdot / coeff) * f0**(-11/3))**(3/5) * c**3 / G
       mc_solar = mc_si / M_SUN
       
       return mc_solar
   ```

**Acceptance Criteria:**
- [ ] ChirpMassLoss penalizes physics violations
- [ ] Threshold at 2.5 M☉ works correctly
- [ ] Loss differentiable for backprop

**Files Modified:**
- `src/models/physics_loss.py`

---

### E7-2: Multi-task Head
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Add chirp mass regression head to CNN

**Implementation Steps:**
1. The `SpectrogramCNNWithMc` class was already defined in E6-2. Enhance it:
   ```python
   class PhysicsInformedCNN(nn.Module):
       """CNN with physics-informed multi-task heads"""
       
       def __init__(self, 
                    backbone: str = 'resnet50',
                    pretrained: bool = True,
                    dropout: float = 0.5):
           super().__init__()
           
           # Shared backbone
           self.backbone = self._build_backbone(backbone, pretrained)
           self.feature_dim = self._get_feature_dim(backbone)
           
           # Classification head
           self.classifier = nn.Sequential(
               nn.Dropout(dropout),
               nn.Linear(self.feature_dim, 256),
               nn.ReLU(),
               nn.Dropout(dropout/2),
               nn.Linear(256, 2)
           )
           
           # Chirp mass regression head
           self.mc_regressor = nn.Sequential(
               nn.Dropout(dropout),
               nn.Linear(self.feature_dim, 128),
               nn.ReLU(),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, 1),
               nn.Softplus()  # Ensure positive Mc
           )
           
           # Optional: frequency evolution head
           self.freq_regressor = nn.Sequential(
               nn.Linear(self.feature_dim, 64),
               nn.ReLU(),
               nn.Linear(64, 2)  # [f0, fdot]
           )
       
       def _build_backbone(self, name, pretrained):
           import torchvision.models as models
           if name == 'resnet50':
               model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
               model.fc = nn.Identity()
               return model
           elif name == 'resnet18':
               model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
               model.fc = nn.Identity()
               return model
           raise ValueError(f"Unknown backbone: {name}")
       
       def _get_feature_dim(self, name):
           dims = {'resnet18': 512, 'resnet50': 2048}
           return dims.get(name, 2048)
       
       def forward(self, x, return_features=False):
           """Forward pass with multi-task outputs"""
           features = self.backbone(x)
           
           logits = self.classifier(features)
           mc_pred = self.mc_regressor(features)
           
           if return_features:
               return logits, mc_pred, features
           return logits, mc_pred
       
       def predict_proba(self, x):
           """Get classification probabilities"""
           logits, _ = self.forward(x)
           return torch.softmax(logits, dim=1)
   ```

**Acceptance Criteria:**
- [ ] Model outputs both logits and Mc prediction
- [ ] Mc prediction always positive (Softplus)
- [ ] Shared features enable joint learning

**Files Modified:**
- `src/models/cnn.py`

---

### E7-3: Combined Loss
**Priority:** 🟠 High | **Estimate:** 2 hours

**Objective:** Implement weighted combination of classification and physics losses

**Implementation Steps:**
1. Add combined loss class:
   ```python
   class CombinedPhysicsLoss(nn.Module):
       """Combined classification + physics-informed loss"""
       
       def __init__(self,
                    alpha: float = 1.0,
                    beta: float = 0.1,
                    gamma: float = 0.1,
                    class_weights: torch.Tensor = None):
           """
           Total Loss = α * L_classification + β * L_mc_constraint + γ * L_mc_regression
           
           Args:
               alpha: Weight for classification loss
               beta: Weight for chirp mass constraint (physics)
               gamma: Weight for chirp mass regression (if true Mc available)
               class_weights: Weights for imbalanced classes
           """
           super().__init__()
           self.alpha = alpha
           self.beta = beta
           self.gamma = gamma
           
           self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
           self.mc_constraint_loss = ChirpMassLoss(mc_threshold=2.5)
           self.mc_regression_loss = ChirpMassRegressionLoss(log_scale=True)
       
       def forward(self, 
                   logits: torch.Tensor,
                   mc_pred: torch.Tensor,
                   labels: torch.Tensor,
                   mc_true: torch.Tensor = None) -> dict:
           """
           Args:
               logits: Classification logits [B, 2]
               mc_pred: Predicted chirp mass [B, 1]
               labels: Ground truth labels [B]
               mc_true: True chirp mass [B] (optional, for synthetic data)
           
           Returns:
               Dictionary with total loss and components
           """
           # Classification loss
           l_class = self.ce_loss(logits, labels)
           
           # Physics constraint loss
           l_constraint = self.mc_constraint_loss(mc_pred, labels)
           
           # Total loss
           total = self.alpha * l_class + self.beta * l_constraint
           
           losses = {
               'total': total,
               'classification': l_class,
               'mc_constraint': l_constraint
           }
           
           # Optional: regression loss if true Mc available
           if mc_true is not None:
               l_regression = self.mc_regression_loss(mc_pred, mc_true)
               total = total + self.gamma * l_regression
               losses['mc_regression'] = l_regression
               losses['total'] = total
           
           return losses
   ```

2. Usage in training:
   ```python
   criterion = CombinedPhysicsLoss(
       alpha=1.0,    # Classification weight
       beta=0.1,     # Physics constraint weight (tune this)
       gamma=0.05,   # Mc regression weight
       class_weights=train_loader.dataset.get_class_weights().cuda()
   )
   
   # Training step
   logits, mc_pred = model(images)
   loss_dict = criterion(logits, mc_pred, labels, mc_true)
   loss = loss_dict['total']
   loss.backward()
   ```

**Acceptance Criteria:**
- [ ] Loss combines all components
- [ ] Weights configurable
- [ ] Returns breakdown for logging

**Files Modified:**
- `src/models/physics_loss.py`

---

### E7-4: PINN Training
**Priority:** 🟠 High | **Estimate:** 4-6 hours (compute)

**Objective:** Train physics-informed neural network

**Implementation Steps:**
1. Create training script:
   ```python
   # scripts/train_pinn.py
   
   import torch
   from src.models.cnn import PhysicsInformedCNN
   from src.models.physics_loss import CombinedPhysicsLoss
   from src.training.trainer import Trainer
   from src.data.dataset import create_dataloaders
   
   def train_pinn(config):
       """Train physics-informed CNN"""
       
       # Data
       train_loader, val_loader = create_dataloaders(
           config, 
           batch_size=config['batch_size'],
           include_mc=True  # Include chirp mass in batch
       )
       
       # Model
       model = PhysicsInformedCNN(
           backbone=config['backbone'],
           pretrained=True,
           dropout=config['dropout']
       ).cuda()
       
       # Loss
       criterion = CombinedPhysicsLoss(
           alpha=config.get('alpha', 1.0),
           beta=config.get('beta', 0.1),
           gamma=config.get('gamma', 0.05),
           class_weights=train_loader.dataset.get_class_weights().cuda()
       )
       
       # Optimizer with different LR for backbone vs heads
       params = [
           {'params': model.backbone.parameters(), 'lr': config['lr'] * 0.1},
           {'params': model.classifier.parameters(), 'lr': config['lr']},
           {'params': model.mc_regressor.parameters(), 'lr': config['lr']}
       ]
       optimizer = torch.optim.AdamW(params, weight_decay=config['weight_decay'])
       
       # Scheduler
       scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
           optimizer, T_0=10, T_mult=2
       )
       
       # Custom training loop for multi-output
       best_recall = 0
       for epoch in range(config['epochs']):
           model.train()
           epoch_losses = {'total': 0, 'classification': 0, 'mc_constraint': 0}
           
           for images, labels, mc_true in train_loader:
               images = images.cuda()
               labels = labels.cuda()
               mc_true = mc_true.cuda()
               
               optimizer.zero_grad()
               
               logits, mc_pred = model(images)
               loss_dict = criterion(logits, mc_pred, labels, mc_true)
               
               loss_dict['total'].backward()
               optimizer.step()
               
               for k, v in loss_dict.items():
                   epoch_losses[k] += v.item()
           
           # Validation
           val_metrics = validate_pinn(model, val_loader)
           
           print(f"Epoch {epoch+1}: "
                 f"Loss={epoch_losses['total']/len(train_loader):.4f}, "
                 f"Mc_constraint={epoch_losses['mc_constraint']/len(train_loader):.4f}, "
                 f"Val_Recall={val_metrics['recall']:.4f}")
           
           if val_metrics['recall'] > best_recall:
               best_recall = val_metrics['recall']
               torch.save(model.state_dict(), 'outputs/models/phase2/best_pinn.pth')
           
           scheduler.step()
       
       return model
   
   if __name__ == '__main__':
       config = {
           'backbone': 'resnet50',
           'batch_size': 64,
           'lr': 1e-4,
           'weight_decay': 1e-4,
           'dropout': 0.5,
           'epochs': 50,
           'alpha': 1.0,
           'beta': 0.1,
           'gamma': 0.05
       }
       train_pinn(config)
   ```

2. Create Colab notebook `notebooks/08_phase2_pinn.ipynb`

**Acceptance Criteria:**
- [ ] PINN trains successfully
- [ ] Physics loss decreases over training
- [ ] Mc predictions reasonable

**Files Created:**
- `scripts/train_pinn.py`
- `notebooks/08_phase2_pinn.ipynb`

---

### E7-5: Ablation Study
**Priority:** 🟡 Medium | **Estimate:** 4-6 hours (compute)

**Objective:** Compare CNN with and without physics loss

**Implementation Steps:**
1. Run experiments:
   ```python
   ABLATION_CONFIGS = [
       {'name': 'cnn_baseline', 'beta': 0.0, 'gamma': 0.0},
       {'name': 'cnn_constraint', 'beta': 0.1, 'gamma': 0.0},
       {'name': 'cnn_regression', 'beta': 0.0, 'gamma': 0.1},
       {'name': 'cnn_full_pinn', 'beta': 0.1, 'gamma': 0.05},
       {'name': 'cnn_high_physics', 'beta': 0.5, 'gamma': 0.1},
   ]
   
   results = []
   for cfg in ABLATION_CONFIGS:
       model = train_pinn({**base_config, **cfg})
       metrics = evaluate_on_test(model)
       results.append({'config': cfg['name'], **metrics})
   
   pd.DataFrame(results).to_csv('outputs/results/ablation_physics.csv')
   ```

2. Visualize results:
   ```python
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
   # Accuracy comparison
   axes[0].bar(range(len(results)), [r['accuracy'] for r in results])
   axes[0].set_xticklabels([r['config'] for r in results], rotation=45)
   axes[0].set_ylabel('Accuracy')
   
   # NS Recall comparison
   axes[1].bar(range(len(results)), [r['ns_recall'] for r in results])
   axes[1].set_ylabel('NS-present Recall')
   
   # Mc MAE comparison
   axes[2].bar(range(len(results)), [r['mc_mae'] for r in results])
   axes[2].set_ylabel('Chirp Mass MAE (M☉)')
   
   plt.savefig('outputs/figures/ablation_physics.png', dpi=150)
   ```

**Expected Results:**
| Config | Accuracy | NS Recall | Mc MAE |
|--------|----------|-----------|--------|
| cnn_baseline | ~92% | ~80% | N/A |
| cnn_constraint | ~93% | ~85% | ~5.2 |
| cnn_full_pinn | ~94% | ~88% | ~3.8 |

**Acceptance Criteria:**
- [ ] All ablation experiments run
- [ ] Physics loss improves NS recall
- [ ] Results documented

**Files Created:**
- `outputs/results/ablation_physics.csv`
- `outputs/figures/ablation_physics.png`

---

### E7-6: Mc Validation
**Priority:** 🟡 Medium | **Estimate:** 2-3 hours

**Objective:** Validate chirp mass predictions against GWTC catalog

**Implementation Steps:**
1. Compare predictions to catalog:
   ```python
   def validate_mc_predictions(model, test_loader, catalog_mc):
       """Compare predicted Mc to GWTC catalog values"""
       
       model.eval()
       predictions = []
       
       with torch.no_grad():
           for images, labels, event_ids in test_loader:
               _, mc_pred = model(images.cuda())
               for eid, mc in zip(event_ids, mc_pred.cpu().numpy()):
                   predictions.append({
                       'event_id': eid,
                       'mc_predicted': mc[0],
                       'mc_catalog': catalog_mc.get(eid, np.nan)
                   })
       
       df = pd.DataFrame(predictions).dropna()
       
       # Metrics
       mae = np.abs(df['mc_predicted'] - df['mc_catalog']).mean()
       rmse = np.sqrt(((df['mc_predicted'] - df['mc_catalog'])**2).mean())
       r2 = 1 - ((df['mc_predicted'] - df['mc_catalog'])**2).sum() / \
                ((df['mc_catalog'] - df['mc_catalog'].mean())**2).sum()
       
       print(f"Chirp Mass Validation:")
       print(f"  MAE: {mae:.2f} M☉")
       print(f"  RMSE: {rmse:.2f} M☉")
       print(f"  R²: {r2:.3f}")
       
       # Scatter plot
       plt.figure(figsize=(8, 8))
       plt.scatter(df['mc_catalog'], df['mc_predicted'], alpha=0.5)
       plt.plot([0, 100], [0, 100], 'r--', label='Perfect')
       plt.xlabel('Catalog Mc (M☉)')
       plt.ylabel('Predicted Mc (M☉)')
       plt.title(f'Chirp Mass: Predicted vs Catalog (R²={r2:.3f})')
       plt.legend()
       plt.savefig('outputs/figures/mc_validation.png', dpi=150)
       
       return df, {'mae': mae, 'rmse': rmse, 'r2': r2}
   ```

2. Analyze by source type:
   ```python
   # BBH should have higher Mc, NS-present lower
   df['source_type'] = df['mc_catalog'].apply(
       lambda x: 'NS-present' if x < 3.0 else 'BBH'
   )
   
   for st in ['BBH', 'NS-present']:
       subset = df[df['source_type'] == st]
       print(f"{st}: MAE={subset['mc_predicted'] - subset['mc_catalog']).abs().mean():.2f}")
   ```

**Acceptance Criteria:**
- [ ] Mc predictions correlate with catalog
- [ ] MAE < 10 M☉ for synthetic
- [ ] MAE < 15 M☉ for real events

**Files Created:**
- `outputs/figures/mc_validation.png`
- `outputs/results/mc_validation.csv`

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

---

### E9-1: Example Spectrograms
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Generate publication-quality spectrogram figures

**Implementation Steps:**
1. Create figure generation script:
   ```python
   # scripts/generate_figures.py
   
   import matplotlib.pyplot as plt
   import numpy as np
   from pathlib import Path
   
   plt.rcParams.update({
       'font.size': 12,
       'font.family': 'serif',
       'axes.labelsize': 14,
       'axes.titlesize': 14,
       'xtick.labelsize': 12,
       'ytick.labelsize': 12,
       'legend.fontsize': 11,
       'figure.dpi': 300,
       'savefig.dpi': 300,
       'savefig.bbox': 'tight'
   })
   
   def create_example_spectrograms():
       """Create side-by-side BBH vs NS-present spectrograms"""
       
       # Select canonical examples
       examples = {
           'BBH': ['GW150914', 'GW190521', 'GW191109'],
           'NS-present': ['GW170817', 'GW190425', 'GW200115']
       }
       
       fig, axes = plt.subplots(2, 3, figsize=(12, 8))
       
       for row, (label, events) in enumerate(examples.items()):
           for col, event in enumerate(events):
               ax = axes[row, col]
               
               # Load spectrogram
               spec = np.load(f'data/spectrograms/matrices/{event}_H1.npy')
               
               im = ax.imshow(spec, aspect='auto', origin='lower',
                             cmap='viridis', extent=[0, 4, 20, 500])
               
               ax.set_title(event, fontweight='bold')
               
               if col == 0:
                   ax.set_ylabel(f'{label}\nFrequency (Hz)')
               if row == 1:
                   ax.set_xlabel('Time (s)')
       
       # Colorbar
       cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
       cbar.set_label('Normalized Power')
       
       plt.suptitle('Spectrogram Examples: BBH vs NS-present', fontsize=16, y=1.02)
       plt.tight_layout()
       plt.savefig('outputs/figures/example_spectrograms.png')
       plt.savefig('outputs/figures/example_spectrograms.pdf')  # Vector for paper
       
       return fig
   ```

2. Create chirp evolution comparison:
   ```python
   def create_chirp_comparison():
       """Show chirp pattern differences between source types"""
       
       fig, axes = plt.subplots(1, 3, figsize=(14, 4))
       
       # BBH: high mass, short chirp
       # BNS: low mass, long chirp (outside band mostly)
       # NSBH: intermediate
       
       for ax, (event, params) in zip(axes, [
           ('GW150914 (BBH)', {'m1': 36, 'm2': 29, 'duration': 0.2}),
           ('GW170817 (BNS)', {'m1': 1.4, 'm2': 1.3, 'duration': 100}),
           ('GW200115 (NSBH)', {'m1': 6, 'm2': 1.4, 'duration': 1.0})
       ]):
           # Load and plot
           spec = np.load(f'data/spectrograms/matrices/{event.split()[0]}_H1.npy')
           ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
           ax.set_title(event)
       
       plt.savefig('outputs/figures/chirp_comparison.png')
   ```

**Acceptance Criteria:**
- [ ] High-resolution figures (300 DPI)
- [ ] PDF vector format for paper
- [ ] Clear labels and colorbars
- [ ] Representative examples selected

**Files Created:**
- `outputs/figures/example_spectrograms.png`
- `outputs/figures/example_spectrograms.pdf`
- `outputs/figures/chirp_comparison.png`

---

### E9-2: Feature Distribution Plots
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Visualize feature distributions by class

**Implementation Steps:**
1. Create violin/box plots:
   ```python
   import seaborn as sns
   
   def create_feature_distributions():
       """Create violin plots of key features"""
       
       features = pd.read_csv('data/features/combined_features.csv')
       
       # Key discriminative features
       key_features = [
           'chirp_mass_est', 'peak_frequency', 'freq_bandwidth',
           'chirp_rate', 'ridge_snr', 'duration_above_threshold'
       ]
       
       fig, axes = plt.subplots(2, 3, figsize=(14, 10))
       
       for ax, feat in zip(axes.flat, key_features):
           sns.violinplot(data=features, x='label', y=feat, ax=ax,
                         palette={'BBH': '#1f77b4', 'NS-present': '#d62728'})
           ax.set_xlabel('')
           ax.set_ylabel(feat.replace('_', ' ').title())
       
       plt.suptitle('Feature Distributions by Source Type', fontsize=16)
       plt.tight_layout()
       plt.savefig('outputs/figures/feature_distributions.png')
       
       # Also create box plots for paper
       fig, axes = plt.subplots(2, 3, figsize=(14, 10))
       for ax, feat in zip(axes.flat, key_features):
           sns.boxplot(data=features, x='label', y=feat, ax=ax,
                      palette={'BBH': '#1f77b4', 'NS-present': '#d62728'})
       plt.savefig('outputs/figures/feature_boxplots.png')
   ```

2. Feature correlation heatmap:
   ```python
   def create_correlation_heatmap():
       """Feature correlation matrix"""
       
       features = pd.read_csv('data/features/combined_features.csv')
       numeric_cols = features.select_dtypes(include=[np.number]).columns
       
       corr = features[numeric_cols].corr()
       
       plt.figure(figsize=(12, 10))
       sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5)
       plt.title('Feature Correlation Matrix')
       plt.tight_layout()
       plt.savefig('outputs/figures/feature_correlation.png')
   ```

**Acceptance Criteria:**
- [ ] Clear class separation visible
- [ ] Key features highlighted
- [ ] Consistent color scheme

**Files Created:**
- `outputs/figures/feature_distributions.png`
- `outputs/figures/feature_boxplots.png`
- `outputs/figures/feature_correlation.png`

---

### E9-3: ROC/PR Curves
**Priority:** 🔴 Critical | **Estimate:** 2-3 hours

**Objective:** Publication-quality evaluation curves

**Implementation Steps:**
1. Create ROC curves for all models:
   ```python
   from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
   
   def create_roc_curves(results_dict):
       """
       Args:
           results_dict: {model_name: {'y_true': [...], 'y_prob': [...]}}
       """
       
       fig, axes = plt.subplots(1, 2, figsize=(12, 5))
       
       colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
       
       # ROC Curve
       ax = axes[0]
       for (name, res), color in zip(results_dict.items(), colors):
           fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
           roc_auc = auc(fpr, tpr)
           ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{name} (AUC = {roc_auc:.3f})')
       
       ax.plot([0, 1], [0, 1], 'k--', lw=1)
       ax.set_xlim([0, 1])
       ax.set_ylim([0, 1.05])
       ax.set_xlabel('False Positive Rate')
       ax.set_ylabel('True Positive Rate (Recall)')
       ax.set_title('ROC Curve')
       ax.legend(loc='lower right')
       ax.grid(True, alpha=0.3)
       
       # Precision-Recall Curve
       ax = axes[1]
       for (name, res), color in zip(results_dict.items(), colors):
           precision, recall, _ = precision_recall_curve(res['y_true'], res['y_prob'])
           ap = average_precision_score(res['y_true'], res['y_prob'])
           ax.plot(recall, precision, color=color, lw=2,
                   label=f'{name} (AP = {ap:.3f})')
       
       # Baseline: class ratio
       baseline = sum(results_dict[list(results_dict.keys())[0]]['y_true']) / \
                  len(results_dict[list(results_dict.keys())[0]]['y_true'])
       ax.axhline(y=baseline, color='k', linestyle='--', lw=1, label='Baseline')
       
       ax.set_xlabel('Recall')
       ax.set_ylabel('Precision')
       ax.set_title('Precision-Recall Curve')
       ax.legend(loc='lower left')
       ax.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.savefig('outputs/figures/roc_pr_curves.png')
       plt.savefig('outputs/figures/roc_pr_curves.pdf')
       
       return fig
   
   # Usage
   results = {
       'Random Forest': load_results('rf'),
       'XGBoost': load_results('xgb'),
       'CNN (ResNet50)': load_results('cnn'),
       'Physics-Informed CNN': load_results('pinn')
   }
   create_roc_curves(results)
   ```

**Acceptance Criteria:**
- [ ] All models on same plot
- [ ] AUC/AP values in legend
- [ ] Grid and proper scaling
- [ ] Both PNG and PDF

**Files Created:**
- `outputs/figures/roc_pr_curves.png`
- `outputs/figures/roc_pr_curves.pdf`

---

### E9-4: Confusion Matrices
**Priority:** 🟠 High | **Estimate:** 1-2 hours

**Objective:** Side-by-side confusion matrices

**Implementation Steps:**
1. Create comparison figure:
   ```python
   from sklearn.metrics import confusion_matrix
   
   def create_confusion_matrices(results_dict):
       """Create confusion matrix comparison"""
       
       n_models = len(results_dict)
       fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
       
       if n_models == 1:
           axes = [axes]
       
       for ax, (name, res) in zip(axes, results_dict.items()):
           cm = confusion_matrix(res['y_true'], res['y_pred'])
           
           # Normalize
           cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
           
           sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['BBH', 'NS-present'],
                       yticklabels=['BBH', 'NS-present'])
           
           # Add percentages
           for i in range(2):
               for j in range(2):
                   ax.text(j+0.5, i+0.7, f'({cm_norm[i,j]:.0%})',
                          ha='center', va='center', fontsize=10, color='gray')
           
           ax.set_title(name)
           ax.set_ylabel('True Label')
           ax.set_xlabel('Predicted Label')
       
       plt.tight_layout()
       plt.savefig('outputs/figures/confusion_matrices.png')
       plt.savefig('outputs/figures/confusion_matrices.pdf')
   ```

**Acceptance Criteria:**
- [ ] All models compared
- [ ] Both counts and percentages
- [ ] Clear labeling

**Files Created:**
- `outputs/figures/confusion_matrices.png`
- `outputs/figures/confusion_matrices.pdf`

---

### E9-5: Architecture Diagram
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Visual pipeline overview diagram

**Implementation Steps:**
1. Create using matplotlib/draw.io:
   ```python
   import matplotlib.patches as mpatches
   from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
   
   def create_pipeline_diagram():
       """Create visual pipeline diagram"""
       
       fig, ax = plt.subplots(1, 1, figsize=(16, 8))
       ax.set_xlim(0, 16)
       ax.set_ylim(0, 8)
       ax.axis('off')
       
       # Boxes for each stage
       stages = [
           (1, 6, 'GWOSC\nData', '#E8F4FD'),
           (4, 6, 'Preprocessing\n(Bandpass, Whiten)', '#FFF3E0'),
           (7, 6, 'Spectrogram\nGeneration', '#E8F5E9'),
           (10, 6, 'Feature\nExtraction', '#FCE4EC'),
           (13, 6, 'Classification', '#E3F2FD'),
       ]
       
       for x, y, text, color in stages:
           box = FancyBboxPatch((x-1, y-0.8), 2, 1.6,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='black', lw=2)
           ax.add_patch(box)
           ax.text(x, y, text, ha='center', va='center', fontsize=10,
                   fontweight='bold')
       
       # Arrows
       for i in range(4):
           x1 = stages[i][0] + 1
           x2 = stages[i+1][0] - 1
           ax.annotate('', xy=(x2, 6), xytext=(x1, 6),
                       arrowprops=dict(arrowstyle='->', lw=2))
       
       # Add Phase labels
       ax.text(5.5, 4, 'Phase 1: Feature-based ML\n(Random Forest, XGBoost)',
               ha='center', fontsize=11, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat'))
       
       ax.text(10.5, 4, 'Phase 2: Deep Learning\n(CNN, Physics-Informed)',
               ha='center', fontsize=11, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightblue'))
       
       plt.title('Gravitational Wave Event Classification Pipeline', fontsize=14, pad=20)
       plt.savefig('outputs/figures/pipeline_diagram.png', dpi=300, bbox_inches='tight')
       plt.savefig('outputs/figures/pipeline_diagram.pdf', bbox_inches='tight')
   ```

2. Alternative: Create in draw.io/Figma for better control

**Acceptance Criteria:**
- [ ] All pipeline stages shown
- [ ] Phase 1 vs Phase 2 clear
- [ ] Professional appearance

**Files Created:**
- `outputs/figures/pipeline_diagram.png`
- `outputs/figures/pipeline_diagram.pdf`

---

### E9-6: Methods Section
**Priority:** 🔴 Critical | **Estimate:** 4-6 hours

**Objective:** Write data and methods for paper/report

**Implementation Steps:**
1. Structure for methods section:
   ```markdown
   # Methods
   
   ## 2.1 Data Sources
   
   We use gravitational wave strain data from the Gravitational Wave Open Science 
   Center (GWOSC) [1]. Our dataset comprises N events from the GWTC catalogs:
   - GWTC-1 (O1/O2): X BBH, Y BNS
   - GWTC-2 (O3a): X BBH, Y NSBH
   - GWTC-3 (O3b): X BBH, Y NSBH
   
   Events are classified as "NS-present" if at least one component has mass 
   < 3 M☉, following [2]. This yields a binary classification with ~85 BBH 
   and ~5 NS-present events (highly imbalanced).
   
   ## 2.2 Data Preprocessing
   
   Raw strain data is processed following standard GW analysis:
   1. Bandpass filter: 20-500 Hz (contains merger signal)
   2. Whitening: Normalize by estimated PSD
   3. Event-centered windowing: 4s around merger time
   
   ## 2.3 Spectrogram Generation
   
   Q-transform spectrograms computed with:
   - Q-range: [4, 64]
   - Frequency range: 20-500 Hz
   - Time resolution: Adaptive to Q
   - Normalization: Per-event to [0, 1]
   
   ## 2.4 Synthetic Data Generation
   
   To address class imbalance, we generate 10,000 synthetic injections using 
   PyCBC [3]:
   - BBH: 5,000 signals using IMRPhenomD waveforms
   - BNS: 3,000 signals using TaylorF2 waveforms  
   - NSBH: 2,000 signals using IMRPhenomNSBH waveforms
   
   Waveforms injected into real LIGO noise at SNR 8-30.
   
   ## 2.5 Feature Extraction
   
   Ridge-based features extracted:
   - Chirp mass estimate from f-fdot relation
   - Peak frequency and bandwidth
   - Signal duration above threshold
   - Statistical moments of spectrogram
   
   ## 2.6 Classification Models
   
   **Phase 1 (Feature-based):**
   - Random Forest (100 trees)
   - XGBoost with class weighting
   - 5-fold stratified cross-validation
   
   **Phase 2 (Deep Learning):**
   - ResNet50 pretrained on ImageNet
   - Physics-informed loss with chirp mass constraint
   - Training on Colab A100 GPU
   ```

**Acceptance Criteria:**
- [ ] All methods documented
- [ ] References included
- [ ] Reproducible from description

**Files Created:**
- `docs/paper/methods.md`

---

### E9-7: Results Section
**Priority:** 🔴 Critical | **Estimate:** 4-6 hours

**Objective:** Write results with tables and figures

**Implementation Steps:**
1. Results structure:
   ```markdown
   # Results
   
   ## 3.1 Phase 1: Feature-Based Classification
   
   Table 1 shows cross-validation results on real events (O1-O3):
   
   | Model | Accuracy | NS Recall | BBH Recall | F1 | AUC |
   |-------|----------|-----------|------------|-----|-----|
   | Random Forest | 87.2±3.1% | 60.0±12% | 91.2±2.8% | 0.71 | 0.82 |
   | XGBoost | 89.1±2.8% | 70.0±10% | 92.1±2.5% | 0.76 | 0.85 |
   
   Key discriminative features (SHAP analysis):
   1. Estimated chirp mass (importance: 0.32)
   2. Peak frequency (importance: 0.18)
   3. Ridge SNR (importance: 0.15)
   
   ## 3.2 Phase 2: Deep Learning
   
   Table 2 shows CNN results trained on synthetic + real data:
   
   | Model | Accuracy | NS Recall | AUC |
   |-------|----------|-----------|-----|
   | CNN (ResNet50) | 93.2% | 85.0% | 0.91 |
   | Physics-Informed CNN | 94.5% | 88.0% | 0.93 |
   
   The physics-informed loss improved NS recall by 3 percentage points.
   
   ## 3.3 Ablation Studies
   
   Physics loss ablation (Table 3)...
   
   ## 3.4 Generalization to O4a
   
   Testing on held-out O4a events showed...
   ```

**Acceptance Criteria:**
- [ ] All results tabulated
- [ ] Figures referenced
- [ ] Statistical significance noted

**Files Created:**
- `docs/paper/results.md`

---

### E9-8: Discussion/Limitations
**Priority:** 🟠 High | **Estimate:** 2-3 hours

**Objective:** Honest assessment of limitations

**Implementation Steps:**
1. Discussion points:
   ```markdown
   # Discussion
   
   ## 4.1 Key Findings
   
   - Deep learning outperforms feature-based methods
   - Physics-informed loss improves minority class recall
   - Synthetic data critical for training with <100 real events
   
   ## 4.2 Limitations
   
   1. **Small NS-present sample**: Only ~5 confirmed events
   2. **Synthetic-real domain gap**: Models may overfit to PyCBC waveforms
   3. **Detector sensitivity**: Results specific to O1-O3 noise characteristics
   4. **Binary classification**: Cannot distinguish BNS from NSBH
   5. **Single detector**: Used only H1; multi-detector could improve
   
   ## 4.3 Future Work
   
   - Multi-detector fusion
   - Continuous Mc regression
   - Real-time inference pipeline
   - Extension to O4/O5 data
   ```

**Acceptance Criteria:**
- [ ] Limitations honestly stated
- [ ] Future work outlined
- [ ] Balanced perspective

**Files Created:**
- `docs/paper/discussion.md`

---

### E9-9: README Polish
**Priority:** 🔴 Critical | **Estimate:** 2-3 hours

**Objective:** Final project README with setup instructions

**Implementation Steps:**
1. Update `README.md`:
   ```markdown
   # Gravitational Wave Event Classification
   
   [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]
   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]
   
   Binary classification of gravitational wave events into **BBH** 
   (Binary Black Hole) vs **NS-present** (contains Neutron Star) using 
   deep learning and physics-informed neural networks.
   
   ## 🎯 Project Overview
   
   This project was developed for [Science Fair Name]. It achieves:
   - **94.5% accuracy** on real gravitational wave events
   - **88% recall** on rare NS-present events
   - Physics-informed CNN with chirp mass constraints
   
   ## 🚀 Quick Start
   
   ```bash
   # Clone repository
   git clone https://github.com/Arctic-XD/gw-event-classification.git
   cd gw-event-classification
   
   # Create environment
   conda env create -f environment.yml
   conda activate gw-classify
   
   # Download data
   python scripts/download_events.py
   
   # Run Phase 1 baseline
   python scripts/train_baseline.py
   ```
   
   ## 📁 Project Structure
   
   ```
   ├── configs/          # Configuration files
   ├── data/             # Data directory (gitignored)
   ├── docs/             # Documentation and ADRs
   ├── notebooks/        # Jupyter notebooks
   ├── outputs/          # Results, models, figures
   ├── scripts/          # Executable scripts
   ├── src/              # Source code
   └── tests/            # Unit tests
   ```
   
   ## 📊 Results
   
   | Model | Accuracy | NS Recall | AUC |
   |-------|----------|-----------|-----|
   | Random Forest | 87.2% | 60.0% | 0.82 |
   | XGBoost | 89.1% | 70.0% | 0.85 |
   | CNN | 93.2% | 85.0% | 0.91 |
   | **Physics-Informed CNN** | **94.5%** | **88.0%** | **0.93** |
   
   ## 🔬 Methodology
   
   See [docs/PLAN.md](docs/PLAN.md) for detailed methodology and 
   [docs/ADR/](docs/ADR/) for architecture decisions.
   
   ## 📚 References
   
   - [GWOSC](https://gwosc.org/)
   - [PyCBC](https://pycbc.org/)
   - [GWpy](https://gwpy.github.io/)
   
   ## 📝 License
   
   MIT License - see [LICENSE](LICENSE) for details.
   ```

**Acceptance Criteria:**
- [ ] Clear setup instructions
- [ ] Results summary
- [ ] Professional appearance

**Files Modified:**
- `README.md`

---

### E9-10: Presentation Prep
**Priority:** 🔴 Critical | **Estimate:** 6-8 hours

**Objective:** Slides for science fair presentation

**Implementation Steps:**
1. Slide outline (15-20 slides):
   ```
   1. Title Slide
      - "Classifying Gravitational Wave Events with Physics-Informed Deep Learning"
   
   2. What are Gravitational Waves?
      - Einstein's prediction (1916)
      - First detection (2015)
      - Ripples in spacetime
   
   3. Why Classification Matters
      - Different source physics (BH vs NS)
      - Neutron star mergers → heavy elements
      - Multi-messenger astronomy
   
   4. The Challenge
      - Rare events (~90 total)
      - Extreme class imbalance (~5 NS-present)
      - Noisy data
   
   5. Data: LIGO Spectrograms
      - Example BBH spectrogram
      - Example NS-present spectrogram
      - Key differences (chirp pattern)
   
   6. Approach Overview
      - Phase 1: Traditional ML
      - Phase 2: Deep Learning + Physics
   
   7. Synthetic Data Generation
      - Why synthetic? (address imbalance)
      - 10,000 simulated signals
   
   8. Feature Extraction
      - Chirp mass estimation
      - Ridge extraction
   
   9. Phase 1 Results
      - Random Forest / XGBoost
      - 87-89% accuracy
   
   10. Deep Learning Architecture
       - ResNet50 diagram
       - Transfer learning
   
   11. Physics-Informed Loss
       - Chirp mass constraint
       - Why physics helps
   
   12. Phase 2 Results
       - CNN: 93% accuracy
       - PINN: 94.5% accuracy, 88% NS recall
   
   13. Confusion Matrix Comparison
       - Visual comparison
   
   14. Key Findings
       - DL > traditional ML
       - Physics loss helps rare class
       - Synthetic data essential
   
   15. Limitations
       - Small sample size
       - Synthetic-real gap
   
   16. Future Work
       - Real-time detection
       - Multi-detector fusion
   
   17. Conclusion
       - Successfully classified GW events
       - Physics-informed approach effective
   
   18. Acknowledgments
       - GWOSC, advisors, etc.
   
   19. Questions?
   
   20. Backup: Technical Details
   ```

2. Create visual assets for slides:
   - Pipeline diagram
   - Example spectrograms
   - Results charts
   - Architecture diagram

**Acceptance Criteria:**
- [ ] 15-20 slides
- [ ] Clear narrative
- [ ] Visual-heavy (minimal text)
- [ ] Practice timing (~10 min)

**Files Created:**
- `docs/presentation/slides.pptx` (or Google Slides)
- `docs/presentation/speaker_notes.md`

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
