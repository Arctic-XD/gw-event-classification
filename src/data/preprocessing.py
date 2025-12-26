"""
Preprocessing utilities for gravitational wave strain data.

Provides bandpass filtering, whitening, and other signal processing
operations needed before spectrogram generation.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal
from gwpy.timeseries import TimeSeries

logger = logging.getLogger(__name__)


def bandpass_filter(
    strain: Union[TimeSeries, np.ndarray],
    low_freq: float = 20.0,
    high_freq: float = 512.0,
    sample_rate: float = 4096.0,
    order: int = 8
) -> Union[TimeSeries, np.ndarray]:
    """
    Apply a Butterworth bandpass filter to strain data.
    
    Args:
        strain: Input strain data (TimeSeries or numpy array)
        low_freq: Lower cutoff frequency in Hz
        high_freq: Upper cutoff frequency in Hz  
        sample_rate: Sample rate in Hz (only needed if strain is numpy array)
        order: Filter order
        
    Returns:
        Filtered strain data (same type as input)
    """
    # Get sample rate from TimeSeries if available
    if isinstance(strain, TimeSeries):
        sample_rate = strain.sample_rate.value
        data = strain.value
    else:
        data = strain
    
    # Nyquist frequency
    nyquist = sample_rate / 2.0
    
    # Normalized frequencies
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter (forward-backward for zero phase)
    filtered = signal.filtfilt(b, a, data)
    
    # Return same type as input
    if isinstance(strain, TimeSeries):
        return TimeSeries(filtered, sample_rate=sample_rate, t0=strain.t0)
    return filtered


def whiten(
    strain: Union[TimeSeries, np.ndarray],
    sample_rate: float = 4096.0,
    segment_duration: float = 4.0,
    low_freq: float = 20.0,
    high_freq: float = 512.0
) -> Union[TimeSeries, np.ndarray]:
    """
    Whiten strain data using Welch PSD estimation.
    
    Whitening divides the data by the estimated noise amplitude spectral
    density (ASD), making the noise approximately white (flat spectrum).
    
    Args:
        strain: Input strain data
        sample_rate: Sample rate in Hz
        segment_duration: Duration of segments for PSD estimation
        low_freq: Minimum frequency for whitening
        high_freq: Maximum frequency for whitening
        
    Returns:
        Whitened strain data
    """
    # Get data as numpy array
    if isinstance(strain, TimeSeries):
        sample_rate = strain.sample_rate.value
        data = strain.value
        t0 = strain.t0
    else:
        data = strain
        t0 = None
    
    # FFT parameters
    nfft = int(segment_duration * sample_rate)
    
    # Estimate PSD using Welch method
    freqs, psd = signal.welch(
        data,
        fs=sample_rate,
        nperseg=nfft,
        noverlap=nfft // 2,
        window='hann'
    )
    
    # Interpolate PSD to full frequency resolution
    fft_freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
    psd_interp = np.interp(fft_freqs, freqs, psd)
    
    # Avoid division by zero
    psd_interp = np.maximum(psd_interp, 1e-50)
    
    # FFT of data
    data_fft = np.fft.rfft(data)
    
    # Whiten in frequency domain
    whitened_fft = data_fft / np.sqrt(psd_interp)
    
    # Apply frequency band mask
    freq_mask = (fft_freqs >= low_freq) & (fft_freqs <= high_freq)
    whitened_fft[~freq_mask] = 0
    
    # Inverse FFT
    whitened = np.fft.irfft(whitened_fft, n=len(data))
    
    # Normalize
    whitened = whitened / np.std(whitened)
    
    # Return same type as input
    if isinstance(strain, TimeSeries):
        return TimeSeries(whitened, sample_rate=sample_rate, t0=t0)
    return whitened


def notch_filter(
    strain: Union[TimeSeries, np.ndarray],
    frequencies: list = [60, 120, 180],
    sample_rate: float = 4096.0,
    quality_factor: float = 30.0
) -> Union[TimeSeries, np.ndarray]:
    """
    Apply notch filters to remove power line harmonics.
    
    Args:
        strain: Input strain data
        frequencies: List of frequencies to notch out (Hz)
        sample_rate: Sample rate in Hz
        quality_factor: Q factor for notch filter (higher = narrower notch)
        
    Returns:
        Filtered strain data
    """
    if isinstance(strain, TimeSeries):
        sample_rate = strain.sample_rate.value
        data = strain.value.copy()
        t0 = strain.t0
    else:
        data = strain.copy()
        t0 = None
    
    # Apply notch filter at each frequency
    for freq in frequencies:
        # Skip if frequency is above Nyquist
        if freq >= sample_rate / 2:
            continue
            
        # Design notch filter
        b, a = signal.iirnotch(freq, quality_factor, sample_rate)
        
        # Apply filter
        data = signal.filtfilt(b, a, data)
    
    if isinstance(strain, TimeSeries):
        return TimeSeries(data, sample_rate=sample_rate, t0=t0)
    return data


def preprocess_strain(
    strain: Union[TimeSeries, np.ndarray],
    sample_rate: float = 4096.0,
    bandpass: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 512.0,
    do_whiten: bool = True,
    notch: bool = True,
    notch_freqs: list = [60, 120, 180]
) -> Union[TimeSeries, np.ndarray]:
    """
    Apply full preprocessing pipeline to strain data.
    
    Pipeline:
    1. Remove mean (detrend)
    2. Bandpass filter
    3. Notch filter (optional)
    4. Whiten
    
    Args:
        strain: Input strain data
        sample_rate: Sample rate in Hz
        bandpass: Whether to apply bandpass filter
        low_freq: Bandpass lower frequency
        high_freq: Bandpass upper frequency
        do_whiten: Whether to whiten the data
        notch: Whether to apply notch filters
        notch_freqs: Frequencies for notch filtering
        
    Returns:
        Preprocessed strain data
    """
    # Get data
    if isinstance(strain, TimeSeries):
        sample_rate = strain.sample_rate.value
        data = strain.value.copy()
        t0 = strain.t0
        is_timeseries = True
    else:
        data = strain.copy()
        t0 = None
        is_timeseries = False
    
    # 1. Remove mean
    data = data - np.mean(data)
    
    # 2. Bandpass filter
    if bandpass:
        data = bandpass_filter(
            data,
            low_freq=low_freq,
            high_freq=high_freq,
            sample_rate=sample_rate
        )
    
    # 3. Notch filter
    if notch:
        data = notch_filter(
            data,
            frequencies=notch_freqs,
            sample_rate=sample_rate
        )
    
    # 4. Whiten
    if do_whiten:
        data = whiten(
            data,
            sample_rate=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq
        )
    
    # Return same type as input
    if is_timeseries:
        return TimeSeries(data, sample_rate=sample_rate, t0=t0)
    return data


def crop_around_merger(
    strain: Union[TimeSeries, np.ndarray],
    merger_time: float,
    before: float = 2.0,
    after: float = 0.5,
    sample_rate: float = 4096.0
) -> Union[TimeSeries, np.ndarray]:
    """
    Crop strain data to a window around the merger time.
    
    Args:
        strain: Input strain data
        merger_time: GPS time of merger (or sample index if numpy array)
        before: Seconds before merger to include
        after: Seconds after merger to include
        sample_rate: Sample rate in Hz
        
    Returns:
        Cropped strain data
    """
    if isinstance(strain, TimeSeries):
        start = merger_time - before
        end = merger_time + after
        return strain.crop(start, end)
    else:
        # Assume merger_time is a sample index or time in seconds from start
        samples_before = int(before * sample_rate)
        samples_after = int(after * sample_rate)
        merger_idx = int(merger_time * sample_rate) if merger_time < 1e6 else int(merger_time)
        
        start_idx = max(0, merger_idx - samples_before)
        end_idx = min(len(strain), merger_idx + samples_after)
        
        return strain[start_idx:end_idx]
