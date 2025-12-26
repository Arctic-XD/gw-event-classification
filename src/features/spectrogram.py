"""
Spectrogram generation utilities.

Generates time-frequency spectrograms from gravitational wave strain data.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal
from PIL import Image

logger = logging.getLogger(__name__)


class SpectrogramGenerator:
    """
    Generate spectrograms from strain data with consistent parameters.
    
    Attributes:
        sample_rate: Sample rate in Hz
        fft_length: FFT window length in seconds
        overlap: Overlap fraction (0-1)
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz
        
    Example:
        >>> gen = SpectrogramGenerator(sample_rate=4096)
        >>> spec, times, freqs = gen.generate(strain_data)
    """
    
    def __init__(
        self,
        sample_rate: int = 4096,
        fft_length: float = 1.0,
        overlap: float = 0.5,
        freq_min: float = 20.0,
        freq_max: float = 500.0,
        window: str = "hann",
        log_scale: bool = True
    ):
        """
        Initialize the spectrogram generator.
        
        Args:
            sample_rate: Sample rate in Hz
            fft_length: FFT window length in seconds
            overlap: Overlap fraction between windows
            freq_min: Minimum frequency to include
            freq_max: Maximum frequency to include
            window: Window function name
            log_scale: Apply log scaling to power
        """
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.overlap = overlap
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.window = window
        self.log_scale = log_scale
        
        # Derived parameters
        self.nperseg = int(fft_length * sample_rate)
        self.noverlap = int(self.nperseg * overlap)
    
    def generate(
        self,
        strain: np.ndarray,
        normalize: str = "per_event"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a spectrogram from strain data.
        
        Args:
            strain: 1D array of strain data
            normalize: Normalization method ("per_event", "global", "none")
            
        Returns:
            Tuple of (spectrogram, times, frequencies)
        """
        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            strain,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            mode='magnitude'
        )
        
        # Crop to frequency range
        freq_mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        frequencies = frequencies[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Apply log scaling
        if self.log_scale:
            Sxx = np.log10(Sxx + 1e-10)
        
        # Normalize
        if normalize == "per_event":
            Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-10)
        elif normalize == "minmax":
            Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx) + 1e-10)
        
        return Sxx, times, frequencies
    
    def generate_image(
        self,
        strain: np.ndarray,
        size: Tuple[int, int] = (224, 224),
        normalize: str = "per_event"
    ) -> np.ndarray:
        """
        Generate a spectrogram image suitable for CNN input.
        
        Args:
            strain: 1D array of strain data
            size: Output image size (width, height)
            normalize: Normalization method
            
        Returns:
            2D array of shape (height, width) with values in [0, 255]
        """
        # Generate spectrogram
        Sxx, _, _ = self.generate(strain, normalize=normalize)
        
        # Resize to target size
        img = Image.fromarray(Sxx)
        img = img.resize(size, Image.BILINEAR)
        Sxx_resized = np.array(img)
        
        # Convert to uint8 for image
        Sxx_norm = (Sxx_resized - Sxx_resized.min()) / (Sxx_resized.max() - Sxx_resized.min() + 1e-10)
        Sxx_uint8 = (Sxx_norm * 255).astype(np.uint8)
        
        return Sxx_uint8
    
    def save_spectrogram(
        self,
        strain: np.ndarray,
        output_path: Union[str, Path],
        size: Tuple[int, int] = (224, 224),
        format: str = "png"
    ) -> None:
        """
        Generate and save a spectrogram image.
        
        Args:
            strain: 1D array of strain data
            output_path: Path to save the image
            size: Output image size
            format: Image format ("png", "npy")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "npy":
            Sxx, _, _ = self.generate(strain)
            np.save(output_path.with_suffix(".npy"), Sxx)
        else:
            img_array = self.generate_image(strain, size=size)
            img = Image.fromarray(img_array)
            img.save(output_path.with_suffix(f".{format}"))


def generate_spectrogram(
    strain: np.ndarray,
    sample_rate: int = 4096,
    fft_length: float = 1.0,
    overlap: float = 0.5,
    freq_min: float = 20.0,
    freq_max: float = 500.0,
    log_scale: bool = True,
    normalize: str = "per_event"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to generate a spectrogram.
    
    Args:
        strain: 1D array of strain data
        sample_rate: Sample rate in Hz
        fft_length: FFT window length in seconds
        overlap: Overlap fraction
        freq_min: Minimum frequency
        freq_max: Maximum frequency
        log_scale: Apply log scaling
        normalize: Normalization method
        
    Returns:
        Tuple of (spectrogram, times, frequencies)
    """
    generator = SpectrogramGenerator(
        sample_rate=sample_rate,
        fft_length=fft_length,
        overlap=overlap,
        freq_min=freq_min,
        freq_max=freq_max,
        log_scale=log_scale
    )
    return generator.generate(strain, normalize=normalize)
