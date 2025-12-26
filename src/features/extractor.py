"""
Feature extraction from spectrograms.

Extracts quantitative features from spectrograms for Phase 1 ML models.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.ndimage import label

from .spectrogram import SpectrogramGenerator
from .ridge import RidgeExtractor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract quantitative features from spectrograms.
    
    Features include:
    - Chirp ridge statistics (slope, duration, peak frequency)
    - Band power ratios
    - Spectral statistics (centroid, bandwidth, entropy)
    - Texture features
    
    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract(spectrogram, times, frequencies)
    """
    
    def __init__(
        self,
        bands: List[Tuple[float, float]] = None,
        ridge_threshold: float = 90.0
    ):
        """
        Initialize the feature extractor.
        
        Args:
            bands: List of (low, high) frequency bands for power ratios
            ridge_threshold: Percentile threshold for ridge extraction
        """
        self.bands = bands or [
            (20, 50),
            (50, 100),
            (100, 200),
            (200, 500)
        ]
        self.ridge_extractor = RidgeExtractor(threshold_percentile=ridge_threshold)
    
    def extract(
        self,
        spectrogram: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract all features from a spectrogram.
        
        Args:
            spectrogram: 2D array (freq x time)
            times: Time array
            frequencies: Frequency array
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        # Ridge features
        ridge_features = self._extract_ridge_features(
            spectrogram, times, frequencies
        )
        features.update(ridge_features)
        
        # Band power features
        band_features = self._extract_band_power_features(
            spectrogram, frequencies
        )
        features.update(band_features)
        
        # Spectral features
        spectral_features = self._extract_spectral_features(
            spectrogram, times, frequencies
        )
        features.update(spectral_features)
        
        # Texture features
        texture_features = self._extract_texture_features(spectrogram)
        features.update(texture_features)
        
        return features
    
    def _extract_ridge_features(
        self,
        spectrogram: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray
    ) -> Dict[str, float]:
        """Extract features from the chirp ridge."""
        features = {}
        
        # Extract ridge
        ridge_times, ridge_freqs = self.ridge_extractor.extract(
            spectrogram, times, frequencies
        )
        
        if len(ridge_times) < 2:
            # Return default values if ridge not found
            return {
                "ridge_duration": 0.0,
                "ridge_slope": 0.0,
                "ridge_slope_std": 0.0,
                "ridge_freq_min": 0.0,
                "ridge_freq_max": 0.0,
                "ridge_freq_mean": 0.0,
                "ridge_freq_range": 0.0,
                "chirp_fit_residual": float('inf')
            }
        
        # Duration
        features["ridge_duration"] = ridge_times[-1] - ridge_times[0]
        
        # Slope statistics
        slope, _ = self.ridge_extractor.compute_slope(ridge_times, ridge_freqs)
        features["ridge_slope"] = slope
        
        # Local slope variation
        if len(ridge_times) > 3:
            local_slopes = np.diff(ridge_freqs) / np.diff(ridge_times)
            features["ridge_slope_std"] = np.std(local_slopes)
        else:
            features["ridge_slope_std"] = 0.0
        
        # Frequency statistics
        features["ridge_freq_min"] = np.min(ridge_freqs)
        features["ridge_freq_max"] = np.max(ridge_freqs)
        features["ridge_freq_mean"] = np.mean(ridge_freqs)
        features["ridge_freq_range"] = features["ridge_freq_max"] - features["ridge_freq_min"]
        
        # Chirp fit quality
        _, _, residual = self.ridge_extractor.fit_chirp(ridge_times, ridge_freqs)
        features["chirp_fit_residual"] = residual
        
        return features
    
    def _extract_band_power_features(
        self,
        spectrogram: np.ndarray,
        frequencies: np.ndarray
    ) -> Dict[str, float]:
        """Extract power in different frequency bands."""
        features = {}
        
        total_power = np.sum(spectrogram ** 2)
        band_powers = []
        
        for i, (low, high) in enumerate(self.bands):
            mask = (frequencies >= low) & (frequencies < high)
            if np.any(mask):
                band_power = np.sum(spectrogram[mask, :] ** 2)
            else:
                band_power = 0.0
            
            features[f"band_power_{low}_{high}"] = band_power
            band_powers.append(band_power)
        
        # Power ratios
        if total_power > 0:
            for i, power in enumerate(band_powers):
                low, high = self.bands[i]
                features[f"band_ratio_{low}_{high}"] = power / total_power
        
        # Low vs high frequency ratio
        if len(band_powers) >= 2:
            low_power = sum(band_powers[:len(band_powers)//2])
            high_power = sum(band_powers[len(band_powers)//2:])
            if high_power > 0:
                features["low_high_ratio"] = low_power / high_power
            else:
                features["low_high_ratio"] = 0.0
        
        return features
    
    def _extract_spectral_features(
        self,
        spectrogram: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray
    ) -> Dict[str, float]:
        """Extract spectral statistics."""
        features = {}
        
        # Time-averaged spectrum
        avg_spectrum = np.mean(spectrogram, axis=1)
        
        # Spectral centroid
        if np.sum(avg_spectrum) > 0:
            centroid = np.sum(frequencies * avg_spectrum) / np.sum(avg_spectrum)
        else:
            centroid = 0.0
        features["spectral_centroid"] = centroid
        
        # Spectral bandwidth
        if np.sum(avg_spectrum) > 0 and centroid > 0:
            bandwidth = np.sqrt(
                np.sum(((frequencies - centroid) ** 2) * avg_spectrum) / np.sum(avg_spectrum)
            )
        else:
            bandwidth = 0.0
        features["spectral_bandwidth"] = bandwidth
        
        # Spectral entropy
        spec_norm = avg_spectrum / (np.sum(avg_spectrum) + 1e-10)
        spec_norm = spec_norm[spec_norm > 0]
        entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10))
        features["spectral_entropy"] = entropy
        
        # Peak frequency
        features["peak_frequency"] = frequencies[np.argmax(avg_spectrum)]
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(avg_spectrum + 1e-10)))
        arithmetic_mean = np.mean(avg_spectrum)
        if arithmetic_mean > 0:
            features["spectral_flatness"] = geometric_mean / arithmetic_mean
        else:
            features["spectral_flatness"] = 0.0
        
        # Time evolution of centroid
        centroids_over_time = []
        for t in range(spectrogram.shape[1]):
            col = spectrogram[:, t]
            if np.sum(col) > 0:
                c = np.sum(frequencies * col) / np.sum(col)
                centroids_over_time.append(c)
        
        if centroids_over_time:
            features["centroid_slope"] = np.polyfit(
                range(len(centroids_over_time)), centroids_over_time, 1
            )[0]
            features["centroid_std"] = np.std(centroids_over_time)
        else:
            features["centroid_slope"] = 0.0
            features["centroid_std"] = 0.0
        
        return features
    
    def _extract_texture_features(
        self,
        spectrogram: np.ndarray
    ) -> Dict[str, float]:
        """Extract texture/shape features from spectrogram."""
        features = {}
        
        # Basic statistics
        features["spec_mean"] = np.mean(spectrogram)
        features["spec_std"] = np.std(spectrogram)
        features["spec_skewness"] = stats.skew(spectrogram.flatten())
        features["spec_kurtosis"] = stats.kurtosis(spectrogram.flatten())
        
        # Gradient features
        grad_t = np.gradient(spectrogram, axis=1)
        grad_f = np.gradient(spectrogram, axis=0)
        
        features["grad_time_mean"] = np.mean(np.abs(grad_t))
        features["grad_freq_mean"] = np.mean(np.abs(grad_f))
        
        # Energy concentration
        sorted_power = np.sort(spectrogram.flatten())[::-1]
        total = np.sum(sorted_power)
        if total > 0:
            # What fraction of pixels contain 90% of the energy?
            cumsum = np.cumsum(sorted_power) / total
            concentration = np.searchsorted(cumsum, 0.9) / len(cumsum)
            features["energy_concentration"] = concentration
        else:
            features["energy_concentration"] = 1.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        # Generate a dummy extraction to get feature names
        dummy_spec = np.random.randn(50, 100)
        dummy_times = np.linspace(0, 1, 100)
        dummy_freqs = np.linspace(20, 500, 50)
        
        features = self.extract(dummy_spec, dummy_times, dummy_freqs)
        return list(features.keys())


def extract_features(
    spectrogram: np.ndarray,
    times: np.ndarray,
    frequencies: np.ndarray,
    bands: List[Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Convenience function to extract features.
    
    Args:
        spectrogram: 2D spectrogram array
        times: Time values
        frequencies: Frequency values
        bands: Frequency bands for power ratios
        
    Returns:
        Dictionary of features
    """
    extractor = FeatureExtractor(bands=bands)
    return extractor.extract(spectrogram, times, frequencies)
