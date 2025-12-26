"""
Chirp ridge extraction from spectrograms.

Extracts the dominant frequency track (chirp ridge) from a spectrogram
for feature computation.
"""

import logging
from typing import Tuple, Optional

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class RidgeExtractor:
    """
    Extract the chirp ridge from a spectrogram.
    
    The chirp ridge is the dominant frequency track that shows the
    characteristic frequency evolution of a CBC signal.
    
    Attributes:
        threshold_percentile: Percentile threshold for ridge detection
        min_duration: Minimum duration for valid ridge (seconds)
    """
    
    def __init__(
        self,
        threshold_percentile: float = 90.0,
        min_duration: float = 0.1,
        smooth_sigma: float = 1.0
    ):
        """
        Initialize the ridge extractor.
        
        Args:
            threshold_percentile: Percentile for power threshold
            min_duration: Minimum ridge duration in seconds
            smooth_sigma: Gaussian smoothing sigma for spectrogram
        """
        self.threshold_percentile = threshold_percentile
        self.min_duration = min_duration
        self.smooth_sigma = smooth_sigma
    
    def extract(
        self,
        spectrogram: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the chirp ridge from a spectrogram.
        
        Args:
            spectrogram: 2D array (freq x time)
            times: Time values for each column
            frequencies: Frequency values for each row
            
        Returns:
            Tuple of (ridge_times, ridge_frequencies)
        """
        # Smooth spectrogram
        if self.smooth_sigma > 0:
            spec_smooth = ndimage.gaussian_filter(spectrogram, sigma=self.smooth_sigma)
        else:
            spec_smooth = spectrogram
        
        # Find peak frequency at each time step
        peak_indices = np.argmax(spec_smooth, axis=0)
        peak_freqs = frequencies[peak_indices]
        peak_powers = np.array([spec_smooth[idx, t] for t, idx in enumerate(peak_indices)])
        
        # Threshold by power
        threshold = np.percentile(peak_powers, self.threshold_percentile)
        valid_mask = peak_powers >= threshold
        
        # Filter by minimum duration
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        min_samples = int(self.min_duration / dt)
        
        # Find contiguous segments above threshold
        valid_mask = self._filter_short_segments(valid_mask, min_samples)
        
        ridge_times = times[valid_mask]
        ridge_freqs = peak_freqs[valid_mask]
        
        return ridge_times, ridge_freqs
    
    def fit_chirp(
        self,
        ridge_times: np.ndarray,
        ridge_freqs: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fit a chirp model to the ridge.
        
        Fits f(t) = f0 * (tc - t)^(-3/8) where tc is the coalescence time.
        
        Args:
            ridge_times: Time values of ridge
            ridge_freqs: Frequency values of ridge
            
        Returns:
            Tuple of (f0, tc, fit_residual)
        """
        if len(ridge_times) < 3:
            return 0.0, 0.0, float('inf')
        
        # Estimate coalescence time as slightly after last point
        tc_init = ridge_times[-1] + 0.1
        
        def chirp_model(t, f0, tc):
            dt = tc - t
            dt = np.maximum(dt, 1e-6)  # Avoid division by zero
            return f0 * dt ** (-3/8)
        
        try:
            # Initial guess
            f0_init = ridge_freqs[0] * (tc_init - ridge_times[0]) ** (3/8)
            
            popt, _ = curve_fit(
                chirp_model,
                ridge_times,
                ridge_freqs,
                p0=[f0_init, tc_init],
                maxfev=1000
            )
            
            # Calculate residual
            fitted = chirp_model(ridge_times, *popt)
            residual = np.sqrt(np.mean((ridge_freqs - fitted) ** 2))
            
            return popt[0], popt[1], residual
            
        except Exception as e:
            logger.warning(f"Chirp fitting failed: {e}")
            return 0.0, 0.0, float('inf')
    
    def compute_slope(
        self,
        ridge_times: np.ndarray,
        ridge_freqs: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute the slope of the ridge using linear regression.
        
        Args:
            ridge_times: Time values
            ridge_freqs: Frequency values
            
        Returns:
            Tuple of (slope, intercept)
        """
        if len(ridge_times) < 2:
            return 0.0, 0.0
        
        # Linear regression
        coeffs = np.polyfit(ridge_times, ridge_freqs, 1)
        return coeffs[0], coeffs[1]  # slope, intercept
    
    def _filter_short_segments(
        self,
        mask: np.ndarray,
        min_length: int
    ) -> np.ndarray:
        """Filter out segments shorter than min_length."""
        result = mask.copy()
        
        # Find contiguous True segments
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        
        # Filter short segments
        for start, end in zip(starts, ends):
            if end - start < min_length:
                result[start:end] = False
        
        return result


def extract_ridge(
    spectrogram: np.ndarray,
    times: np.ndarray,
    frequencies: np.ndarray,
    threshold_percentile: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to extract chirp ridge.
    
    Args:
        spectrogram: 2D array (freq x time)
        times: Time values
        frequencies: Frequency values
        threshold_percentile: Power threshold percentile
        
    Returns:
        Tuple of (ridge_times, ridge_frequencies)
    """
    extractor = RidgeExtractor(threshold_percentile=threshold_percentile)
    return extractor.extract(spectrogram, times, frequencies)
