"""
Synthetic waveform injection using PyCBC.

This module generates synthetic CBC waveforms and injects them into
real LIGO noise for training data augmentation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Try to import PyCBC (may not be available in all environments)
try:
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries as PyCBCTimeSeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    logger.warning("PyCBC not available. Synthetic injection will not work.")


@dataclass
class WaveformParams:
    """Parameters for a CBC waveform."""
    mass1: float  # Solar masses
    mass2: float  # Solar masses
    spin1z: float  # Dimensionless spin
    spin2z: float  # Dimensionless spin
    distance: float  # Mpc
    inclination: float  # Radians
    coa_phase: float  # Radians
    approximant: str
    f_lower: float  # Hz
    delta_t: float  # Sample interval (1/sample_rate)


class SyntheticInjector:
    """
    Generate synthetic CBC waveforms and inject into noise.
    
    This class provides methods to:
    1. Sample waveform parameters from astrophysical priors
    2. Generate waveforms using PyCBC
    3. Inject waveforms into real LIGO noise at target SNR
    
    Attributes:
        config: Configuration dictionary
        sample_rate: Sample rate in Hz
        random_state: Random state for reproducibility
        
    Example:
        >>> injector = SyntheticInjector(config, sample_rate=4096)
        >>> strain, params = injector.generate_bbh_injection(noise_segment, target_snr=15)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        sample_rate: int = 4096,
        random_seed: int = 42
    ):
        """
        Initialize the synthetic injector.
        
        Args:
            config: Configuration dictionary with injection parameters
            sample_rate: Sample rate in Hz
            random_seed: Random seed for reproducibility
        """
        if not PYCBC_AVAILABLE:
            raise ImportError("PyCBC is required for synthetic injection")
        
        self.config = config
        self.sample_rate = sample_rate
        self.delta_t = 1.0 / sample_rate
        self.rng = np.random.RandomState(random_seed)
        
    def sample_bbh_params(self) -> WaveformParams:
        """
        Sample parameters for a BBH waveform from priors.
        
        Returns:
            WaveformParams object with sampled parameters
        """
        cfg = self.config.get("bbh", {})
        
        # Sample masses (power law)
        mass1 = self._sample_power_law(
            cfg.get("mass1", {}).get("min", 5),
            cfg.get("mass1", {}).get("max", 100),
            cfg.get("mass1", {}).get("power_law_index", -2.3)
        )
        mass2 = self._sample_power_law(
            cfg.get("mass2", {}).get("min", 5),
            cfg.get("mass2", {}).get("max", 100),
            cfg.get("mass2", {}).get("power_law_index", -2.3)
        )
        
        # Ensure mass1 >= mass2
        if mass2 > mass1:
            mass1, mass2 = mass2, mass1
        
        # Sample spins
        spin1z = self.rng.uniform(
            cfg.get("spin1z", {}).get("min", -0.99),
            cfg.get("spin1z", {}).get("max", 0.99)
        )
        spin2z = self.rng.uniform(
            cfg.get("spin2z", {}).get("min", -0.99),
            cfg.get("spin2z", {}).get("max", 0.99)
        )
        
        # Sample distance (uniform in volume)
        distance = self._sample_uniform_in_volume(
            self.config.get("injection", {}).get("distance", {}).get("min", 10),
            self.config.get("injection", {}).get("distance", {}).get("max", 1000)
        )
        
        # Sample angles
        inclination = self._sample_sin_distribution(0, np.pi)
        coa_phase = self.rng.uniform(0, 2 * np.pi)
        
        return WaveformParams(
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            distance=distance,
            inclination=inclination,
            coa_phase=coa_phase,
            approximant=cfg.get("approximant", "IMRPhenomD"),
            f_lower=cfg.get("f_lower", 20.0),
            delta_t=self.delta_t
        )
    
    def sample_bns_params(self) -> WaveformParams:
        """Sample parameters for a BNS waveform."""
        cfg = self.config.get("bns", {})
        
        # Sample masses (Gaussian around 1.35 Msun)
        mass1 = self._sample_truncated_gaussian(
            cfg.get("mass1", {}).get("mean", 1.35),
            cfg.get("mass1", {}).get("std", 0.15),
            cfg.get("mass1", {}).get("min", 1.0),
            cfg.get("mass1", {}).get("max", 2.5)
        )
        mass2 = self._sample_truncated_gaussian(
            cfg.get("mass2", {}).get("mean", 1.35),
            cfg.get("mass2", {}).get("std", 0.15),
            cfg.get("mass2", {}).get("min", 1.0),
            cfg.get("mass2", {}).get("max", 2.5)
        )
        
        if mass2 > mass1:
            mass1, mass2 = mass2, mass1
        
        # NS spins are small
        spin1z = self.rng.uniform(-0.05, 0.05)
        spin2z = self.rng.uniform(-0.05, 0.05)
        
        distance = self._sample_uniform_in_volume(10, 500)  # BNS closer
        inclination = self._sample_sin_distribution(0, np.pi)
        coa_phase = self.rng.uniform(0, 2 * np.pi)
        
        return WaveformParams(
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            distance=distance,
            inclination=inclination,
            coa_phase=coa_phase,
            approximant=cfg.get("approximant", "TaylorF2"),
            f_lower=cfg.get("f_lower", 20.0),
            delta_t=self.delta_t
        )
    
    def sample_nsbh_params(self) -> WaveformParams:
        """Sample parameters for an NSBH waveform."""
        cfg = self.config.get("nsbh", {})
        
        # BH mass
        mass1 = self._sample_power_law(
            cfg.get("mass1", {}).get("min", 5),
            cfg.get("mass1", {}).get("max", 50),
            cfg.get("mass1", {}).get("power_law_index", -2.3)
        )
        
        # NS mass
        mass2 = self._sample_truncated_gaussian(
            cfg.get("mass2", {}).get("mean", 1.35),
            cfg.get("mass2", {}).get("std", 0.15),
            cfg.get("mass2", {}).get("min", 1.0),
            cfg.get("mass2", {}).get("max", 2.5)
        )
        
        # BH spin can be large, NS spin small
        spin1z = self.rng.uniform(-0.99, 0.99)
        spin2z = self.rng.uniform(-0.05, 0.05)
        
        distance = self._sample_uniform_in_volume(10, 800)
        inclination = self._sample_sin_distribution(0, np.pi)
        coa_phase = self.rng.uniform(0, 2 * np.pi)
        
        return WaveformParams(
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            distance=distance,
            inclination=inclination,
            coa_phase=coa_phase,
            approximant=cfg.get("approximant", "IMRPhenomNSBH"),
            f_lower=cfg.get("f_lower", 20.0),
            delta_t=self.delta_t
        )
    
    def generate_waveform(self, params: WaveformParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a waveform from parameters.
        
        Args:
            params: WaveformParams object
            
        Returns:
            Tuple of (hp, hc) - plus and cross polarizations
        """
        hp, hc = get_td_waveform(
            mass1=params.mass1,
            mass2=params.mass2,
            spin1z=params.spin1z,
            spin2z=params.spin2z,
            distance=params.distance,
            inclination=params.inclination,
            coa_phase=params.coa_phase,
            approximant=params.approximant,
            f_lower=params.f_lower,
            delta_t=params.delta_t
        )
        
        return np.array(hp), np.array(hc)
    
    def inject_into_noise(
        self,
        noise: np.ndarray,
        waveform: np.ndarray,
        injection_time: Optional[float] = None,
        target_snr: Optional[float] = None,
        psd: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Inject a waveform into noise segment.
        
        Args:
            noise: Noise data array
            waveform: Waveform to inject
            injection_time: Time (in samples) to place the merger
            target_snr: Target SNR (will rescale waveform)
            psd: Power spectral density for SNR calculation
            
        Returns:
            Noise + injected signal
        """
        # Default injection in middle of segment
        if injection_time is None:
            injection_time = len(noise) // 2
        
        # Pad or truncate waveform to fit
        if len(waveform) > len(noise):
            waveform = waveform[-len(noise):]
        
        # Create injection array
        injected = noise.copy()
        
        # Calculate where to place waveform (merger at injection_time)
        start_idx = max(0, injection_time - len(waveform))
        end_idx = injection_time
        wf_start = max(0, len(waveform) - injection_time)
        
        # Rescale for target SNR if specified
        if target_snr is not None and psd is not None:
            current_snr = self._calculate_snr(waveform, psd)
            if current_snr > 0:
                waveform = waveform * (target_snr / current_snr)
        
        # Add waveform to noise
        injected[start_idx:end_idx] += waveform[wf_start:]
        
        return injected
    
    def generate_injection(
        self,
        noise: np.ndarray,
        source_class: str,
        target_snr: float = 15.0,
        psd: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a complete injection (waveform + noise).
        
        Args:
            noise: Background noise array
            source_class: "bbh", "bns", or "nsbh"
            target_snr: Target SNR for injection
            psd: PSD for SNR calculation
            
        Returns:
            Tuple of (injected_strain, parameters_dict)
        """
        # Sample parameters
        if source_class == "bbh":
            params = self.sample_bbh_params()
        elif source_class == "bns":
            params = self.sample_bns_params()
        elif source_class == "nsbh":
            params = self.sample_nsbh_params()
        else:
            raise ValueError(f"Unknown source class: {source_class}")
        
        # Generate waveform
        hp, hc = self.generate_waveform(params)
        
        # Use plus polarization (simplified - real would use antenna patterns)
        waveform = hp
        
        # Inject into noise
        injected = self.inject_into_noise(
            noise=noise,
            waveform=waveform,
            target_snr=target_snr,
            psd=psd
        )
        
        # Return parameters as dict
        params_dict = {
            "mass1": params.mass1,
            "mass2": params.mass2,
            "spin1z": params.spin1z,
            "spin2z": params.spin2z,
            "distance": params.distance,
            "inclination": params.inclination,
            "coa_phase": params.coa_phase,
            "approximant": params.approximant,
            "target_snr": target_snr,
            "source_class": source_class
        }
        
        return injected, params_dict
    
    # Helper methods for sampling distributions
    def _sample_power_law(self, min_val: float, max_val: float, index: float) -> float:
        """Sample from power law distribution."""
        if index == -1:
            return np.exp(self.rng.uniform(np.log(min_val), np.log(max_val)))
        else:
            exp = index + 1
            u = self.rng.uniform(0, 1)
            return (u * (max_val**exp - min_val**exp) + min_val**exp) ** (1/exp)
    
    def _sample_uniform_in_volume(self, min_dist: float, max_dist: float) -> float:
        """Sample distance uniform in volume (p(d) ∝ d^2)."""
        u = self.rng.uniform(0, 1)
        return (u * (max_dist**3 - min_dist**3) + min_dist**3) ** (1/3)
    
    def _sample_sin_distribution(self, min_val: float, max_val: float) -> float:
        """Sample from sin distribution (for inclination)."""
        u = self.rng.uniform(np.cos(max_val), np.cos(min_val))
        return np.arccos(u)
    
    def _sample_truncated_gaussian(
        self, mean: float, std: float, min_val: float, max_val: float
    ) -> float:
        """Sample from truncated Gaussian."""
        while True:
            val = self.rng.normal(mean, std)
            if min_val <= val <= max_val:
                return val
    
    def _calculate_snr(self, waveform: np.ndarray, psd: np.ndarray) -> float:
        """Calculate SNR of waveform given PSD."""
        # Simplified SNR calculation
        wf_fft = np.fft.rfft(waveform)
        snr_sq = 4 * np.sum(np.abs(wf_fft)**2 / psd) / len(waveform)
        return np.sqrt(snr_sq)
