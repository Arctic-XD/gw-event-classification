"""
Data module for GW event classification.

Handles data fetching, preprocessing, synthetic injection, and PyTorch datasets.
"""

from .fetcher import GWDataFetcher
from .preprocessing import preprocess_strain, bandpass_filter, whiten
from .injection import SyntheticInjector
from .dataset import GWDataset, SpectrogramDataset

__all__ = [
    "GWDataFetcher",
    "preprocess_strain",
    "bandpass_filter", 
    "whiten",
    "SyntheticInjector",
    "GWDataset",
    "SpectrogramDataset",
]
