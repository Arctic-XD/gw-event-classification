"""
GWpy-based data fetcher for GWOSC strain data.

This module provides utilities to download gravitational wave strain data
from the Gravitational-Wave Open Science Center (GWOSC).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class GWDataFetcher:
    """
    Fetches gravitational wave strain data from GWOSC.
    
    This class provides methods to download strain data for individual events
    or batches of events from GWTC catalogs.
    
    Attributes:
        cache_dir: Directory to cache downloaded data
        sample_rate: Target sample rate in Hz
        verbose: Whether to print progress messages
    
    Example:
        >>> fetcher = GWDataFetcher(cache_dir="data/raw", sample_rate=4096)
        >>> strain = fetcher.fetch_event("GW150914", detector="L1", duration=32)
    """
    
    def __init__(
        self,
        cache_dir: str = "data/raw",
        sample_rate: int = 4096,
        verbose: bool = True
    ):
        """
        Initialize the data fetcher.
        
        Args:
            cache_dir: Directory to store downloaded strain files
            sample_rate: Sample rate in Hz (4096 or 16384)
            verbose: Print progress messages
        """
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.verbose = verbose
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_event(
        self,
        event_name: str,
        detector: str = "L1",
        gps_time: Optional[float] = None,
        duration: float = 32.0,
        cache: bool = True
    ) -> Optional[TimeSeries]:
        """
        Fetch strain data for a single event.
        
        Args:
            event_name: Name of the event (e.g., "GW150914")
            detector: Detector name ("L1", "H1", or "V1")
            gps_time: GPS time of the event (if None, will look up)
            duration: Duration in seconds before/after merger
            cache: Whether to cache the downloaded data
            
        Returns:
            TimeSeries object containing the strain data, or None if failed
        """
        # Check cache first
        cache_path = self._get_cache_path(event_name, detector)
        if cache and cache_path.exists():
            if self.verbose:
                logger.info(f"Loading {event_name} ({detector}) from cache")
            return TimeSeries.read(cache_path)
        
        # Fetch from GWOSC
        try:
            if self.verbose:
                logger.info(f"Fetching {event_name} ({detector}) from GWOSC...")
            
            # If GPS time not provided, we need to look it up
            if gps_time is None:
                raise ValueError(f"GPS time required for {event_name}")
            
            # Calculate start and end times
            start = gps_time - duration
            end = gps_time + duration
            
            # Fetch open data
            strain = TimeSeries.fetch_open_data(
                detector,
                start,
                end,
                sample_rate=self.sample_rate,
                verbose=self.verbose
            )
            
            # Cache if requested
            if cache:
                strain.write(cache_path)
                if self.verbose:
                    logger.info(f"Cached to {cache_path}")
            
            return strain
            
        except Exception as e:
            logger.error(f"Failed to fetch {event_name} ({detector}): {e}")
            return None
    
    def fetch_events_from_manifest(
        self,
        manifest_path: str,
        detector: str = "L1",
        duration: float = 32.0,
        max_events: Optional[int] = None
    ) -> Dict[str, TimeSeries]:
        """
        Fetch strain data for all events in a manifest CSV.
        
        Args:
            manifest_path: Path to CSV with event_name, gps_time columns
            detector: Detector to fetch
            duration: Duration around merger
            max_events: Maximum number of events to fetch (for testing)
            
        Returns:
            Dictionary mapping event names to TimeSeries objects
        """
        # Load manifest
        manifest = pd.read_csv(manifest_path)
        
        if max_events:
            manifest = manifest.head(max_events)
        
        results = {}
        for _, row in manifest.iterrows():
            event_name = row["event_name"]
            gps_time = row["gps_time"]
            
            strain = self.fetch_event(
                event_name=event_name,
                detector=detector,
                gps_time=gps_time,
                duration=duration
            )
            
            if strain is not None:
                results[event_name] = strain
        
        logger.info(f"Successfully fetched {len(results)}/{len(manifest)} events")
        return results
    
    def fetch_noise_segment(
        self,
        detector: str,
        start_time: float,
        duration: float,
        cache: bool = True
    ) -> Optional[TimeSeries]:
        """
        Fetch a noise segment (no known events) for injection.
        
        Args:
            detector: Detector name
            start_time: GPS start time
            duration: Duration in seconds
            cache: Whether to cache
            
        Returns:
            TimeSeries of noise data
        """
        try:
            strain = TimeSeries.fetch_open_data(
                detector,
                start_time,
                start_time + duration,
                sample_rate=self.sample_rate,
                verbose=self.verbose
            )
            return strain
        except Exception as e:
            logger.error(f"Failed to fetch noise segment: {e}")
            return None
    
    def _get_cache_path(self, event_name: str, detector: str) -> Path:
        """Get the cache file path for an event."""
        return self.cache_dir / f"{event_name}_{detector}_{self.sample_rate}Hz.hdf5"


def get_gwtc_events(catalog: str = "GWTC-3") -> pd.DataFrame:
    """
    Get event list from GWTC catalog.
    
    This is a placeholder - in practice you would query GWOSC API
    or load from a pre-downloaded catalog file.
    
    Args:
        catalog: Which GWTC catalog ("GWTC-1", "GWTC-2", "GWTC-2.1", "GWTC-3")
        
    Returns:
        DataFrame with event information
    """
    # TODO: Implement GWOSC API query or load from file
    raise NotImplementedError("Implement GWOSC catalog query")
