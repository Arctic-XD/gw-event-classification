"""
PyTorch Dataset classes for GW classification.

Provides dataset classes for loading spectrograms and features
for both Phase 1 (feature-based) and Phase 2 (CNN) models.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)


class GWDataset(Dataset):
    """
    Base dataset for gravitational wave classification.
    
    This dataset loads pre-computed features for Phase 1 models.
    """
    
    def __init__(
        self,
        manifest_path: str,
        features_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            manifest_path: Path to event manifest CSV
            features_dir: Directory containing feature files
            split: Which split to use ("train", "val", "test")
            transform: Optional transform to apply to features
        """
        self.features_dir = Path(features_dir)
        self.transform = transform
        
        # Load manifest
        manifest = pd.read_csv(manifest_path)
        
        # Filter by split if column exists
        if "split" in manifest.columns:
            self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        else:
            self.manifest = manifest
        
        # Map class labels to integers
        self.class_to_idx = {"BBH": 0, "NS-present": 1}
        
        logger.info(f"Loaded {len(self.manifest)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        row = self.manifest.iloc[idx]
        event_name = row["event_name"]
        
        # Load features
        feature_path = self.features_dir / f"{event_name}_features.npy"
        features = np.load(feature_path)
        
        # Get label
        label = self.class_to_idx[row["class_label"]]
        
        # Apply transform
        if self.transform:
            features = self.transform(features)
        
        return torch.tensor(features, dtype=torch.float32), label


class SpectrogramDataset(Dataset):
    """
    Dataset for loading spectrogram images for CNN models.
    
    This dataset is optimized for Phase 2 training on A100.
    """
    
    def __init__(
        self,
        manifest_path: str,
        spectrograms_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        cache_in_memory: bool = False
    ):
        """
        Initialize the spectrogram dataset.
        
        Args:
            manifest_path: Path to manifest CSV
            spectrograms_dir: Directory with spectrogram images
            split: Data split
            transform: Image transforms (torchvision)
            image_size: Target image size
            cache_in_memory: Cache all images in memory (for A100)
        """
        self.spectrograms_dir = Path(spectrograms_dir)
        self.transform = transform
        self.image_size = image_size
        self.cache_in_memory = cache_in_memory
        
        # Load manifest
        manifest = pd.read_csv(manifest_path)
        if "split" in manifest.columns:
            self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        else:
            self.manifest = manifest
        
        self.class_to_idx = {"BBH": 0, "NS-present": 1}
        
        # Cache images if requested (useful for A100 with large memory)
        self.cache = {}
        if cache_in_memory:
            logger.info("Caching all spectrograms in memory...")
            for idx in range(len(self.manifest)):
                self.cache[idx] = self._load_image(idx)
            logger.info(f"Cached {len(self.cache)} images")
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
        """
        # Get from cache or load
        if self.cache_in_memory and idx in self.cache:
            image = self.cache[idx]
        else:
            image = self._load_image(idx)
        
        # Get label
        row = self.manifest.iloc[idx]
        label = self.class_to_idx.get(row["class_label"], 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.tensor(np.array(image), dtype=torch.float32)
            if image.dim() == 2:
                image = image.unsqueeze(0)  # Add channel dim
            image = image / 255.0  # Normalize to [0, 1]
        
        return image, label
    
    def _load_image(self, idx: int) -> Image.Image:
        """Load a spectrogram image."""
        row = self.manifest.iloc[idx]
        event_name = row["event_name"]
        
        # Try different file formats
        for ext in [".png", ".npy", ".jpg"]:
            image_path = self.spectrograms_dir / f"{event_name}_spectrogram{ext}"
            if image_path.exists():
                if ext == ".npy":
                    arr = np.load(image_path)
                    # Normalize to 0-255 for PIL
                    arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
                    return Image.fromarray(arr)
                else:
                    return Image.open(image_path).convert("L")  # Grayscale
        
        raise FileNotFoundError(f"No spectrogram found for {event_name}")


class SyntheticDataset(Dataset):
    """
    Dataset for synthetic injections.
    
    Can load pre-generated synthetic samples or generate on-the-fly.
    """
    
    def __init__(
        self,
        manifest_path: str,
        spectrograms_dir: str,
        transform: Optional[Callable] = None
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            manifest_path: Path to synthetic manifest CSV
            spectrograms_dir: Directory with synthetic spectrograms
            transform: Image transforms
        """
        self.spectrograms_dir = Path(spectrograms_dir)
        self.transform = transform
        
        self.manifest = pd.read_csv(manifest_path)
        self.class_to_idx = {"bbh": 0, "ns_present": 1, "bns": 1, "nsbh": 1}
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Get a sample with metadata.
        
        Returns:
            Tuple of (image, label, metadata_dict)
        """
        row = self.manifest.iloc[idx]
        
        # Load spectrogram
        spec_path = self.spectrograms_dir / row["spectrogram_file"]
        if spec_path.suffix == ".npy":
            spec = np.load(spec_path)
            spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        else:
            image = Image.open(spec_path).convert("L")
            spec = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)
            spec = spec / 255.0
        
        # Apply transform
        if self.transform:
            spec = self.transform(spec)
        
        # Label
        label = self.class_to_idx.get(row["source_class"], 0)
        
        # Metadata
        metadata = {
            "mass1": row.get("mass1", 0),
            "mass2": row.get("mass2", 0),
            "snr": row.get("target_snr", 0),
            "chirp_mass": row.get("chirp_mass", 0)
        }
        
        return spec, label, metadata


def create_dataloaders(
    manifest_path: str,
    spectrograms_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
    pin_memory: bool = True,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Optimized for A100 training.
    
    Args:
        manifest_path: Path to manifest CSV
        spectrograms_dir: Directory with spectrograms
        batch_size: Batch size (128-256 for A100)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        transform_train: Transforms for training
        transform_val: Transforms for validation/test
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = SpectrogramDataset(
        manifest_path=manifest_path,
        spectrograms_dir=spectrograms_dir,
        split="train",
        transform=transform_train
    )
    
    val_dataset = SpectrogramDataset(
        manifest_path=manifest_path,
        spectrograms_dir=spectrograms_dir,
        split="val",
        transform=transform_val
    )
    
    test_dataset = SpectrogramDataset(
        manifest_path=manifest_path,
        spectrograms_dir=spectrograms_dir,
        split="test",
        transform=transform_val
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
