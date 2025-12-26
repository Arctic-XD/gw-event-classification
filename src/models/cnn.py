"""
CNN models for Phase 2 deep learning.

Implements ResNet-based classifiers optimized for A100 training.
"""

import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Try to import timm for pretrained models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available. Using basic CNN architecture.")


class GWClassifierCNN(nn.Module):
    """
    CNN classifier for gravitational wave spectrograms.
    
    Uses a pretrained backbone (ResNet, EfficientNet) with a custom
    classification head. Optimized for A100 training with mixed precision.
    
    Attributes:
        backbone: Feature extraction backbone
        classifier: Classification head
        num_classes: Number of output classes
        
    Example:
        >>> model = GWClassifierCNN(backbone="resnet50", num_classes=2)
        >>> output = model(spectrogram_batch)
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        input_channels: int = 1
    ):
        """
        Initialize the CNN classifier.
        
        Args:
            backbone: Backbone architecture name
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            dropout: Dropout rate in classifier head
            input_channels: Number of input channels (1 for grayscale)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Create backbone
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                in_chans=input_channels
            )
            num_features = self.backbone.num_features
        else:
            # Fallback to basic CNN
            self.backbone = self._create_basic_backbone(input_channels)
            num_features = 512
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _create_basic_backbone(self, input_channels: int) -> nn.Module:
        """Create a basic CNN backbone if timm is not available."""
        return nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _init_classifier(self) -> None:
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        features = self.backbone(x)
        
        # Handle different backbone output shapes
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: Tensor) -> Tensor:
        """Extract features without classification."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        return features


class GWClassifierWithChirpMass(nn.Module):
    """
    CNN classifier with auxiliary chirp mass prediction head.
    
    This model has two outputs:
    1. Classification: BBH vs NS-present
    2. Chirp mass estimation: For physics-informed loss
    
    Attributes:
        backbone: Shared feature backbone
        classifier: Classification head
        chirp_mass_head: Chirp mass regression head
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        input_channels: int = 1
    ):
        """Initialize the multi-task model."""
        super().__init__()
        
        self.num_classes = num_classes
        
        # Shared backbone
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                in_chans=input_channels
            )
            num_features = self.backbone.num_features
        else:
            self.backbone = GWClassifierCNN(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                input_channels=input_channels
            )._create_basic_backbone(input_channels)
            num_features = 512
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Chirp mass regression head
        self.chirp_mass_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus()  # Chirp mass must be positive
        )
    
    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input spectrogram tensor
            
        Returns:
            Tuple of (class_logits, chirp_mass_prediction)
        """
        # Extract features
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        
        # Classification
        logits = self.classifier(features)
        
        # Chirp mass prediction
        chirp_mass = self.chirp_mass_head(features)
        
        return logits, chirp_mass


def create_cnn_model(
    config: Dict,
    device: torch.device = None
) -> nn.Module:
    """
    Create a CNN model from configuration.
    
    Args:
        config: Model configuration dictionary
        device: Target device
        
    Returns:
        Initialized model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get config values
    backbone = config.get("backbone", "resnet50")
    num_classes = config.get("num_classes", 2)
    pretrained = config.get("pretrained", True)
    dropout = config.get("dropout", 0.5)
    physics_informed = config.get("physics_informed", False)
    
    if physics_informed:
        model = GWClassifierWithChirpMass(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        model = GWClassifierCNN(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    
    model = model.to(device)
    
    logger.info(f"Created {backbone} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model
