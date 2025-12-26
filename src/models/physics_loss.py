"""
Physics-informed loss functions.

Implements loss functions that incorporate gravitational wave physics
constraints, specifically the chirp mass - frequency relationship.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ChirpMassHead(nn.Module):
    """
    Auxiliary head for chirp mass prediction.
    
    Predicts chirp mass from CNN features, which can then be used
    for physics-informed loss computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize the chirp mass head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive output
        )
    
    def forward(self, features: Tensor) -> Tensor:
        """
        Predict chirp mass from features.
        
        Args:
            features: Feature tensor (batch, feature_dim)
            
        Returns:
            Chirp mass predictions (batch, 1)
        """
        return self.head(features)


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with physics-informed constraints.
    
    Total loss = classification_loss + λ * physics_loss
    
    Physics constraint: f_peak ∝ M_c^(-5/8)
    
    For a CBC signal, the relationship between peak frequency and chirp mass
    is approximately:
        f_peak ≈ c / (6^(3/2) * π * G * M_c)
    
    We enforce this relationship through a soft constraint in the loss.
    
    Attributes:
        classification_weight: Weight for classification loss
        physics_weight: Weight for physics constraint loss
        chirp_mass_weight: Weight for chirp mass regression loss
        
    Example:
        >>> loss_fn = PhysicsInformedLoss(physics_weight=0.1)
        >>> loss = loss_fn(logits, labels, pred_chirp_mass, true_chirp_mass, peak_freq)
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        physics_weight: float = 0.1,
        chirp_mass_weight: float = 0.1,
        label_smoothing: float = 0.1,
        k_constant: float = 1.0
    ):
        """
        Initialize the physics-informed loss.
        
        Args:
            classification_weight: Weight for cross-entropy loss
            physics_weight: Weight for physics constraint
            chirp_mass_weight: Weight for chirp mass regression
            label_smoothing: Label smoothing for classification
            k_constant: Calibration constant for physics constraint
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.physics_weight = physics_weight
        self.chirp_mass_weight = chirp_mass_weight
        self.k_constant = k_constant
        
        # Classification loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Chirp mass regression loss
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        pred_chirp_mass: Optional[Tensor] = None,
        true_chirp_mass: Optional[Tensor] = None,
        peak_frequency: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute the combined loss.
        
        Args:
            logits: Classification logits (batch, num_classes)
            labels: True labels (batch,)
            pred_chirp_mass: Predicted chirp mass (batch, 1)
            true_chirp_mass: True chirp mass (batch, 1) - from synthetic data
            peak_frequency: Peak frequency from spectrogram (batch, 1)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=logits.device)
        
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        losses["classification"] = cls_loss.item()
        total_loss = total_loss + self.classification_weight * cls_loss
        
        # Chirp mass regression loss (if ground truth available)
        if pred_chirp_mass is not None and true_chirp_mass is not None:
            # Log-space loss for better scaling across mass ranges
            pred_log = torch.log(pred_chirp_mass + 1e-6)
            true_log = torch.log(true_chirp_mass + 1e-6)
            chirp_loss = self.mse_loss(pred_log, true_log)
            losses["chirp_mass"] = chirp_loss.item()
            total_loss = total_loss + self.chirp_mass_weight * chirp_loss
        
        # Physics constraint loss
        if pred_chirp_mass is not None and peak_frequency is not None:
            physics_loss = self._physics_constraint_loss(
                pred_chirp_mass, peak_frequency
            )
            losses["physics"] = physics_loss.item()
            total_loss = total_loss + self.physics_weight * physics_loss
        
        losses["total"] = total_loss.item()
        
        return total_loss, losses
    
    def _physics_constraint_loss(
        self,
        chirp_mass: Tensor,
        peak_frequency: Tensor
    ) -> Tensor:
        """
        Compute physics constraint loss.
        
        The constraint enforces: f_peak ∝ M_c^(-5/8)
        
        We compute this as:
        L_physics = || f_peak * M_c^(5/8) - k ||^2
        
        where k is a calibration constant.
        
        Args:
            chirp_mass: Predicted chirp mass (solar masses)
            peak_frequency: Peak frequency from spectrogram (Hz)
            
        Returns:
            Physics constraint loss
        """
        # Compute the product f * M_c^(5/8)
        # This should be approximately constant for GW signals
        product = peak_frequency * torch.pow(chirp_mass + 1e-6, 5/8)
        
        # Loss is deviation from expected constant
        # Normalize by expected value to make loss scale-invariant
        physics_loss = torch.mean((product / self.k_constant - 1.0) ** 2)
        
        return physics_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard ones:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Useful when training on imbalanced BBH vs NS-present data.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits (batch, num_classes)
            targets: Labels (batch,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def create_loss_function(config: Dict) -> nn.Module:
    """
    Create a loss function from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Initialized loss function
    """
    loss_type = config.get("type", "cross_entropy")
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=config.get("label_smoothing", 0.0)
        )
    
    elif loss_type == "focal":
        return FocalLoss(
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0)
        )
    
    elif loss_type == "physics_informed":
        return PhysicsInformedLoss(
            classification_weight=config.get("classification_weight", 1.0),
            physics_weight=config.get("physics_weight", 0.1),
            chirp_mass_weight=config.get("chirp_mass_weight", 0.1),
            label_smoothing=config.get("label_smoothing", 0.1),
            k_constant=config.get("k_constant", 1.0)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
