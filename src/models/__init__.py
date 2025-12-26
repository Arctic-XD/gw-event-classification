"""
Models module for GW classification.

Contains baseline ML models (Phase 1) and deep learning models (Phase 2).
"""

from .baseline import train_random_forest, train_xgboost, BaselineClassifier
from .cnn import GWClassifierCNN, create_cnn_model
from .physics_loss import PhysicsInformedLoss, ChirpMassHead

__all__ = [
    "train_random_forest",
    "train_xgboost",
    "BaselineClassifier",
    "GWClassifierCNN",
    "create_cnn_model",
    "PhysicsInformedLoss",
    "ChirpMassHead",
]
