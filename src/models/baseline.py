"""
Baseline ML models for Phase 1.

Implements Random Forest and XGBoost classifiers for feature-based
gravitational wave classification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class BaselineClassifier:
    """
    Wrapper for baseline ML classifiers.
    
    Supports Random Forest and XGBoost with consistent interface.
    
    Attributes:
        model_type: Type of model ("random_forest" or "xgboost")
        model: The underlying sklearn/xgboost model
        feature_names: Names of input features
        
    Example:
        >>> clf = BaselineClassifier(model_type="random_forest")
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        **kwargs
    ):
        """
        Initialize the classifier.
        
        Args:
            model_type: "random_forest" or "xgboost"
            **kwargs: Model-specific hyperparameters
        """
        self.model_type = model_type
        self.feature_names = None
        self.model = self._create_model(model_type, **kwargs)
    
    def _create_model(self, model_type: str, **kwargs) -> Any:
        """Create the underlying model."""
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_split=kwargs.get("min_samples_split", 5),
                min_samples_leaf=kwargs.get("min_samples_leaf", 2),
                class_weight=kwargs.get("class_weight", "balanced"),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1)
            )
        
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed")
            
            return xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                scale_pos_weight=kwargs.get("scale_pos_weight", 1.0),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1),
                use_label_encoder=False,
                eval_metric="logloss"
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "BaselineClassifier":
        """
        Fit the model.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: Names of features
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif feature_names is not None:
            self.feature_names = feature_names
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
            metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return metrics
    
    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Tuple[float, float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of metric -> (mean, std)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        results = {}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            scores = cross_val_score(
                self.model, X, y, cv=cv, scoring=metric
            )
            results[metric] = (np.mean(scores), np.std(scores))
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model_type == "random_forest":
            importances = self.model.feature_importances_
        elif self.model_type == "xgboost":
            importances = self.model.feature_importances_
        else:
            return {}
        
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        return dict(enumerate(importances))
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaselineClassifier":
        """Load a model from disk."""
        data = joblib.load(path)
        
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.model_type = data["model_type"]
        instance.feature_names = data["feature_names"]
        
        return instance


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> Tuple[BaselineClassifier, Dict[str, float]]:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        feature_names: Feature names
        **kwargs: Model hyperparameters
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    clf = BaselineClassifier(model_type="random_forest", **kwargs)
    clf.fit(X_train, y_train, feature_names=feature_names)
    
    metrics = {}
    if X_val is not None and y_val is not None:
        metrics = clf.evaluate(X_val, y_val)
    
    return clf, metrics


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> Tuple[BaselineClassifier, Dict[str, float]]:
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        feature_names: Feature names
        **kwargs: Model hyperparameters
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed")
    
    # Auto-compute scale_pos_weight for imbalanced data
    if "scale_pos_weight" not in kwargs:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        if n_pos > 0:
            kwargs["scale_pos_weight"] = n_neg / n_pos
    
    clf = BaselineClassifier(model_type="xgboost", **kwargs)
    clf.fit(X_train, y_train, feature_names=feature_names)
    
    metrics = {}
    if X_val is not None and y_val is not None:
        metrics = clf.evaluate(X_val, y_val)
    
    return clf, metrics
