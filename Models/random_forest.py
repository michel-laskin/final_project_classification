"""
Random Forest Classifier for HRV-based Classification

A simpler, more stable alternative to deep learning for small datasets.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path


class HRVRandomForest:
    """
    Random Forest classifier for HRV feature classification.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5,
                 min_samples_leaf=2, random_state=42, class_weight='balanced'):
        """
        Initialize the Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random seed for reproducibility
            class_weight: 'balanced' to handle class imbalance
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1  # Use all CPU cores
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """
        Train the Random Forest model.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            
        Returns:
            self
        """
        # Handle tensor inputs
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if hasattr(y, 'numpy'):
            y = y.numpy()
        
        # Reshape if needed (from [N, 1, F] to [N, F])
        if len(X.shape) == 3:
            X = X.squeeze(1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            predictions: Array of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Handle tensor inputs
        if hasattr(X, 'numpy'):
            X = X.numpy()
        
        # Reshape if needed
        if len(X.shape) == 3:
            X = X.squeeze(1)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            probabilities: Array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Handle tensor inputs
        if hasattr(X, 'numpy'):
            X = X.numpy()
        
        # Reshape if needed
        if len(X.shape) == 3:
            X = X.squeeze(1)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Feature array
            y: True labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Handle tensor inputs
        if hasattr(y, 'numpy'):
            y = y.numpy()
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='binary', zero_division=0),
            'recall': recall_score(y, predictions, average='binary', zero_division=0),
            'f1': f1_score(y, predictions, average='binary', zero_division=0),
            'confusion_matrix': confusion_matrix(y, predictions)
        }
        
        return metrics
    
    def get_feature_importances(self, feature_names=None):
        """
        Get feature importance rankings.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            importances: List of (feature_name, importance) tuples, sorted by importance
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        importances = self.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        
        return [(feature_names[i], importances[i]) for i in sorted_idx]
    
    def save(self, filepath):
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importances': self.feature_importances_
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importances_ = data['feature_importances']
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
        return self

