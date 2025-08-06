#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imaging_net.py

ImagingNet implementation for MRI analysis
ResNet-18 + Logistic Regression architecture as described in the paper
"""

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Tuple, Optional


class ImagingNet:
    """
    ImagingNet - ResNet-18 + Logistic Regression model for MRI analysis
    
    Architecture:
    1. ResNet-18 (frozen) for feature extraction
    2. Logistic Regression for classification
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2, random_state: int = 42):
        """
        Initialize ImagingNet
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (2 for AS vs Healthy)
            random_state: Random seed for reproducibility
        """
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.random_state = random_state
        
        # Initialize ResNet-18 feature extractor
        self.feature_extractor = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Freeze ResNet parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Initialize Logistic Regression classifier
        self.classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )
        
        self.is_fitted = False
        self.feature_dim = 512  # ResNet-18 output dimension
        
    def extract_features(self, X: torch.Tensor) -> np.ndarray:
        """
        Extract features using ResNet-18
        
        Args:
            X: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature array of shape (batch_size, 512)
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(X)
            # Flatten features
            features = features.view(features.size(0), -1)
            return features.cpu().numpy()
    
    def fit(self, X: torch.Tensor, y: np.ndarray) -> 'ImagingNet':
        """
        Train the ImagingNet model
        
        Args:
            X: Input tensor of shape (batch_size, 3, height, width)
            y: Target labels of shape (batch_size,)
            
        Returns:
            Self for method chaining
        """
        # Extract features
        features = self.extract_features(X)
        
        # Train Logistic Regression classifier
        self.classifier.fit(features, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Probability array of shape (batch_size, num_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Extract features
        features = self.extract_features(X)
        
        # Get probabilities
        return self.classifier.predict_proba(features)
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Get class predictions
        
        Args:
            X: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Class predictions of shape (batch_size,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Extract features
        features = self.extract_features(X)
        
        # Get predictions
        return self.classifier.predict(features)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from Logistic Regression
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return np.abs(self.classifier.coef_[0])
    
    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """
        Get Logistic Regression coefficients and intercept
        
        Returns:
            Tuple of (coefficients, intercept)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        return self.classifier.coef_[0], self.classifier.intercept_[0]
    
    def save_model(self, filepath: str):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        model_data = {
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier': self.classifier,
            'pretrained': self.pretrained,
            'num_classes': self.num_classes,
            'random_state': self.random_state,
            'feature_dim': self.feature_dim
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ImagingNet':
        """
        Load the model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ImagingNet instance
        """
        import joblib
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            pretrained=model_data['pretrained'],
            num_classes=model_data['num_classes'],
            random_state=model_data['random_state']
        )
        
        # Load state
        instance.feature_extractor.load_state_dict(model_data['feature_extractor_state_dict'])
        instance.classifier = model_data['classifier']
        instance.feature_dim = model_data['feature_dim']
        instance.is_fitted = True
        
        return instance


def create_imaging_net(pretrained: bool = True, random_state: int = 42) -> ImagingNet:
    """
    Factory function to create ImagingNet instance
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        random_state: Random seed for reproducibility
        
    Returns:
        ImagingNet instance
    """
    return ImagingNet(pretrained=pretrained, random_state=random_state) 