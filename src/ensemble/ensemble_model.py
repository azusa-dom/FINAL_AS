#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_model.py

Ensemble model implementation for DDI-AS framework
Simple averaging fusion of ClinicalNet and ImagingNet probabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib


class EnsembleModel:
    """
    Ensemble model combining ClinicalNet and ImagingNet predictions
    
    Formula: P_Ensemble = 0.5 × P_ClinicalNet + 0.5 × P_ImagingNet
    """
    
    def __init__(self, clinical_weight: float = 0.5, imaging_weight: float = 0.5):
        """
        Initialize ensemble model
        
        Args:
            clinical_weight: Weight for ClinicalNet predictions
            imaging_weight: Weight for ImagingNet predictions
        """
        self.clinical_weight = clinical_weight
        self.imaging_weight = imaging_weight
        
        # Validate weights
        if abs(clinical_weight + imaging_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.clinical_model = None
        self.imaging_model = None
        self.is_fitted = False
        
    def set_models(self, clinical_model, imaging_model):
        """
        Set the individual models
        
        Args:
            clinical_model: Trained ClinicalNet model
            imaging_model: Trained ImagingNet model
        """
        self.clinical_model = clinical_model
        self.imaging_model = imaging_model
        self.is_fitted = True
        
    def predict_proba(self, clinical_data, imaging_data) -> np.ndarray:
        """
        Get ensemble probability predictions
        
        Args:
            clinical_data: Input data for ClinicalNet
            imaging_data: Input data for ImagingNet
            
        Returns:
            Ensemble probabilities of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Models must be set before making predictions")
        
        # Get individual predictions
        clinical_probs = self.clinical_model.predict_proba(clinical_data)
        imaging_probs = self.imaging_model.predict_proba(imaging_data)
        
        # Weighted average
        ensemble_probs = (
            self.clinical_weight * clinical_probs + 
            self.imaging_weight * imaging_probs
        )
        
        return ensemble_probs
    
    def predict(self, clinical_data, imaging_data) -> np.ndarray:
        """
        Get ensemble class predictions
        
        Args:
            clinical_data: Input data for ClinicalNet
            imaging_data: Input data for ImagingNet
            
        Returns:
            Ensemble class predictions
        """
        ensemble_probs = self.predict_proba(clinical_data, imaging_data)
        return np.argmax(ensemble_probs, axis=1)
    
    def evaluate(self, clinical_data, imaging_data, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble model performance
        
        Args:
            clinical_data: Input data for ClinicalNet
            imaging_data: Input data for ImagingNet
            y_true: True labels
            
        Returns:
            Dictionary of performance metrics
        """
        ensemble_probs = self.predict_proba(clinical_data, imaging_data)
        ensemble_preds = self.predict(clinical_data, imaging_data)
        
        # Calculate metrics
        auc = roc_auc_score(y_true, ensemble_probs[:, 1])
        accuracy = accuracy_score(y_true, ensemble_preds)
        logloss = log_loss(y_true, ensemble_probs)
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'log_loss': logloss
        }
    
    def compare_models(self, clinical_data, imaging_data, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare individual models and ensemble performance
        
        Args:
            clinical_data: Input data for ClinicalNet
            imaging_data: Input data for ImagingNet
            y_true: True labels
            
        Returns:
            Dictionary of performance metrics for each model
        """
        results = {}
        
        # ClinicalNet performance
        clinical_probs = self.clinical_model.predict_proba(clinical_data)
        clinical_preds = self.clinical_model.predict(clinical_data)
        results['clinical_net'] = {
            'auc': roc_auc_score(y_true, clinical_probs[:, 1]),
            'accuracy': accuracy_score(y_true, clinical_preds),
            'log_loss': log_loss(y_true, clinical_probs)
        }
        
        # ImagingNet performance
        imaging_probs = self.imaging_model.predict_proba(imaging_data)
        imaging_preds = self.imaging_model.predict(imaging_data)
        results['imaging_net'] = {
            'auc': roc_auc_score(y_true, imaging_probs[:, 1]),
            'accuracy': accuracy_score(y_true, imaging_preds),
            'log_loss': log_loss(y_true, imaging_probs)
        }
        
        # Ensemble performance
        results['ensemble'] = self.evaluate(clinical_data, imaging_data, y_true)
        
        return results
    
    def save_ensemble(self, filepath: str):
        """
        Save ensemble model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Models must be set before saving")
        
        model_data = {
            'clinical_weight': self.clinical_weight,
            'imaging_weight': self.imaging_weight,
            'clinical_model': self.clinical_model,
            'imaging_model': self.imaging_model
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_ensemble(cls, filepath: str) -> 'EnsembleModel':
        """
        Load ensemble model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded EnsembleModel instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            clinical_weight=model_data['clinical_weight'],
            imaging_weight=model_data['imaging_weight']
        )
        
        # Load models
        instance.clinical_model = model_data['clinical_model']
        instance.imaging_model = model_data['imaging_model']
        instance.is_fitted = True
        
        return instance


def create_ensemble_model(clinical_weight: float = 0.5, imaging_weight: float = 0.5) -> EnsembleModel:
    """
    Factory function to create ensemble model
    
    Args:
        clinical_weight: Weight for ClinicalNet predictions
        imaging_weight: Weight for ImagingNet predictions
        
    Returns:
        EnsembleModel instance
    """
    return EnsembleModel(clinical_weight=clinical_weight, imaging_weight=imaging_weight)


def late_fusion_predictions(clinical_probs: np.ndarray, imaging_probs: np.ndarray, 
                          clinical_weight: float = 0.5, imaging_weight: float = 0.5) -> np.ndarray:
    """
    Perform late fusion of predictions using simple averaging
    
    Args:
        clinical_probs: ClinicalNet probability predictions
        imaging_probs: ImagingNet probability predictions
        clinical_weight: Weight for ClinicalNet predictions
        imaging_weight: Weight for ImagingNet predictions
        
    Returns:
        Fused probability predictions
    """
    return clinical_weight * clinical_probs + imaging_weight * imaging_probs 