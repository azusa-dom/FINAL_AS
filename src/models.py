import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =============================================================================
# === ENGINE 1: MODELS FOR TABULAR CLINICAL DATA                          ===
# =============================================================================

class ResBlock(nn.Module):
    """A residual block for fully-connected layers."""
    def __init__(self, in_features, out_features):
        # ... (your existing code) ...
        # ...

# ... (all your other tabular model classes: TabularResNet, SimpleMLP, etc.) ...


# =============================================================================
# === ENGINE 2: FEATURE EXTRACTOR FOR MRI DATA                            ===
# =============================================================================

def get_feature_extractor():
    """
    Loads a pre-trained ResNet50 model and removes its final classification
    layer to use it as a feature extractor.

    Sets the model to evaluation mode.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Remove the final fully-connected layer
    layers = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*layers)
    
    # Set the model to evaluation mode
    feature_extractor.eval()
    
    return feature_extractor
