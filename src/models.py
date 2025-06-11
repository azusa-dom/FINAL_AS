import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- ADD THIS NEW CLASS -----------------
class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Shortcut connection to match dimensions if necessary
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += identity # Add the residual
        out = F.relu(out)
        
        return out

class SimpleResNet(nn.Module):
    def __init__(self, input_dim=21, num_classes=2): # Assuming input_dim might need adjustment
        super(SimpleResNet, self).__init__()
        # Determine input dimension dynamically later if needed
        self.fc_in = nn.Linear(input_dim, 64)
        self.bn_in = nn.BatchNorm1d(64)
        
        self.layer1 = ResBlock(64, 128)
        self.layer2 = ResBlock(128, 256)
        self.layer3 = ResBlock(256, 128)
        
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # The training script passes features directly.
        # If your input has a different dimension, you'll need to adjust the `input_dim` above.
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc_out(x)
        return x
# ----------------- END OF NEW CLASS -----------------


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):
    """A simple 1D CNN for tabular data."""
    def __init__(self, num_features=21, num_classes=2): # Example num_features
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the flattened size dynamically or set manually
        # Example: (num_features / 2 / 2) * 32
        # For num_features=21, after one pool: floor(21/2)=10. After two pools: floor(10/2)=5
        self.fc1 = nn.Linear(32 * (num_features // 2 // 2), 64) 
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Add a channel dimension for Conv1d: (batch_size, num_features) -> (batch_size, 1, num_features)
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
