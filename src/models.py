import torch.nn as nn

class SimpleMLP(nn.Module):
    """一个带有 Dropout 正则化的简单多层感知机。"""
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    """一个简单的一维卷积神经网络。"""
    def __init__(self, num_features, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((num_features // 4) * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension
        x = self.conv(x)
        return self.fc(x)
