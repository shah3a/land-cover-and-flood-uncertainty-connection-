import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, n_features: int, hidden_channels: int = 32, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x).squeeze(1)
