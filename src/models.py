"""Small convolutional network for Fashion-MNIST classification."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Classify batches of 28-by-28 grayscale images into ten classes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(1568, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape (batch_size, 10)."""
        # The first block produces 16 feature maps of size 14 by 14.
        x = self.pool(self.relu(self.conv1(x)))

        # The second block produces 32 feature maps of size 7 by 7.
        x = self.pool(self.relu(self.conv2(x)))

        # Preserve the batch dimension while flattening 32 * 7 * 7 features.
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
