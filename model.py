"""Baseline CNN architecture for FakeWave, shared by train.py and predict.py
(kept in its own module so predict.py never needs to import the trainer)."""

import torch.nn as nn


class DeepfakeCNN(nn.Module):
    """Small conv stack over log-mel input. A baseline, not tuned -- the
    point is confirming the pipeline feeds a model that learns at all."""

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)  # raw logits
