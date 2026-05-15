"""
LSTM model architecture for dynamic sign language classification (J, Z).
"""
import torch
import torch.nn as nn


class DynamicSignLSTM(nn.Module):
    """
    Bidirectional LSTM for recognizing dynamic sign language letters.
    Input: (batch, sequence_length, features) = (batch, 30, 63)
    Output: (batch, num_classes) = (batch, 2)
    """

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: LSTM encoding, take last output, classify."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)
