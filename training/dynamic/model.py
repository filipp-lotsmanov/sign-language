"""
LSTM Model for Dynamic Sign Language Recognition (J/Z)
"""

import torch
import torch.nn as nn


class DynamicSignLSTM(nn.Module):
    """
    LSTM model for recognizing dynamic sign language letters.
    Input: (batch, sequence_length, features) = (batch, 30, 63)
    Output: (batch, num_classes) = (batch, 2)
    """
    
    def __init__(
        self,
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        x: (batch, seq_len, features)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last output (or use attention)
        # lstm_out: (batch, seq_len, hidden*2)
        last_output = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Classification
        out = self.fc(last_output)
        return out