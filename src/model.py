"""
PyTorch LSTM model for binary SPY direction prediction.

Architecture:
  Input  →  2-layer LSTM (hidden_size=64, dropout between layers)
         →  Dropout on final hidden state
         →  Linear(64 → 1)
         →  raw logit  (use BCEWithLogitsLoss during training)

The model outputs a single unnormalized logit per sample.
Apply sigmoid at inference to convert to a probability.
"""

import torch
import torch.nn as nn


class SPYLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch only applies inter-layer dropout; single-layer models ignore it
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last time-step's hidden state for classification
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden_size)
        out = self.dropout(last_hidden)
        logit = self.fc(out)                       # (batch, 1)
        return logit.squeeze(-1)                   # (batch,)
