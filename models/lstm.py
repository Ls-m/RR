import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM model for PPG to respiratory waveform estimation."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(1, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        # Transpose to (batch_size, sequence_length, 1)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Output projection
        output = self.output_proj(lstm_out)
        
        # Transpose back to (batch_size, 1, sequence_length)
        output = output.transpose(1, 2)
        
        return output


class BiLSTM(nn.Module):
    """Bidirectional LSTM with attention mechanism."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(BiLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        batch_size, _, seq_len = x.shape
        
        # Transpose to (batch_size, sequence_length, 1)
        x = x.transpose(1, 2)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = lstm_out * attention_weights
        
        # Output projection
        output = self.output_layers(attended_output)
        
        # Transpose back to (batch_size, 1, sequence_length)
        output = output.transpose(1, 2)
        
        return output
