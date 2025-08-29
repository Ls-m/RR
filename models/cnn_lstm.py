import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    """Hybrid CNN-LSTM model for PPG to respiratory waveform estimation."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(CNN_LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN feature extractor
        self.cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_size // 2, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # CNN feature extraction
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Transpose for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Output projection
        output = self.output_layers(lstm_out)
        
        # Transpose back to (batch_size, 1, sequence_length)
        output = output.transpose(1, 2)
        
        return output


class AttentionCNN_LSTM(nn.Module):
    """CNN-LSTM with attention mechanism for better performance."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(AttentionCNN_LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-scale CNN feature extractor
        self.conv_blocks = nn.ModuleList([
            # Scale 1: Fine-grained features
            nn.Sequential(
                nn.Conv1d(1, hidden_size // 4, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Scale 2: Medium-grained features
            nn.Sequential(
                nn.Conv1d(1, hidden_size // 4, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Scale 3: Coarse-grained features
            nn.Sequential(
                nn.Conv1d(1, hidden_size // 4, kernel_size=15, padding=7),
                nn.BatchNorm1d(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Scale 4: Very coarse features
            nn.Sequential(
                nn.Conv1d(1, hidden_size // 4, kernel_size=31, padding=15),
                nn.BatchNorm1d(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
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
        
        # Multi-scale CNN feature extraction
        conv_features = []
        for conv_block in self.conv_blocks:
            conv_features.append(conv_block(x))
        
        # Concatenate multi-scale features
        x = torch.cat(conv_features, dim=1)
        
        # Feature fusion
        x = self.feature_fusion(x)
        
        # Transpose for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        x = lstm_out + attended_out
        
        # Output decoding
        output = self.decoder(x)
        
        # Transpose back to (batch_size, 1, sequence_length)
        output = output.transpose(1, 2)
        
        return output
