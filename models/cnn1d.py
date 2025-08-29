import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """1D CNN model for PPG to respiratory waveform estimation."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(CNN1D, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Encoder layers
        self.encoder = nn.ModuleList()
        
        # First layer
        self.encoder.append(nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # Middle layers
        for i in range(num_layers - 2):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder layers
        self.decoder = nn.ModuleList()
        
        # First decoder layer
        self.decoder.append(nn.Sequential(
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # Middle decoder layers
        for i in range(num_layers - 2):
            self.decoder.append(nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output layer
        self.output_layer = nn.Conv1d(hidden_size, 1, kernel_size=7, padding=3)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for layer in self.decoder:
            x = layer(x)
        
        # Output
        x = self.output_layer(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for improved CNN architecture."""
    
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class ResidualCNN1D(nn.Module):
    """Residual 1D CNN model for better performance."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(ResidualCNN1D, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Conv1d(hidden_size, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output
        x = self.output_layer(x)
        
        return x
