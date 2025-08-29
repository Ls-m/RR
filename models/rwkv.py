import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class RWKVBlockV5(nn.Module):
#     """RWKV-v5 block (multi-head style) for time series."""
#     def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = None, dropout: float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.head_dim = d_model // n_heads
#         self.d_ff = d_ff or 4 * d_model

#         # Multi-head projections
#         self.key = nn.Linear(d_model, d_model, bias=False)
#         self.value = nn.Linear(d_model, d_model, bias=False)
#         self.receptance = nn.Linear(d_model, d_model, bias=False)

#         # Per-head decay params
#         self.time_decay = nn.Parameter(torch.zeros(n_heads, self.head_dim))
#         self.time_first = nn.Parameter(torch.zeros(n_heads, self.head_dim))

#         self.output = nn.Linear(d_model, d_model, bias=False)

#         # Channel mixing
#         self.fc1 = nn.Linear(d_model, self.d_ff, bias=False)
#         self.fc2 = nn.Linear(self.d_ff, d_model, bias=False)
#         self.receptance_c = nn.Linear(d_model, d_model, bias=False)

#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, state=None):
#         B, T, C = x.shape
#         x_ln = self.ln1(x)

#         k = self.key(x_ln).view(B, T, self.n_heads, self.head_dim)
#         v = self.value(x_ln).view(B, T, self.n_heads, self.head_dim)
#         r = self.receptance(x_ln).view(B, T, self.n_heads, self.head_dim)

#         wkv = self._compute_wkv_v5(k, v, r)
#         x = x + self.dropout(self.output(wkv.view(B, T, C)))

#         # Channel mixing
#         x_ln2 = self.ln2(x)
#         hidden = F.relu(self.fc1(x_ln2)) ** 2
#         r_c = torch.sigmoid(self.receptance_c(x_ln2))
#         x = x + self.dropout(r_c * self.fc2(hidden))

#         new_state = x[:, -1]
#         return x, new_state

#     def _compute_wkv_v5(self, k, v, r):
#         """Multi-head RWKV-v5 style WKV."""
#         B, T, H, D = k.shape
#         outputs = []
#         decay = torch.exp(-torch.exp(self.time_decay))  # (H, D)

#         state = torch.zeros(B, H, D, device=k.device, dtype=k.dtype)
#         for t in range(T):
#             k_t, v_t, r_t = k[:, t], v[:, t], r[:, t]
#             state = decay * state + torch.exp(self.time_first + k_t) * v_t
#             out_t = torch.sigmoid(r_t) * state
#             outputs.append(out_t.unsqueeze(1))
#         return torch.cat(outputs, dim=1)  # (B, T, H, D)

# class CrossAttention(nn.Module):
#     def __init__(self, d_model: int, n_heads: int = 8):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.head_dim = d_model // n_heads
        
#         assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
#         self.query = nn.Linear(d_model, d_model)
#         self.key = nn.Linear(d_model, d_model)
#         self.value = nn.Linear(d_model, d_model)
#         self.out = nn.Linear(d_model, d_model)
        
#     def forward(self, query, key_value):
#         # query is from time branch: (B, T, C)
#         # key_value is from freq branch: (B, F, C)
#         B, T, C = query.shape
#         _, seq_len_freq, _ = key_value.shape

#         Q = self.query(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
#         K = self.key(key_value).view(B, seq_len_freq, self.n_heads, self.head_dim).transpose(1, 2)
#         V = self.value(key_value).view(B, seq_len_freq, self.n_heads, self.head_dim).transpose(1, 2)

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn_weights = F.softmax(scores, dim=-1)  # now F is torch.nn.functional again
        
#         # Apply attention to values
#         context = torch.matmul(attn_weights, V)
#         context = context.transpose(1, 2).contiguous().view(B, T, C)
        
#         return self.out(context)


# class RWKVBlock(nn.Module):
#     """RWKV block for time series processing."""
    
#     def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.d_ff = d_ff or 4 * d_model
        
#         # Time mixing (attention-like mechanism)
#         self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
#         self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
#         self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
#         self.key = nn.Linear(d_model, d_model, bias=False)
#         self.value = nn.Linear(d_model, d_model, bias=False)
#         self.receptance = nn.Linear(d_model, d_model, bias=False)
#         self.output = nn.Linear(d_model, d_model, bias=False)
        
#         # Channel mixing (FFN-like mechanism)
#         self.channel_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
#         self.channel_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
#         self.key_c = nn.Linear(d_model, self.d_ff, bias=False)
#         self.receptance_c = nn.Linear(d_model, d_model, bias=False)
#         self.value_c = nn.Linear(self.d_ff, d_model, bias=False)
        
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, state=None):
#         B, T, C = x.shape
        
#         # Time mixing
#         x_tm = self.ln1(x)
        
#         if state is None:
#             # Initialize state for the first step
#             state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        
#         # Shift and mix with previous state
#         x_shifted = torch.cat([state.unsqueeze(1), x_tm[:, :-1]], dim=1)
        
#         k = self.key(x_tm * self.time_mix_k + x_shifted * (1 - self.time_mix_k))
#         v = self.value(x_tm * self.time_mix_v + x_shifted * (1 - self.time_mix_v))
#         r = self.receptance(x_tm * self.time_mix_r + x_shifted * (1 - self.time_mix_r))
        
#         # WKV computation (simplified RWKV attention)
#         wkv = self._compute_wkv(k, v, r)
#         x = x + self.dropout(self.output(wkv))
        
#         # Channel mixing
#         x_cm = self.ln2(x)
#         x_shifted_c = torch.cat([state.unsqueeze(1), x_cm[:, :-1]], dim=1)
        
#         k_c = self.key_c(x_cm * self.channel_mix_k + x_shifted_c * (1 - self.channel_mix_k))
#         r_c = self.receptance_c(x_cm * self.channel_mix_r + x_shifted_c * (1 - self.channel_mix_r))
        
#         vk = self.value_c(F.relu(k_c) ** 2)
#         x = x + self.dropout(torch.sigmoid(r_c) * vk)
        
#         # Update state for next step
#         new_state = x[:, -1]
        
#         return x, new_state
    
#     def _compute_wkv(self, k, v, r):
        
#         """
#         Improved WKV (weighted key-value) computation with better numerical stability
#         and more efficient recurrent processing.
#         """
#         B, T, C = k.shape
#         device = k.device
#         dtype = k.dtype
        
#         # Initialize state for recurrent computation
#         wkv_state = torch.zeros(B, C, device=device, dtype=dtype)
#         wkv_outputs = []
        
#         # Process each time step recurrently for better temporal modeling
#         for t in range(T):
#             k_t = k[:, t, :]  # (B, C)
#             v_t = v[:, t, :]  # (B, C)
#             r_t = r[:, t, :]  # (B, C)
            
#             # Improved attention mechanism with exponential decay
#             # Use learnable decay parameter for better adaptation
#             decay = torch.sigmoid(k_t)  # Adaptive decay based on key
            
#             # Update state with exponential moving average
#             wkv_state = decay * wkv_state + (1 - decay) * v_t
            
#             # Apply receptance gating with improved activation
#             receptance = torch.sigmoid(r_t)
            
#             # Compute output with residual connection for better gradient flow
#             wkv_t = receptance * wkv_state + 0.1 * v_t  # Small residual connection
            
#             wkv_outputs.append(wkv_t.unsqueeze(1))
        
#         # Concatenate all time steps
#         wkv = torch.cat(wkv_outputs, dim=1)  # (B, T, C)
        
#         # Apply layer normalization for better stability
#         wkv = F.layer_norm(wkv, wkv.shape[-1:])
        
#         return wkv


# class RWKV(nn.Module):
#     """RWKV model for PPG to respiratory waveform estimation."""
    
#     def __init__(self, input_size: int, hidden_size: int = 256, 
#                  num_layers: int = 6, dropout: float = 0.1):
#         super().__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # Input projection
#         self.input_proj = nn.Sequential(
#             nn.Conv1d(1, hidden_size // 2, kernel_size=7, padding=3),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.ReLU(),
#             nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, padding=2),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # RWKV blocks
#         self.rwkv_blocks = nn.ModuleList([
#             RWKVBlock(hidden_size, dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         # Output projection
#         self.output_proj = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, hidden_size // 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 4, 1)
#         )
        
#         self.ln_out = nn.LayerNorm(hidden_size)
        
#     def forward(self, x):
#         # x shape: (batch_size, 1, sequence_length)
        
#         # Input feature extraction
#         x = self.input_proj(x)  # (B, hidden_size, T)
#         x = x.transpose(1, 2)   # (B, T, hidden_size)
        
#         # RWKV processing
#         state = None
#         for rwkv_block in self.rwkv_blocks:
#             x, state = rwkv_block(x, state)
        
#         x = self.ln_out(x)
        
#         # Output projection
#         output = self.output_proj(x)  # (B, T, 1)
#         output = output.transpose(1, 2)  # (B, 1, T)
        
#         return output

# class RWKV_DualBranch(nn.Module):
#     """RWKV model with dual branches for time and frequency domain processing."""
    
#     def __init__(self, input_size: int, hidden_size: int = 256, 
#                  num_layers: int = 6, dropout: float = 0.1):
#         super().__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # Time domain input projection
#         self.time_proj = nn.Sequential(
#             nn.Conv1d(1, hidden_size // 2, kernel_size=7, padding=3),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.ReLU(),
#             nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, padding=2),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Time domain RWKV blocks
#         self.time_rwkv = nn.ModuleList([
#             RWKVBlockV5(hidden_size, dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         # Frequency domain projection (magnitude and phase as 2 channels)
#         self.freq_proj = nn.Sequential(
#             nn.Conv1d(2, hidden_size // 2, kernel_size=1),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.ReLU(),
#             nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=1),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Frequency domain RWKV blocks
#         self.freq_rwkv = nn.ModuleList([
#             RWKVBlockV5(hidden_size, dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         self.cross_attention = CrossAttention(hidden_size, n_heads=8)

#         # Fusion normalization
#         self.fusion_norm = nn.LayerNorm(hidden_size)
        
#         # Output projection (same as original RWKV)
#         self.output_proj = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, hidden_size // 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 4, 1)
#         )
        
#         self.ln_out = nn.LayerNorm(hidden_size)
        
#     def forward(self, x):
#         # x shape: (batch_size, 1, sequence_length)
#         B, _, T = x.shape
        
#         # Time domain branch
#         time_x = self.time_proj(x)  # (B, hidden_size, T)
#         time_x = time_x.transpose(1, 2)  # (B, T, hidden_size)
        
#         state_time = None
#         for rwkv_block in self.time_rwkv:
#             time_x, state_time = rwkv_block(time_x, state_time)
        
#         # Frequency domain branch
#         x_squeeze = x.squeeze(1)  # (B, T)
#         x_fft = torch.fft.rfft(x_squeeze, dim=1)  # (B, F) where F = T//2 + 1, complex
#         mag = torch.abs(x_fft)  # (B, F)
#         phase = torch.atan2(x_fft.imag, x_fft.real)

#         freq_input = torch.stack([mag, phase], dim=1)  # (B, 2, F)
        
#         freq_x = self.freq_proj(freq_input)  # (B, hidden_size, F)
#         freq_x = freq_x.transpose(1, 2)  # (B, F, hidden_size)
        
#         state_freq = None
#         for rwkv_block in self.freq_rwkv:
#             freq_x, state_freq = rwkv_block(freq_x, state_freq)
        
#         # Get context from frequency features based on time features
#         freq_context = self.cross_attention(query=time_x, key_value=freq_x)
        
#         # Fuse with a residual connection
#         fused = time_x + freq_context 
#         fused = self.fusion_norm(fused)
#         fused = self.ln_out(fused)
        
#         # Output projection
#         output = self.output_proj(fused)  # (B, T, 1)
#         output = output.transpose(1, 2)  # (B, 1, T)
        
#         return output


# class RWKV_DualBranch(nn.Module):
#     """RWKV model with dual branches for time and frequency domain processing."""
    
#     def __init__(self, input_size: int, hidden_size: int = 256, 
#                  num_layers: int = 6, dropout: float = 0.1):
#         super().__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # Time domain input projection
#         self.time_proj = nn.Sequential(
#             nn.Conv1d(1, hidden_size // 2, kernel_size=7, padding=3),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.ReLU(),
#             nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, padding=2),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Time domain RWKV blocks
#         self.time_rwkv = nn.ModuleList([
#             RWKVBlock(hidden_size, dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         # Frequency domain projection (magnitude and phase as 2 channels)
#         self.freq_proj = nn.Sequential(
#             nn.Conv1d(2, hidden_size // 2, kernel_size=1),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.ReLU(),
#             nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=1),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Frequency domain RWKV blocks
#         self.freq_rwkv = nn.ModuleList([
#             RWKVBlock(hidden_size, dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         # Fusion normalization
#         self.fusion_norm = nn.LayerNorm(hidden_size)
        
#         # Output projection (same as original RWKV)
#         self.output_proj = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, hidden_size // 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 4, 1)
#         )
        
#         self.ln_out = nn.LayerNorm(hidden_size)
        
#     def forward(self, x):
#         # x shape: (batch_size, 1, sequence_length)
#         B, _, T = x.shape
        
#         # Time domain branch
#         time_x = self.time_proj(x)  # (B, hidden_size, T)
#         time_x = time_x.transpose(1, 2)  # (B, T, hidden_size)
        
#         state_time = None
#         for rwkv_block in self.time_rwkv:
#             time_x, state_time = rwkv_block(time_x, state_time)
        
#         # Frequency domain branch
#         x_squeeze = x.squeeze(1)  # (B, T)
#         x_fft = torch.fft.rfft(x_squeeze, dim=1)  # (B, F) where F = T//2 + 1, complex
#         mag = torch.abs(x_fft)  # (B, F)
#         phase = torch.angle(x_fft)  # (B, F)
#         freq_input = torch.stack([mag, phase], dim=1)  # (B, 2, F)
        
#         freq_x = self.freq_proj(freq_input)  # (B, hidden_size, F)
#         freq_x = freq_x.transpose(1, 2)  # (B, F, hidden_size)
        
#         state_freq = None
#         for rwkv_block in self.freq_rwkv:
#             freq_x, state_freq = rwkv_block(freq_x, state_freq)
        
#         # Global average pooling on frequency features to get (B, 1, hidden_size)
#         freq_global = freq_x.mean(dim=1, keepdim=True)  # (B, 1, hidden_size)
        
#         # Repeat frequency global features across time dimension
#         freq_repeated = freq_global.repeat(1, T, 1)  # (B, T, hidden_size)
        
#         # Fuse time and frequency features (additive fusion)
#         fused = time_x + freq_repeated
#         fused = self.fusion_norm(fused)
#         fused = self.ln_out(fused)
        
#         # Output projection
#         output = self.output_proj(fused)  # (B, T, 1)
#         output = output.transpose(1, 2)  # (B, 1, T)
        
#         return output

class ImprovedTransformer(nn.Module):
    """Improved Transformer model with better architecture for signal processing."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-scale input processing
        self.input_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        
        self.input_norm = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(input_size, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output decoder with skip connections
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Linear(hidden_size // 4, 1)
        ])
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        B, _, T = x.shape
        
        # Multi-scale feature extraction
        conv_features = []
        for conv in self.input_convs:
            conv_features.append(conv(x))
        
        # Combine features
        x = torch.cat(conv_features, dim=1)  # (B, hidden_size, T)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Transpose for transformer: (B, T, hidden_size)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        if T <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :T, :]
        else:
            # Handle sequences longer than max_len
            pe_extended = self.pos_encoding.repeat(1, (T // self.pos_encoding.size(1)) + 1, 1)
            x = x + pe_extended[:, :T, :]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Decoder with skip connections
        skip = x
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i == 0:  # Add skip connection after first layer
                x = x + skip[:, :, :x.size(-1)]
        
        # Transpose back: (B, 1, T)
        output = x.transpose(1, 2)
        
        return output


class WaveNet(nn.Module):
    """WaveNet-inspired model for signal-to-signal prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 10, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_conv = nn.Conv1d(1, hidden_size, kernel_size=1)
        
        # Dilated convolution layers
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** (i % 8)  # Cycle through dilations
            
            # Dilated convolution
            self.dilated_convs.append(
                nn.Conv1d(hidden_size, hidden_size * 2, kernel_size, 
                         dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
            )
            
            # Residual connection
            self.residual_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
            )
            
            # Skip connection
            self.skip_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
            )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size // 2, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # Input projection
        x = self.input_conv(x)
        
        # Skip connections accumulator
        skip_connections = 0
        
        # Process through dilated convolution layers
        for i in range(self.num_layers):
            # Store input for residual connection
            residual = x
            
            # Dilated convolution with gated activation
            conv_out = self.dilated_convs[i](x)
            tanh_out, sigmoid_out = conv_out.chunk(2, dim=1)
            gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
            
            # Residual connection
            x = self.residual_convs[i](gated) + residual
            
            # Skip connection
            skip_connections = skip_connections + self.skip_convs[i](gated)
        
        # Output processing
        output = self.output_layers(skip_connections)
        
        return output





class RWKV_TimeMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Learnable parameters
        self.w = nn.Parameter(torch.zeros(dim))     # decay rates
        self.u = nn.Parameter(torch.zeros(dim))     # output bias

        # Projections
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim, bias=False)

        # Mixing coefficient for x_t and x_{t-1}
        self.mu_r = nn.Parameter(torch.rand(dim))
        self.mu_k = nn.Parameter(torch.rand(dim))
        self.mu_v = nn.Parameter(torch.rand(dim))

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, dim)
        state: dict for recurrent memory {a, b, x_prev}
        """

        B, T, C = x.shape
        if state is None:
            state = {
                "a": torch.zeros(B, C, device=x.device),
                "b": torch.zeros(B, C, device=x.device),
                "x_prev": torch.zeros(B, C, device=x.device),
            }

        a, b, x_prev = state["a"], state["b"], state["x_prev"]
        outputs = []

        decay = torch.exp(-self.w)   # per-channel decay (positive)

        for t in range(T):
            xt = x[:, t, :]

            # Time mixing: blend current and previous input
            xt_r = self.mu_r * xt + (1 - self.mu_r) * x_prev
            xt_k = self.mu_k * xt + (1 - self.mu_k) * x_prev
            xt_v = self.mu_v * xt + (1 - self.mu_v) * x_prev

            r_t = torch.sigmoid(self.W_r(xt_r))
            k_t = self.W_k(xt_k)
            v_t = self.W_v(xt_v)

            exp_k = torch.exp(k_t)

            # === RWKV recurrence update (from paper) ===
            a = a * decay 
            b = b * decay 

            # add the "current boosted term" (exp(u + k_t) v_t)
            num = a + torch.exp(self.u + k_t) * v_t
            den = b + torch.exp(self.u + k_t)

            wkv_t = num / (den + 1e-8)  # (batch, dim)
            out_t = self.W_out(r_t * wkv_t)

            outputs.append(out_t.unsqueeze(1))
            x_prev = xt  # update x_prev

        state = {"a": a, "b": b, "x_prev": x_prev}
        return torch.cat(outputs, dim=1), state


class RWKV_ChannelMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # In this formulation, W_k and W_v are like the two layers of an FFN
        # W_r is the gate
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False) # This projects the activated k_t

        self.mu_r = nn.Parameter(torch.rand(dim))
        self.mu_k = nn.Parameter(torch.rand(dim))

    def forward(self, x, state=None):
        B, T, C = x.shape
        if state is None:
            state = {"x_prev": torch.zeros(B, C, device=x.device)}
        x_prev = state["x_prev"]

        outputs = []
        for t in range(T):
            xt = x[:, t, :]
            
            # Mix current and previous tokens
            xt_r = self.mu_r * xt + (1 - self.mu_r) * x_prev
            xt_k = self.mu_k * xt + (1 - self.mu_k) * x_prev

            # Calculate r and k
            r_t = torch.sigmoid(self.W_r(xt_r))
            k_t = self.W_k(xt_k)
            
            # === This is the corrected part ===
            # Apply squared ReLU activation to k_t
            k_activated = torch.square(torch.relu(k_t))
            
            # Project the activated k with W_v and multiply by the gate r_t
            out_t = r_t * self.W_v(k_activated)
            
            outputs.append(out_t.unsqueeze(1))
            x_prev = xt

        return torch.cat(outputs, dim=1), {"x_prev": x_prev}


class RWKV_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mixing = RWKV_TimeMix(dim)
        self.channel_mixing = RWKV_ChannelMix(dim)

    def forward(self, x, state=None):
        if state is None:
            state = {"time": None, "channel": None}

        # TimeMix + residual
        h, new_time_state = self.time_mixing(self.ln1(x), state["time"])
        x = x + h

        # ChannelMix + residual
        h, new_channel_state = self.channel_mixing(self.ln2(x), state["channel"])
        x = x + h

        return x, {"time": new_time_state, "channel": new_channel_state}


class RWKV(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, patch_size=90, stride=45):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.input_size = input_size
        self.norm = nn.InstanceNorm1d(1, affine=True)  # Univariate norm over time dim
        self.embedding = nn.Linear(self.patch_size, hidden_size)
        self.blocks = nn.ModuleList([RWKV_Block(hidden_size) for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.patch_size)  # Per-patch reconstruction

    def forward(self, x, state=None):
        B, T, C = x.shape
        if C == 1:
            time_dim = T
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            time_dim = C
            x = self.norm(x)

        x = x.unfold(-1, self.patch_size, self.stride)  # (B, T or channels, num_patches, patch_size)
        num_patches = x.shape[-2]
        x = x.reshape(B, num_patches, self.patch_size)  # (B, num_patches, patch_size)

        h = self.embedding(x)
        new_states = []
        for i, block in enumerate(self.blocks):
            h, st = block(h, state[i] if state else None)
            new_states.append(st)
        h = self.ln_out(h)
        out = self.head(h)  # (B, num_patches, patch_size)

        # Reconstruct with overlap averaging
        reconstructed = torch.zeros(B, time_dim, device=x.device)
        count = torch.zeros(B, time_dim, device=x.device)
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_size
            reconstructed[:, start:end] += out[:, i, :]
            count[:, start:end] += 1
        reconstructed /= (count + 1e-8)
        return reconstructed.unsqueeze(1)  # (B, 1, time_dim=240)