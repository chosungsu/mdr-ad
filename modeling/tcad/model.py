import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Positional Embedding
# ----------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ----------------------
# Transformer Global Encoder
# ----------------------
class TransformerGlobal(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, seq, dim)
        # Permute to (seq, batch, dim) for PyTorch transformer
        out = x.permute(1, 0, 2)
        out = self.transformer(out)
        return out.permute(1, 0, 2)  # back to (batch, seq, dim)


# ----------------------
# ResNet Block for Local Encoder
# ----------------------
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        # x: (batch, dim, seq)
        residual = x if self.residual_conv is None else self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


# ----------------------
# Local ResNet Encoder
# ----------------------
class ResNetLocal(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        in_ch = input_dim
        for h in hidden_dims:
            layers.append(ResNetBlock(in_ch, h))
            in_ch = h

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq, dim) -> we want (batch, dim, seq)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # back to (batch, seq, dim)
        return out.permute(0, 2, 1)


# ----------------------
# Decoder (Reconstruction)
# ----------------------
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


# ----------------------
# Full TCAD Model
# ----------------------
class TCAD(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        d_model=128,
        transformer_layers=3,
        nhead=4,
        # 로컬 인코더 출력도 글로벌 인코더(d_model)와 동일한 차원을 유지해야
        # z1 + z2가 가능하므로 128로 맞춘다.
        resnet_dims=[128, 128],
        decoder_dims=[128],
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len=seq_len)

        # Global & Local encoders
        self.global_encoder = TransformerGlobal(d_model, nhead, transformer_layers)
        self.local_encoder = ResNetLocal(d_model, resnet_dims)

        # Final Bottleneck
        self.project_z = nn.Linear(d_model, d_model)

        # Decoder
        self.decoder = Decoder(d_model, decoder_dims, output_dim=input_dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        emb = self.embedding(x)
        emb = self.pos_emb(emb)

        # Global
        z1 = self.global_encoder(emb)

        # Local
        z2 = self.local_encoder(emb)

        # Combined latent
        z = z1 + z2  # representation fusion

        # bottleneck
        z_proj = self.project_z(z)

        # Reconstruction
        x_hat = self.decoder(z_proj)

        return x_hat, z1, z2
