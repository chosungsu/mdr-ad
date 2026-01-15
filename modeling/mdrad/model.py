from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import attribute_matrix


# ----------------------
# Positional Embedding (TCAD 스타일)
# ----------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ----------------------
# Transformer Global Encoder (TCAD 스타일)
# ----------------------
class TransformerGlobal(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model) -> (seq, batch, d_model)
        out = self.transformer(x.permute(1, 0, 2))
        return out.permute(1, 0, 2)


# ----------------------
# ResNet Local Encoder (TCAD 스타일)
# ----------------------
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual_conv is None else self.residual_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class ResNetLocal(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for h in hidden_dims:
            layers.append(ResNetBlock(in_ch, h))
            in_ch = h
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, dim) -> (batch, dim, seq)
        y = self.network(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)


# ----------------------
# Correlation (Attribute Matrix) Encoder/Decoder (간단 버전)
# - MSCVAE의 attribute matrix 개념을 사용하되, 시퀀스 평균 행렬 1장을 사용 (계산 단순화)
# ----------------------
class CorrEncoder(nn.Module):
    def __init__(self, dim: int, d_model: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * dim * dim, d_model)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # m: (batch, dim, dim) -> (batch, 1, dim, dim)
        h = self.net(m.unsqueeze(1))
        return self.fc(h.flatten(start_dim=1))


class CorrDecoder(nn.Module):
    def __init__(self, dim: int, d_model: int):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(d_model, dim * dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        return out.view(z.size(0), self.dim, self.dim)


# ----------------------
# Fusion (attention/gating) over z1/z2/z3
# ----------------------
class LatentAttentionFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # z_list: [z1, z2, z3], each (batch, d_model)
        zs = torch.stack(z_list, dim=1)  # (batch, 3, d_model)
        logits = self.score(zs)  # (batch, 3, 1)
        w = torch.softmax(logits.squeeze(-1), dim=1)  # (batch, 3)
        z = torch.sum(zs * w.unsqueeze(-1), dim=1)  # (batch, d_model)
        return z, w


# ----------------------
# Raw decoder: z -> (seq_len, input_dim)
# ----------------------
class RawDecoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, input_dim: int, hidden: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len * input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(z.size(0), self.seq_len, self.input_dim)


class MDRAD(nn.Module):
    """
    MDRAD: TCAD(전역/지역) + MSCVAE(상관구조) 특징을 결합한 다중 스트림 이상탐지 모델(최소 구현).
    - 입력: x (batch, seq_len, input_dim)
    - 출력:
      - x_hat: (batch, seq_len, input_dim) 원본 재구성
      - m_hat: (batch, input_dim, input_dim) 평균 attribute matrix 재구성
      - z1, z2, z3: (batch, d_model) 전역/지역/상관 latent
      - z: (batch, d_model) 융합 latent
      - w: (batch, 3) 융합 가중치
    """

    def __init__(
        self,
        *,
        seq_len: int,
        input_dim: int,
        d_model: int = 128,
        transformer_layers: int = 3,
        nhead: int = 4,
        resnet_dims: list[int] | None = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)

        resnet_dims = resnet_dims or [d_model, d_model]

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len=seq_len)
        self.global_encoder = TransformerGlobal(d_model, nhead, transformer_layers)
        self.local_encoder = ResNetLocal(d_model, resnet_dims)

        self.corr_encoder = CorrEncoder(dim=input_dim, d_model=d_model)
        self.fusion = LatentAttentionFusion(d_model=d_model)

        self.raw_decoder = RawDecoder(d_model=d_model, seq_len=seq_len, input_dim=input_dim)
        self.corr_decoder = CorrDecoder(dim=input_dim, d_model=d_model)

        self.project = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq, dim)
        emb = self.pos_emb(self.embedding(x))
        z1_seq = self.global_encoder(emb)  # (batch, seq, d_model)
        z2_seq = self.local_encoder(emb)   # (batch, seq, d_model)

        # pool to sequence-level representations
        z1 = z1_seq.mean(dim=1)
        z2 = z2_seq.mean(dim=1)

        # correlation (attribute matrix) branch: average over time
        m_seq = attribute_matrix(x)  # (batch, seq, dim, dim)
        m_avg = m_seq.mean(dim=1)    # (batch, dim, dim)
        z3 = self.corr_encoder(m_avg)  # (batch, d_model)

        z_fused, w = self.fusion([z1, z2, z3])
        z = self.project(z_fused)

        x_hat = self.raw_decoder(z)
        m_hat = self.corr_decoder(z)
        return x_hat, m_hat, m_avg, z1, z2, z3, z, w

