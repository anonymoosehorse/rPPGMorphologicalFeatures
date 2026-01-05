import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    The standard sine/cosine positional encoding from the
    original Transformer paper, kept unchanged.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)                                   # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))                            # (d_model/2)
        pe = torch.zeros(max_len, 1, d_model)                                           # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerPPGHR(nn.Module):
    """
    End-to-end 1-D CNN + Transformer that maps a window of raw / pre-processed
    PPG samples to
        • a scalar heart-rate estimate (x_reg) and
        • an optional class prediction (x_cls) – e.g. rhythm class.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int = 256,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 4,
        dropout: float = 0.1,
        n_regression_targets: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()

        # ─── 1-D convolutional front-end ────────────────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv1d(1, 12,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ─── Transformer encoder ────────────────────────────────────────────────────
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=nlayers)

        # ─── Global pooling + heads ─────────────────────────────────────────────────
        self.avg_pool = nn.AdaptiveAvgPool1d(1)               # output shape: (N, C, 1)
        self.reg_head = nn.Linear(d_model, n_regression_targets)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    # --------------------------------------------------------------------- helpers
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------- forward
    def forward(self, x: Tensor):
        """
        x: (batch, seq_len)  – raw or band-pass-filtered PPG samples
        """
        # 1) Add a channel dim and run the conv stack
        x = x.unsqueeze(1)                   # (N, 1, L)
        x = self.conv(x)                     # (N, d_model, L)

        # 2) Transformer expects (L, N, d_model)
        x = x.permute(2, 0, 1)               # (L, N, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)      # (L, N, d_model)

        # 3) Back to (N, d_model, L) ➜ global-avg-pool ➜ flatten
        x = x.permute(1, 2, 0)               # (N, d_model, L)
        x = self.avg_pool(x).squeeze(-1)     # (N, d_model)

        # 4) Heads
        x_reg = self.reg_head(x)             # (N, n_regression_targets)
        x_cls = self.softmax(self.cls_head(x))

        return x_reg, x_cls
