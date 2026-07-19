import torch
from torch import nn


class EncoderAttentionBlock(nn.Module):
    def __init__(self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        x = self.norm1(x)

        y = self.ffn(x)
        x = self.norm2(x)

        return x


class FusionEncoder(nn.Module):
    def __init__(self,
        input_dims,
        fused_dim: int = 256,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU()
            ) for dim in input_dims
        ])

        # CLS

        self.cls = nn.Parameter(
            torch.randn(1, 1, fused_dim),
        )

        self.encoder_embeddings = nn.Parameter(
            torch.randn(len(input_dims), fused_dim),
        )

        # Transformer block

        self.encoder_blocks = nn.ModuleList([
            EncoderAttentionBlock(
                fused_dim,
                hidden_dim,
                num_heads,
                dropout
            ) for _ in range(num_layers)
        ])

        # Out network

        self.output = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embeddings):
        # Project encoders independently
        # [
        #   (B, d1),
        #   (B, d2),
        #   ...,
        #   (B, dn)
        # ]

        tokens = [
            proj(e) for proj, e in zip(self.projections, embeddings)
        ]

        # [
        #   (B, fused),
        #   (B, fused),
        #   ...,
        #   (B, fused)
        # ]

        x = torch.stack(tokens, dim=1)  # (B, n, fused_dim)
        x = x + self.encoder_embeddings.unsqueeze(0)

        # Add CLS

        batch_size = x.size(0)
        cls = self.cls.expand(batch_size, -1, -1)

        x = torch.cat([cls, x], dim=1)

        # Attention

        for block in self.encoder_blocks:
            x = block(x)

        # Extract CLS

        x = x[:, 0]

        return self.output(x)
