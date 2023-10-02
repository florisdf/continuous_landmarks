from typing import Literal

import torch
from torch import nn

from .xca import XCABlock


class LandmarkPredictor(nn.Module):
    def __init__(
        self,
        query_size: int,
        feature_size: int,
        model_name: Literal['MLP', 'Transformer'],
    ):
        super().__init__()
        self.model_name = model_name

        if model_name == 'MLP':
            self.model = MLPLandmarkPredictor(query_size, feature_size)
        elif model_name == 'Transformer':
            self.model = TransformerLandmarkPredictor(query_size, feature_size)
        else:
            raise ValueError(f'Unknown model name "{model_name}"')

    def forward(self, query_sequence, feature):
        if self.model_name == 'MLP':
            B, N, _ = query_sequence.shape
            query = query_sequence.flatten(end_dim=1)
            feature = feature[:, None, :].expand(-1, N, -1).flatten(end_dim=1)

            lm_pred, var_pred = self.model(query, feature)
            lm_pred = lm_pred.unflatten(0, (B, N))
            var_pred = var_pred.unflatten(0, (B, N))
        else:
            lm_pred, var_pred = self.model(query_sequence, feature)

        return lm_pred, var_pred


class MLPLandmarkPredictor(nn.Module):
    def __init__(
        self,
        query_size: int,
        feature_size: int,
        hidden_size: int = 512
    ):
        super().__init__()
        self.query_size = query_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(in_features=query_size + feature_size, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=3),
        )

    def forward(self, query, feature):
        x = torch.cat([query, feature], dim=1)
        out = self.model(x)
        lm_pred = out[..., :2]
        var_pred = torch.exp(out[..., 2])
        return lm_pred, var_pred


class TransformerLandmarkPredictor(nn.Module):
    def __init__(
        self,
        query_size: int,
        feature_size: int,
        hidden_size: int = 768
    ):
        super().__init__()
        self.query_size = query_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        self.pre_layer = nn.Linear(
            in_features=query_size + feature_size,
            out_features=hidden_size
        )
        self.xca_blocks = nn.ModuleList([
            XCABlock(dim=hidden_size, lpi_kernel_size=1)
            for _ in range(4)
        ])
        self.post_layer = nn.Linear(
            in_features=hidden_size,
            out_features=3
        )

    def forward(self, query_sequence, feature):
        # query_sequence: B x N x 64
        # feature: B x 768
        B, N, _ = query_sequence.shape

        tokens = torch.cat([
            query_sequence,
            feature[:, None, :].expand(-1, N, -1)
        ], dim=-1)

        # Pass tokens to pre-layer
        # tokens: B x N x (64 + 768)
        tokens = self.pre_layer(tokens.flatten(end_dim=1)).unflatten(0, (B, N))

        # Pass tokens to transformer
        H = tokens.shape[-2]
        W = 1
        for xca_block in self.xca_blocks:
            tokens = xca_block(tokens, H, W)

        # Pass tokens to post-layer to get sequence of predicted landmarks
        out = self.post_layer(tokens.flatten(end_dim=1)).unflatten(0, (B, N))

        lm_pred = out[..., :2]
        var_pred = torch.exp(out[..., 2])

        return lm_pred, var_pred
