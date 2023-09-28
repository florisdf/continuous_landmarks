from torch import nn


class PositionEncoder(nn.Module):
    def __init__(
        self,
        encoding_size=64,
    ):
        super().__init__()
        self.encoding_size = encoding_size
        self.model = nn.Sequential(
            nn.Linear(in_features=3, out_features=encoding_size),
            nn.GELU(),
            nn.Linear(in_features=encoding_size, out_features=encoding_size),
            nn.GELU(),
        )

    def forward(self, position):
        return self.model(position)