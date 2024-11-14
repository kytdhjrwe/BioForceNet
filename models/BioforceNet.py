import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Feature extraction with TCN
class TCNEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 5), (1, 1), padding=(0, 2)),
            nn.Conv2d(40, 40, (1, 5), (1, 1), dilation=2, padding=(0, 4)),
            nn.Conv2d(40, 40, (1, 5), (1, 1), dilation=3, padding=(0, 6)),
            nn.Conv2d(40, 40, (4, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 10), (1, 2)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

# Multi-head attention for Transformers
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # Calculate attention weights
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            energy.mask_fill(~mask, float('-inf'))

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        # Output
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)

# Residual connection to stabilize gradients
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

# Feed-forward block for Transformer encoder
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# Transformer encoder block with residuals
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

# Stacked Transformer encoder
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])

# Prediction head for regression
class PredictionHead(nn.Module):
    def __init__(self, emb_size, pre_len):
        super().__init__()
        self.predhead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, pre_len)
        )

    def forward(self, x):
        return x, self.predhead(x)

# Complete model integrating feature extraction, temporal modeling, and prediction
class BioforceNet(nn.Sequential):
    def __init__(self, emb_size=40, depth=8, pre_len=1):
        super().__init__(
            TCNEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            PredictionHead(emb_size, pre_len)
        )



if __name__ == "__main__":
    model = BioforceNet()
    print(model)
