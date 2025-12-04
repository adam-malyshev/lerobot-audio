import sys

sys.path.append("")
import torch
import torch.nn.functional as F
from torch import nn
from .htsat import HTSATWrapper


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias"):
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        """Initialize a Batchnorm layer."""
        m.bias.data.fill_(0.0)
        m.weight.data.fill_(1.0)


def get_audio_encoder(name: str):
    if name == "DyMN":
        return DYMN
    elif name == "HTSAT":
        return HTSATWrapper
    else:
        raise Exception(
            "The audio encoder name {} is incorrect or not supported".format(name)
        )


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

        self.init_weight()

    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class AudioEncoder(nn.Module):
    def __init__(
        self,
        audioenc_name: str,
        d_in: int,
        d_out: int,
    ) -> None:
        super().__init__()

        audio_encoder = HTSATWrapper
        self.base = audio_encoder()
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = (
            out_dict["embedding"],
            out_dict["clipwise_output"],
        )
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output, out_dict
