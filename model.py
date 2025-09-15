from typing import NamedTuple

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    top_indices: Tensor


class ForwardOutput(NamedTuple):
    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor
    loss: Tensor


class TopKSAEConfig(PretrainedConfig):
    model_type = "topk_sae"
    def __init__(
        self,
        d_in: int = 2048,
        num_latents: int = 32768,
        k: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.num_latents = num_latents
        self.k = k


class TopKSAE(PreTrainedModel):
    config_class = TopKSAEConfig

    def __init__(self, cfg: TopKSAEConfig):
        super().__init__(cfg)
        self.d_in = cfg.d_in
        self.num_latents = cfg.num_latents
        self.k = cfg.k

        self.decoder = nn.Linear(self.num_latents, self.d_in, bias=False)
        self.set_decoder_norm_to_unit_norm()

        self.encoder = nn.Linear(self.d_in, self.num_latents)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

    def encode(self, x: torch.Tensor, return_topk: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        topk_acts_BK = post_topk.values
        topk_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=topk_indices_BK, src=topk_acts_BK)

        if return_topk:
            return (
                encoded_acts_BF,
                topk_acts_BK,
                topk_indices_BK,
            )
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor):
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_output = self.encode(x, return_topk=output_features)
        if output_features:
            encoded_acts_BF, topk_acts_BK, topk_indices_BK = encoded_output
        else:
            encoded_acts_BF = encoded_output

        x_hat_BD = self.decode(encoded_acts_BF)

        if output_features:
            return (
                x_hat_BD,
                topk_acts_BK,
                topk_indices_BK,
            )
        else:
            return x_hat_BD

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps
