from pulseppg.nets.CrossAttn.utils.DilatedConv import dilated_conv_net
from pulseppg.nets.CrossAttn.utils.RevIN import RevIN
from pulseppg.nets.CrossAttn.utils.functional_mod import multi_head_attention_forward
import torch
from torch import nn

# import torch.nn.functional as F # .multi_head_attention_forward


class Net(torch.nn.Module):
    """Transformer language model."""

    def __init__(
        self,
        input_dims=1,
        query_dimsize=None,
        key_dimsize=None,
        embed_dim=256,
        sparsemax=False,
        kernel_size=15,
        double_receptivefield=5,
        stride=1,
    ):
        super().__init__()
        # import pdb; pdb.set_trace()
        self.embed_dim = embed_dim
        self.stride = stride
        self.sparsemax = sparsemax

        # dilated convs used
        if query_dimsize is None:
            query_dimsize = input_dims

        self.revin = RevIN(num_features=query_dimsize)

        self.q_func = dilated_conv_net(
            in_channel=query_dimsize,
            out_channel=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck=embed_dim // 8,
            double_receptivefield=double_receptivefield,
        )
        if key_dimsize is None:
            key_dimsize = input_dims
        self.k_func = dilated_conv_net(
            in_channel=key_dimsize,
            out_channel=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck=embed_dim // 8,
            double_receptivefield=double_receptivefield,
        )
        self.v_func = dilated_conv_net(
            in_channel=key_dimsize,
            out_channel=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck=embed_dim // 8,
            double_receptivefield=double_receptivefield,
        )

        # identity matrix because we are already using convs for in_projections
        self.in_proj_weight = torch.concat(
            (
                torch.eye(self.embed_dim),
                torch.eye(self.embed_dim),
                torch.eye(self.embed_dim),
            )
        ).requires_grad_(False)
        self.out_proj = nn.Linear(embed_dim, query_dimsize)

    def forward(self, query_in, key_in, mask=None):
        # these inputs are N L C but should really not be ...
        if self.revin:
            key_in = self.revin(key_in, "norm").transpose(1, 2)
            query_in = self.revin(query_in, "norm", recalc_stats=False).transpose(
                1, 2
            )  # batch_size, num_features, sequence_length,
        else:
            key_in = key_in.transpose(1, 2)
            query_in = query_in.transpose(1, 2)

        if mask is not None:
            mask = mask.transpose(1, 2)
        q_out = self.q_func(query_in, mask).permute(2, 0, 1)  # Time, Batch, Channel
        k_out = self.k_func(key_in).permute(2, 0, 1)
        v_out = self.v_func(key_in).permute(2, 0, 1)

        reconstruction, attn_weights = multi_head_attention_forward(
            query=q_out,
            key=k_out,
            value=v_out,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            in_proj_weight=self.in_proj_weight.to(q_out.device),
            need_weights=self.training,
            sparsemax=self.sparsemax,
            ### can ignore everything else, which is just default values used to make function work ###
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            embed_dim_to_check=self.embed_dim,
            num_heads=1,
            use_separate_proj_weight=False,
            add_zero_attn=False,
            dropout_p=0.1,
            training=self.training,
        )

        reconstruction = reconstruction.permute(1, 0, 2)
        reconstruction = self.revin(
            reconstruction, "denorm"
        )  # shape [batch_size, length, embed_dim]

        return reconstruction, attn_weights
