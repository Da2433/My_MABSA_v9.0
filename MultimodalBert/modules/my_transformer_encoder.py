from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn

class my_transformer_encoder(nn.Module):
    def __init__(self, args, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5,norm_first=False) -> None:
        super(my_transformer_encoder, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim=args.att_hidden_size,
                                            num_heads=args.num_head, batch_first=True, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(args.att_hidden_size, args.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(args.dim_feedforward, args.att_hidden_size)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.conv1 = nn.Conv1d(in_channels=args.att_hidden_size, out_channels=args.dim_feedforward, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=args.dim_feedforward, out_channels=args.att_hidden_size, kernel_size=1)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(my_transformer_encoder, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None) -> Tensor:

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            # x = self.norm1(x)
            x = self.norm2(x + self._ff_block(x))
            # x = self.norm2(x + x)
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Tensor = None, key_padding_mask: Tensor = None) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))  ### 32 128  11
        # x = self.dropout(self.conv2(x).transpose(-1, 1))  # 32 11 128
        return self.dropout2(x)