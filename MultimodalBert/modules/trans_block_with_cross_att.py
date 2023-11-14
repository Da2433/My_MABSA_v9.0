from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn


class TransWithCrossAtt(nn.Module):
    def __init__(self, args, dropout = 0.1,
                 activation = F.relu,
                 layer_norm_eps=1e-5,
                 norm_first=False,):
        super(TransWithCrossAtt, self).__init__()
        self.cross_att = MultiheadAttention(embed_dim=args.att_hidden_size,
                                            num_heads=args.num_head, batch_first=True, dropout=dropout)

        self.linear1 = nn.Linear(args.att_hidden_size, args.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(args.dim_feedforward, args.att_hidden_size)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, query: Tensor,
                key_value: Tensor,
                att_mask = None,
                key_padding_mask = None):


        x = query
        if self.norm_first:
            x = x + self._crossatt_block(self.norm1(x), key_value, att_mask, key_padding_mask)
            # x = x + self._ff_block(self.norm2(x))
        else:
            # x = self.norm1(x + self._crossatt_block(x, key_value, att_mask, key_padding_mask))
            x = self.norm1(x + self._crossatt_block(self.norm1(x), key_value, att_mask, key_padding_mask))
#            x = self.norm2(x + self._ff_block(x))

        return x

    # cross-attention block
    def _crossatt_block(self, x: Tensor,
                        key_value: Tensor,
                        attn_mask,
                        key_padding_mask):
        x = self.cross_att(x, key_value, key_value,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)