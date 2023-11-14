from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn

class GatingMechanism(nn.Module):
    def __init__(self, args, batch_first = True):
        super().__init__()
        # self.x_linear = Linear(, 1)
        # self.x_linear = Linear(batch_len, 1)

        self.batch_first = batch_first
        self.fc_img = nn.Linear(args.att_hidden_size * 2, 1)

        # self.fc_img_x = Linear(args.gating_dim, 128)

    def forward(self, x, grid_img_features):
        if self.batch_first:
            x = x.transpose(0, 1)
            grid_img_features = grid_img_features.transpose(0, 1)

        grid_img_features = torch.mean(grid_img_features, dim=0, keepdim=True)  ## 1*batch*dim
        t, b, c = x.shape
        grid_img_features = grid_img_features.expand(t, b, c)
        merge = torch.cat([x, grid_img_features], dim=-1)

        gate = torch.sigmoid(self.fc_img(merge))  # T B C
        img_features = torch.mul(gate, x)
        return img_features, gate


class MM_gating(nn.Module):
    def __init__(self, args):
        super(MM_gating, self).__init__()
        self.xlin = nn.Linear(args.att_hidden_size, 1)
        self.ylin = nn.Linear(args.att_hidden_size, 1)


    def forward(self, modal1, modal2):
        x = self.xlin(modal1)
        y = self.ylin(modal2)
        merge = x + y
        gate = torch.sigmoid(merge)
        fusion_feat = torch.mul(gate, modal1) + torch.mul((1-gate), modal2)
        return fusion_feat

