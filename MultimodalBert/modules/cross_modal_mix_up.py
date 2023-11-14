import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import random

class my_mixed_up(nn.Module):
    def __init__(self, args, layer_norm_eps=1e-5):
        super(my_mixed_up, self).__init__()
        self.norm1 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)

    def my_multimodal_mix(self, current_input: Tensor, cross_input: Tensor):
        '''
        beta 分布来将两种相同length的模态融合
        '''
        alpha = torch.tensor([random.betavariate(1, 1) for _ in range(current_input.size(1))]).unsqueeze(0).unsqueeze(-1).type_as(current_input)
        mixed_x = alpha * current_input + (1 - alpha) * cross_input
        mixed_x = self.norm1(mixed_x + current_input)
        return mixed_x

