from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.multihead_attention import MultiheadAttention_Image
from MultimodalBert.modules.multihead_attention1 import MultiheadAttention_mask
from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn
import math

class MmFusionTrans_mask(nn.Module):
    def __init__(self, args, dropout = 0.1,
                 activation = F.relu,
                 layer_norm_eps=1e-5,
                 norm_first=False):
        super(MmFusionTrans_mask, self).__init__()



#        self.self_att = MultiheadAttention_mask(embed_dim=args.att_hidden_size,
#                                           num_heads=args.num_head, batch_first=True, dropout=dropout)

        # self.self_att = MultiheadAttention_Image(embed_dim=args.att_hidden_size,
        #                                               num_heads=args.num_head, dropout=dropout, self_attention=True)    ###改过
        #
        # self.cross_att1 = MultiheadAttention(embed_dim=args.att_hidden_size,
        #                                    num_heads=args.num_head, batch_first=True, dropout=dropout)
        self.cross_att = MultiheadAttention(embed_dim=args.att_hidden_size,
                                                  num_heads=args.num_head, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(args.att_hidden_size, args.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(args.dim_feedforward, args.att_hidden_size)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation




#############加入mask

    def getBinaryTensor(self, i, boundary):
        one_matrix = torch.ones_like(i)
        zero_matrix = torch.zeros_like(i)

        return torch.where(i > boundary, one_matrix, zero_matrix)

    def mask(self, x, src_img_features):

        # x = x.transpose(0, 1)  # batch * len * dim     MultiheadAttention_Image 需要
        # src_img_features = src_img_features.transpose(0, 1)  # batch * 49 * dim   MultiheadAttention_Image 需要
        ########  mask  img ######### batch * 49 * len
        mask_img = torch.bmm(src_img_features, x.transpose(1, 2)) / math.sqrt(128)
        mask_img = F.softmax(mask_img, dim=-1)

        # mask_matrix = torch.mean(mask_img, dim=2, keepdim=True).repeat(1,1,49)
        # mask_img = F.sigmoid(mask_img)
        # mask_img = self.getBinaryTensor(mask_img,0.015)

        ########  mask  txt ######### batch * len * 49
#        mask_txt = torch.bmm(x, src_img_features.transpose(1, 2)) / math.sqrt(128)
#        mask_txt = F.softmax(mask_txt, dim=-1)

        # mask_txt = F.sigmoid(mask_txt)

        # mask_txt = self.getBinaryTensor(mask_txt,0.015)

#        mask_matrix = torch.bmm(mask_img, mask_txt).cuda()
        mask_matrix = mask_img.cuda()
        # mask_matrix = F.sigmoid(mask_matrix)
        # mask_matrix_output = torch.bmm(mask_img, mask_txt).cuda()
        # mask_matrix = F.softmax(mask_matrix, dim=-1)

        # mask_matrix_alpha = torch.sort(mask_matrix[1].view(-1),dim=-1,descending=True).values[int(mask_matrix[0].view(-1).size()[0] * 0.7)].detach().cpu().numpy().tolist()
        # mask_matrix_alpha = torch.sort(mask_matrix.view(-1),dim=-1,descending=True).values[int(mask_matrix.view(-1).size()[0] * 0.8)].detach().cpu().numpy().tolist()

        # mask_matrix_output = self.getBinaryTensor(mask_matrix, -1)  ## 最好的值设置为0.02  ## 0.01其中decay = 0.1时结果特别好但是非正常停了
        mask_matrix_output = self.getBinaryTensor(mask_matrix, 0).eq(0)
        # mask_matrix_output1 = torch.full_like(mask_matrix,0.5).eq(0)
        # mask_matrix_output1
        # mask_matrix_output = []
        # for i in mask_matrix:
        #     mask_list = i.reshape(1, -1)  # Ascending    # or i.view(src_img_features.size(1) * src_img_features.size(1))
        #     mask_list = sorted(mask_list.squeeze().tolist())
        #     num_tmp = int(len(mask_list) * 0.15)
        #     mask_matrix_tmp = self.getBinaryTensor(i, mask_list[num_tmp])
        #     mask_matrix_output.append(mask_matrix_tmp.tolist())

        return mask_matrix_output.detach()







 #################







    def forward(self, former_input: Tensor,
                cross_input: Tensor,
                att_mask = None,
                former_padding_mask = None,
                cross_padding_mask = None):

        x = former_input
        if self.norm_first:
            self_out = x + self._selfatt_block(self.norm3(x), x, att_mask, former_padding_mask)
            cross_out = self._crossatt_block(self.norm1(x), cross_input, att_mask, cross_padding_mask)
            # x = self_out + self._ff_block(self.norm2(cross_out))
            x = self_out + self.norm2(cross_out)
        else:
            # self_out = self.norm3(x + self._selfatt_block(x, x, att_mask, former_padding_mask))
            # cross_out = self._crossatt_block(x, cross_input, att_mask, cross_padding_mask)
            # x = self.norm2(self_out + self._ff_block(cross_out))

            # x = x.transpose(0, 1)                ##MultiheadAttention_Image 需要
            # cross_input = cross_input.transpose(0, 1) #######改过   MultiheadAttention_Image 需要

            mask_tmp = self.mask(x, cross_input)  ###改过 加入mask-
            self_out = self.norm3(x + self._selfatt_block(self.norm3(x), x, mask_tmp, att_mask, former_padding_mask))
            # self_out = self.norm3(x + self._selfatt_block(self.norm3(x), x,  att_mask, former_padding_mask))
            # self_out = self.norm3(x + self._selfatt_block(self.norm3(x), x, mask_tmp, att_mask, former_padding_mask.eq(0))).transpose(0,1)
            #####

            # x = x.transpose(0, 1)                ##MultiheadAttention_Image 需要
            # cross_input = cross_input.transpose(0, 1) #######改过   MultiheadAttention_Image 需要






            # cross_out = (self._crossatt_block(self.norm1(self_out), cross_input, mask_tmp,  cross_padding_mask , att_mask))###改过
            # x = x.transpose(0, 1)  ##改回来
            # cross_input = cross_input.transpose(0, 1)  #######改回来
            cross_out = (self._crossatt_block(self.norm1(x), cross_input, att_mask, cross_padding_mask))
            # cross_out = cross_out1 + cross_out2

            x = self.norm2(self_out + cross_out)

        return x

    # cross-attention block
    def _crossatt_block(self, x: Tensor, key_value: Tensor, attn_mask, key_padding_mask):
        x = self.cross_att(x, key_value, key_value,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)



    # def _crossatt_block(self, x: Tensor, key_value: Tensor, mask_matrix_tmp,  key_padding_mask, attn_mask):   ###改过
    #     x = self.cross_att(x, key_value, key_value,
    #                        mask_matrix_tmp=mask_matrix_tmp,
    #                        key_padding_mask=key_padding_mask,
    #                        attn_mask=attn_mask,
    #                        need_weights=False)[0]
    #     return self.dropout1(x)

    # self-attention block
    # def _selfatt_block(self, x: Tensor,
    #                    key_value: Tensor,
    #                    attn_mask,
    #                    key_padding_mask):
    #     x = self.self_att(x, key_value, key_value,
    #                        attn_mask=attn_mask,
    #                        key_padding_mask=key_padding_mask,
    #                        need_weights=False)[0]
    #     return self.dropout3(x)


#####_selfatt_block_mask
    def _selfatt_block(self, x: Tensor,key_value: Tensor, mask_matrix_tmp, attn_mask, key_padding_mask):
        x = self.self_att(x, key_value, key_value,
                          mask_matrix_tmp=mask_matrix_tmp,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout3(x)
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)




from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn

class MmFusionTrans(nn.Module):
    def __init__(self, args, dropout = 0.1,
                 activation = F.relu,
                 layer_norm_eps=1e-5,
                 norm_first=False):
        super(MmFusionTrans, self).__init__()

        self.self_att = MultiheadAttention(embed_dim=args.att_hidden_size,
                                           num_heads=args.num_head, batch_first=True, dropout=dropout)

        self.cross_att = MultiheadAttention(embed_dim=args.att_hidden_size,
                                           num_heads=args.num_head, batch_first=True, dropout=dropout)

        self.linear1 = nn.Linear(args.att_hidden_size, args.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(args.dim_feedforward, args.att_hidden_size)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation


    def forward(self, former_input: Tensor,
                cross_input: Tensor,
                att_mask = None,
                former_padding_mask = None,
                cross_padding_mask = None):

        x = former_input
        if self.norm_first:
            self_out = x + self._selfatt_block(self.norm3(x), x, att_mask, former_padding_mask)
            cross_out = self._crossatt_block(self.norm1(x), cross_input, att_mask, cross_padding_mask)
            # x = self_out + self._ff_block(self.norm2(cross_out))
            x = self_out + self.norm2(cross_out)
        else:
            # self_out = self.norm3(x + self._selfatt_block(x, x, att_mask, former_padding_mask))
            # cross_out = self._crossatt_block(x, cross_input, att_mask, cross_padding_mask)
            # x = self.norm2(self_out + self._ff_block(cross_out))
            self_out = self.norm3(x + self._selfatt_block(self.norm3(x), x, att_mask, former_padding_mask))
            cross_out = self._crossatt_block(self.norm1(x), cross_input, att_mask, cross_padding_mask)
            x = self.norm2(self_out + cross_out)

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

    # self-attention block
    def _selfatt_block(self, x: Tensor,
                       key_value: Tensor,
                       attn_mask,
                       key_padding_mask):
        x = self.self_att(x, key_value, key_value,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout3(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

